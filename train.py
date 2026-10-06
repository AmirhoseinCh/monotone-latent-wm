#!/usr/bin/env python3
"""
Train a latent world model for image-goal navigation.

    Encoder:    frozen DINOv2 / DINOv3 / V-JEPA2           -> (B, T, 1, H, W, D)
    Predictor:  AdaLN ViT (ours) or ViT + action concat (DINO-WM baseline)
    Losses:     rollout loss (Base), + MCR ranking loss (MCR), + InfoNCE (ACT)

The training variant is selected entirely by the config file:
    configs/base.py    Base (rollout counter-curriculum)
    configs/mcr.py     MCR fine-tuning stage (resume from a Base checkpoint)
    configs/tf.py      single-step teacher forcing (ablation)
    configs/act.py     action-contrastive training (investigated variant)
    configs/dinowm.py  DINO-WM baseline

Usage:
    # Single GPU sanity check
    python train.py --config configs/base.py --encoder dino --debug

    # Multi-GPU
    torchrun --nproc_per_node 4 train.py --config configs/base.py --encoder dino --name base
"""

import datetime
import importlib.util
import io
import os
import queue
import shutil
import socket
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from absl import app, flags, logging
import tqdm
import wandb
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Prevent TF from allocating GPU memory and competing with PyTorch.
# TF data augmentation (brightness, contrast) would otherwise default to GPU.
tf.config.set_visible_devices([], "GPU")


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _setup_distributed():
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        torch.distributed.init_process_group(backend="nccl")

    from accelerate import PartialState
    return PartialState()



# ─── Dataset loading ─────────────────────────────────────────────────────────
from lmwm.data.dataset import make_interleaved_dataset
from lmwm.data.utils.data_utils import NormalizationType
from lmwm.data.utils.cast_transforms import gnm_action_angle_dataset_transform
from lmwm.data.spec import ModuleSpec
from lmwm.data.torch_rlds_dataset import TorchRLDSDataset, _np2pt, collate_fn

# ─── World model ─────────────────────────────────────────────────────────────
from lmwm.world_model import build_world_model

# ─── Flags ────────────────────────────────────────────────────────────────────
FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/base.py", "Config path")
flags.DEFINE_string("encoder", "dino", "Encoder type: dino (DINOv2) | dinov3 | vjepa")
flags.DEFINE_string("name", "wm", "W&B run name")
flags.DEFINE_bool("debug", False, "Quick sanity-check mode")
flags.DEFINE_string("save_dir", None, "Override config save_dir")
flags.DEFINE_string("data_dir", None, "Override config data_dir")
flags.DEFINE_float("grad_clip", None, "Override config grad_clip")
flags.DEFINE_string("resume", None, "Path to checkpoint to resume from")
flags.DEFINE_integer("rollout_steps", None, "Override config rollout_steps")
flags.DEFINE_integer("batch_size", None, "Override config batch_size")
flags.DEFINE_integer("num_steps", None, "Override config num_steps")
flags.DEFINE_float("lr", None, "Override config learning rate")
flags.DEFINE_bool("skip_optimizer_state", False, "Skip loading optimizer/scheduler state")
flags.DEFINE_bool("zero_actions", False, "Ablation: zero out actions")
flags.DEFINE_bool("shuffle_actions", False, "Ablation: shuffle actions across batch")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class PrefetchLoader:
    """Runs the data pipeline in a background thread so CPU preprocessing
    (TF pipeline) overlaps with GPU computation.

    Designed for infinite training iterables — the background thread fills a
    bounded queue; the main thread pulls from it with next().  Because it uses
    threading (not multiprocessing), the TF/RLDS dataset that cannot be pickled
    works fine; the GIL is released by TF's C++ I/O, so
    the thread genuinely runs in parallel with GPU kernels on the main thread.
    """

    def __init__(self, iterable, maxsize: int = 2):
        self._queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._fill, args=(iterable,), daemon=True
        )
        self._thread.start()

    def _fill(self, iterable):
        for batch in iterable:
            self._queue.put(batch)  # blocks when queue is full

    # Support both next(train_iter) and for-loop usage.
    def __iter__(self):
        return self

    def __next__(self):
        return self._queue.get()  # blocks until a batch is ready


def _to_device(data, device):
    if isinstance(data, dict):
        return {key: _to_device(val, device) for key, val in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data


def _tensor_stats(prefix: str, tensor: torch.Tensor) -> dict:
    if tensor is None:
        return {}
    t = tensor.float()
    return {
        f"{prefix}/mean": t.mean().item(),
        f"{prefix}/std": t.std().item(),
        f"{prefix}/rms": t.pow(2).mean().sqrt().item(),
        f"{prefix}/abs_max": t.abs().max().item(),
    }


def _make_sample_viz(images, actions, sample_idx=0):
    T = images.shape[1]
    frame_idxs = np.linspace(0, T - 1, 4, dtype=int)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax_i, fi in enumerate(frame_idxs):
        img = images[sample_idx, fi].cpu().numpy()
        if img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        if img.max() > 1.0:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 1.0)
        axes[ax_i].imshow(img)
        axes[ax_i].set_title(f"t={fi}", fontsize=10)
        axes[ax_i].axis("off")
    act = actions[sample_idx].cpu().numpy()
    heading = 0.0
    xy = [np.array([0.0, 0.0])]
    for t in range(act.shape[0]):
        dx_local, dy_local = act[t, 0], act[t, 1]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx_global = cos_h * dx_local - sin_h * dy_local
        dy_global = sin_h * dx_local + cos_h * dy_local
        xy.append(xy[-1] + np.array([dx_global, dy_global]))
        heading += np.arctan2(act[t, 2], act[t, 3])
    xy = np.array(xy)
    ax_traj = axes[4]
    ax_traj.plot(xy[:, 0], xy[:, 1], "b-o", markersize=4, linewidth=1.5)
    ax_traj.plot(0, 0, "k*", markersize=10, label="start")
    ax_traj.plot(xy[-1, 0], xy[-1, 1], "r*", markersize=10, label="end")
    ax_traj.set_xlabel("x"); ax_traj.set_ylabel("y")
    ax_traj.set_title("Action trajectory (xy)")
    ax_traj.legend(fontsize=8); ax_traj.set_aspect("equal", "datalim")
    ax_traj.grid(True, alpha=0.4)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    from PIL import Image as PILImage
    pil_img = PILImage.open(buf).copy()
    plt.close(fig); buf.close()
    return wandb.Image(pil_img, caption=f"sample {sample_idx}")


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path, world_model, optimizer, lr_scheduler, device,
                    skip_optimizer_state=False):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = world_model.module.load_state_dict(
        checkpoint["model_state_dict"], strict=False,
    )
    if missing:
        logging.info(f"  Missing keys: {len(missing)}")
    if unexpected:
        logging.info(f"  Unexpected keys: {len(unexpected)}")

    if skip_optimizer_state:
        logging.info("  ⚠ Skipping optimizer/scheduler state")
    else:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info("  ✓ Loaded optimizer state")
            except Exception as e:
                logging.warning(f"  Could not load optimizer state: {e}")
        if "lr_scheduler_state_dict" in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                logging.info("  ✓ Loaded lr_scheduler state")
            except Exception:
                pass
    resume_step = checkpoint.get("step", 0)
    logging.info(f"  ✓ Resuming from step {resume_step}")
    return resume_step



# ═══════════════════════════════════════════════════════════════════════════════
# Dataset (GNM image portion of CAST, RLDS format)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cast_datasets(cfg):
    cast_transform = ModuleSpec.create(
        gnm_action_angle_dataset_transform,
        action_horizon=cfg.action_horizon,
    )

    dataset_kwargs_list = []
    all_sample_weights = []

    for name, weight in zip(cfg.dataset_names, cfg.sample_weights):
        subdir = "atomic_datasets" if name.startswith("atomic_") else name
        dataset_kwargs_list.append({
            "name": name,
            "data_dir": f"{cfg.data_dir}/{subdir}",
            "image_obs_keys": {"primary": "image"},
            "proprio_obs_key": "state",
            "language_key": "language_instruction",
            "action_proprio_normalization_type": NormalizationType.BOUNDS,
            "standardize_fn": cast_transform,
            "force_recompute_dataset_statistics": cfg.force_recompute_dataset_statistics,
            "skip_norm": False,
            "action_normalization_mask": [True, True, False, False],
        })
        all_sample_weights.append(weight)


    shared_kwargs = dict(
        sample_weights=all_sample_weights,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        frame_transform_kwargs={
            "resize_size": {"primary": (cfg.img_size, cfg.img_size)},
            "image_augment_kwargs": {"primary": cfg.img_augment_kwargs},
        },
        traj_transform_kwargs={
            "window_size": cfg.window_size,
            "action_horizon": cfg.action_horizon,
            "goal_relabeling_strategy": None,
            "task_augment_strategy": None,
            "subsample_length": cfg.subsample_length,
        },
        batch_size=None,
        balance_weights=False,
        traj_transform_threads=cfg.traj_transform_threads,
        traj_read_threads=cfg.traj_read_threads,
    )
    train_ds = make_interleaved_dataset(dataset_kwargs_list, train=True, **shared_kwargs)
    val_ds = make_interleaved_dataset(dataset_kwargs_list, train=False, **shared_kwargs)
    stats = train_ds.dataset_statistics
    weights = train_ds.sample_weights
    train_ds = train_ds.filter(lambda x: tf.reduce_all(x["observation"]["timestep_pad_mask"]))
    val_ds = val_ds.filter(lambda x: tf.reduce_all(x["observation"]["timestep_pad_mask"]))
    train_ds = train_ds.filter(lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255))
    val_ds = val_ds.filter(lambda x: tf.reduce_any(x["observation"]["image_primary"] != 255))
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    for ds in (train_ds, val_ds):
        ds.dataset_statistics = stats
        ds.sample_weights = weights
    return train_ds, val_ds, stats


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def main(_):
    # ── Load config ───────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location("cfg_module", FLAGS.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=FLAGS.encoder)

    # ── Overrides ─────────────────────────────────────────────────────────
    if FLAGS.save_dir is not None: cfg.save_dir = FLAGS.save_dir
    if FLAGS.grad_clip is not None: cfg.grad_clip = FLAGS.grad_clip
    if FLAGS.rollout_steps is not None: cfg.rollout_steps = FLAGS.rollout_steps
    if FLAGS.batch_size is not None: cfg.batch_size = FLAGS.batch_size
    if FLAGS.num_steps is not None: cfg.num_steps = FLAGS.num_steps
    if FLAGS.lr is not None: cfg.lr = FLAGS.lr
    if FLAGS.data_dir is not None: cfg.data_dir = FLAGS.data_dir

    if FLAGS.zero_actions and FLAGS.shuffle_actions:
        raise ValueError("Use only one of --zero_actions or --shuffle_actions")

    if FLAGS.debug:
        cfg.num_steps = 500
        cfg.log_interval = 10
        cfg.eval_interval = 50
        cfg.batch_size = 64
        cfg.contrast_warmup_steps = 50
        cfg.wandb_project = "debug-monotone-latent-wm"
        logging.set_verbosity(logging.DEBUG)

    # ── Distributed setup ─────────────────────────────────────────────────
    pstate = _setup_distributed()
    device_id = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # TF32 is off by default in PyTorch for reproducibility but gives ~10%
    # free speedup on Ampere/Hopper with no meaningful accuracy loss.
    # bf16 autocast is still used for the heavy matmuls; TF32 catches any
    # remaining fp32 ops (e.g. layer-norm, loss reduction).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    is_main = pstate.is_main_process
    if is_main:
        wandb.init(
            project=cfg.wandb_project,
            name=f"{FLAGS.encoder}_{FLAGS.name}",
            config=cfg.to_dict(),
        )

    logging.info("=" * 80)
    logging.info(
        f"TRAINING LATENT WORLD MODEL  (encoder={cfg.encoder_type}, predictor={cfg.pred_type})"
    )
    logging.info("=" * 80)

    # ── Dataset ───────────────────────────────────────────────────────────
    logging.info("Loading dataset ...")
    train_tf_ds, val_tf_ds, _ = build_cast_datasets(cfg)
    dataset_statistics = _np2pt(train_tf_ds.dataset_statistics)
    train_pt = TorchRLDSDataset(train_tf_ds, None, train=True)
    val_pt = TorchRLDSDataset(val_tf_ds, None, train=False)

    def _make_loader(dataset, batch_size):
        # num_workers=0 is required: RLDS/TF dataset objects wrap C++ TF runtime
        # state that cannot be pickled for multiprocessing worker processes.
        # Parallelism is provided by TF's own prefetch/interleave pipeline instead.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    val_loader = _make_loader(val_pt, cfg.batch_size)
    logging.info("Dataset ready.")

    # ── Build model ───────────────────────────────────────────────────────
    logging.info("Building encoder + predictor ...")
    world_model = build_world_model(cfg).to(device)

    # Freeze the encoder explicitly.
    for p in world_model.encoder.parameters():
        p.requires_grad = False

    # Trainable: predictor only.
    trainable_list = [p for p in world_model.predictor.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_list)
    logging.info(f"Trainable params (predictor only): {n_trainable/1e6:.2f}M")

    world_model = DDP(world_model, device_ids=[device_id], find_unused_parameters=False)

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(trainable_list, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    total_steps = cfg.num_steps

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
    use_amp = cfg.mixed_precision in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

    # ── Resume ────────────────────────────────────────────────────────────
    resume_step = 0
    if FLAGS.resume:
        resume_step = load_checkpoint(
            FLAGS.resume, world_model, optimizer, lr_scheduler, device,
            skip_optimizer_state=FLAGS.skip_optimizer_state,
        )

    # ── Rollout schedule helper ───────────────────────────────────────────
    use_rollout_schedule = hasattr(cfg, "rollout_schedule") and cfg.rollout_schedule

    def get_rollout_settings(step, schedule):
        current = schedule[0]
        for s in schedule:
            if step >= s[0]:
                current = s
        return current[1], current[2], current[3]

    save_dir = None
    if is_main and cfg.save_dir:
        resume_dir = Path(FLAGS.resume).parent if FLAGS.resume else None
        same_config = (
            resume_dir is not None
            and (resume_dir / "config.py").exists()
            and (resume_dir / "config.py").read_bytes() == Path(FLAGS.config).read_bytes()
        )
        if same_config:
            # Plain continuation of an interrupted run: keep writing into its dir.
            save_dir = resume_dir
            logging.info(f"Resuming into existing checkpoint dir: {save_dir}")
        else:
            # New run (or a new stage such as MCR started from a Base checkpoint):
            # use a fresh dir so the source run's checkpoints are never overwritten.
            run_id = f"{FLAGS.encoder}_{FLAGS.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_dir = Path(cfg.save_dir) / run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            # The config is stored next to the checkpoints; eval and deploy read it from there.
            shutil.copy(FLAGS.config, save_dir / "config.py")
        logging.info(f"Checkpoints → {save_dir}")
    best_val_loss = float("inf")

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    if use_rollout_schedule:
        current_rollout, current_bs, current_lr = get_rollout_settings(resume_step, cfg.rollout_schedule)
    else:
        current_rollout = getattr(cfg, "rollout_steps", 1)
        current_bs = cfg.batch_size
        current_lr = cfg.lr

    if use_rollout_schedule:
        # Ensure the scheduled LR is applied to the optimizer and scheduler base.
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        lr_scheduler.base_lrs = [current_lr for _ in optimizer.param_groups]

    train_loader = _make_loader(train_pt, current_bs)
    train_iter = PrefetchLoader(infinite_loader(train_loader))
    logging.info(f"  Initial phase: rollout={current_rollout}, bs={current_bs}, lr={current_lr}")

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_validation(step):
        world_model.eval()
        vlosses = []
        n_batches = 10 if FLAGS.debug else 50
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            batch = _to_device(batch, device)
            images = batch["observation"]["image_primary"]
            actions = batch["action"]
            if actions.dim() == 4:
                actions = actions[:, :, 0, :]
            if FLAGS.zero_actions:
                actions = torch.zeros_like(actions)
            elif FLAGS.shuffle_actions:
                # Shuffle actions across batch to break action-image alignment.
                actions = actions[torch.randperm(actions.size(0), device=actions.device)]
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                gt_feats = world_model.module.encode_obs(images)
                pred = world_model.module.forward_pred(gt_feats, actions)
                loss = world_model.module.compute_loss(pred, gt_feats, shift=1)["loss"]
            vlosses.append(loss.item())

        world_model.train()
        val_loss = float(np.mean(vlosses)) if vlosses else float("inf")
        if is_main:
            wandb.log({"val/loss": val_loss, "val/step": step}, step=step)
            logging.info(f"[step {step}] val/loss = {val_loss:.6f}")
        return val_loss

    # ═════════════════════════════════════════════════════════════════════
    # Main training loop
    # ═════════════════════════════════════════════════════════════════════
    world_model.train()
    for global_step in tqdm.tqdm(
        range(resume_step, total_steps),
        initial=resume_step, total=total_steps,
        dynamic_ncols=True, desc="Training",
        disable=not is_main,
    ):
        # ── Phase transition ──────────────────────────────────────────
        if use_rollout_schedule:
            new_r, new_b, new_lr = get_rollout_settings(global_step, cfg.rollout_schedule)
            if new_r != current_rollout or new_b != current_bs or new_lr != current_lr:
                logging.info(f"\n{'='*60}")
                logging.info(f"Phase transition at step {global_step}:")
                logging.info(f"  rollout: {current_rollout} → {new_r}")
                logging.info(f"  batch:   {current_bs} → {new_b}")
                logging.info(f"  lr:      {current_lr} → {new_lr}")
                logging.info(f"{'='*60}\n")
                need_new_loader = (new_b != current_bs)
                current_rollout, current_bs, current_lr = new_r, new_b, new_lr
                if need_new_loader:
                    train_loader = _make_loader(train_pt, new_b)
                    train_iter = PrefetchLoader(infinite_loader(train_loader))
                # Apply the scheduled LR immediately and update scheduler base.
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                lr_scheduler.base_lrs = [new_lr for _ in optimizer.param_groups]
                if is_main:
                    wandb.log({
                        "schedule/rollout_steps": new_r,
                        "schedule/batch_size": new_b,
                        "schedule/lr": new_lr,
                    }, step=global_step)

        batch = next(train_iter)
        batch = _to_device(batch, device)

        images = batch["observation"]["image_primary"]            # (B, T, C, H, W) uint8
        actions = batch["action"]
        if actions.dim() == 4:
            actions = actions[:, :, 0, :]
        if FLAGS.zero_actions:
            actions = torch.zeros_like(actions)
        elif FLAGS.shuffle_actions:
            # Shuffle actions across batch to break action-image alignment.
            actions = actions[torch.randperm(actions.size(0), device=actions.device)]

        record_z = is_main and (global_step + 1) % cfg.log_interval == 0
        world_model.module.predictor.record_z = record_z

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            gt_feats = world_model.module.encode_obs(images)

            rollout_steps_used = current_rollout if use_rollout_schedule else getattr(cfg, "rollout_steps", 1)
            if rollout_steps_used > 1:
                loss_dict = world_model.module.rollout_loss(
                    gt_feats, actions,
                    rollout_steps=rollout_steps_used,
                    ctxt_window=cfg.ctxt_window,
                    tf_loss_weight=getattr(cfg, "tf_loss_weight", 0.0),
                    seed_frames=getattr(cfg, "rollout_seed_frames", 1),
                )
            else:
                # Safety fallback — new counter-curriculum schedule never hits rollout==1
                pred_feats_1step = world_model.module.forward_pred(gt_feats, actions)
                loss_dict = world_model.module.compute_loss(pred_feats_1step, gt_feats, shift=1)

            loss = loss_dict["loss"]

            # ── Contrastive action loss ───────────────────────────────
            contrast_weight = getattr(cfg, "contrast_loss_weight", 0.0)
            contrast_warmup = getattr(cfg, "contrast_warmup_steps", 0)
            if contrast_weight > 0.0 and global_step >= contrast_warmup:
                _csched    = getattr(cfg, "contrast_rollout_schedule", [(0, 2)])
                contrast_k = _csched[0][1]
                for _entry in _csched:
                    if global_step >= _entry[0]:
                        contrast_k = _entry[1]

                contrast_kwargs = dict(
                    rollout_steps=contrast_k,
                    ctxt_window=getattr(cfg, "contrast_ctxt_window", 4),
                    seed_frames=getattr(cfg, "contrast_seed_frames", 2),
                    temperature=getattr(cfg, "infonce_temperature", 0.1),
                )
                if getattr(cfg, "contrast_negatives", "cyclic") == "permutation":
                    ct_loss, sim_diag_val, sim_off_val = world_model.module.contrastive_action_loss_perm(
                        gt_feats, actions,
                        n_perm_negs=getattr(cfg, "contrast_n_perm_negs", 3),
                        **contrast_kwargs,
                    )
                else:
                    ct_loss, sim_diag_val, sim_off_val = world_model.module.contrastive_action_loss(
                        gt_feats, actions, **contrast_kwargs,
                    )
                loss = loss + contrast_weight * ct_loss
            else:
                ct_loss, sim_diag_val, sim_off_val = None, None, None
                contrast_k = 0

            # ── Monotone Cost Ranking (MCR) loss ──────────────────────────
            mcr_weight = getattr(cfg, "mcr_weight", 0.0)
            mcr_warmup = getattr(cfg, "mcr_warmup_steps", 0)
            if mcr_weight > 0.0 and global_step >= mcr_warmup:
                sf_loss, sf_viol, sf_gt_cost = world_model.module.mcr_loss(
                    gt_feats, actions,
                    rollout_steps=getattr(cfg, "mcr_rollout_steps", 6),
                    ctxt_window=getattr(cfg, "mcr_ctxt_window", cfg.ctxt_window),
                    seed_frames=getattr(cfg, "mcr_seed_frames", cfg.rollout_seed_frames),
                    n_perturb=getattr(cfg, "mcr_n_perturb", 8),
                    perturb_stds=tuple(getattr(cfg, "mcr_perturb_stds", (0.05, 0.15, 0.35))),
                    temporal_corr=getattr(cfg, "mcr_temporal_corr", 0.7),
                    margin_scale=getattr(cfg, "mcr_margin_scale", 0.5),
                    margin_min=getattr(cfg, "mcr_margin_min", 0.02),
                    rollout_loss_weight=getattr(cfg, "mcr_anchor_weight", 1.0),
                )
                loss = loss + mcr_weight * sf_loss
            else:
                sf_loss, sf_viol, sf_gt_cost = None, None, None

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"Step {global_step}: NaN/Inf loss — skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_list, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        if is_main and (global_step + 1) % cfg.log_interval == 0:
            log_dict = {
                "train/loss": loss.item(),
                "train/l2_loss": loss_dict["l2_loss"].item(),
                "train/cos_loss": loss_dict["cos_loss"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/step": global_step + 1,
                "train/rollout_steps": rollout_steps_used,
            }
            if ct_loss is not None:
                log_dict["train/contrast_loss"]    = ct_loss.item()
                log_dict["contrast/sim_diag_mean"] = sim_diag_val
                log_dict["contrast/sim_off_mean"]  = sim_off_val
                log_dict["contrast/sim_gap"]       = sim_diag_val - sim_off_val
                log_dict["contrast/rollout_k"]     = contrast_k
            if sf_loss is not None:
                log_dict["mcr/loss"]      = sf_loss.item()
                log_dict["mcr/mean_viol"] = sf_viol
                log_dict["mcr/gt_cost"]   = sf_gt_cost
            if record_z:
                z = world_model.module.predictor._last_z
                if z is not None and z.grad is not None:
                    log_dict["train/adaln_z_grad_norm"] = z.grad.detach().float().norm().item()
            with torch.no_grad():
                predictor = world_model.module.predictor
                stats = {}
                stats.update(_tensor_stats("action/raw", actions))

                if predictor.action_encoder_inpred:
                    act_enc = predictor.action_encoder(actions)
                else:
                    act_enc = actions
                stats.update(_tensor_stats("action/enc", act_enc))
                if getattr(predictor, "action_norm", False):
                    act_enc = predictor.action_norm_layer(act_enc)
                    stats.update(_tensor_stats("action/norm", act_enc))

                # Mirror the z-building logic from VisionTransformerAdaLN.forward
                ac_mode = getattr(predictor, "action_conditioning", "adaln")
                if ac_mode in ("adaln", "both"):
                    fused = act_enc
                else:  # "token": action is a sequence token, not in AdaLN z
                    fused = torch.zeros_like(act_enc)

                stats.update(_tensor_stats("adaln/z", fused))
                if getattr(predictor, "action_encoder_inpred", False):
                    enc = predictor.action_encoder
                    if hasattr(enc, "weight"):
                        grad = enc.weight.grad
                        if grad is not None:
                            log_dict["train/action_encoder_grad_norm"] = grad.detach().float().norm().item()
                    else:
                        grads = [p.grad.detach().float() for p in enc.parameters() if p.grad is not None]
                        if grads:
                            log_dict["train/action_encoder_grad_norm"] = torch.cat([g.flatten() for g in grads]).norm().item()
                if hasattr(predictor, "predictor_blocks"):
                    w_grads, b_grads = [], []
                    for i, blk in enumerate(predictor.predictor_blocks):
                        linear = blk.adaLN_modulation[1]
                        w_grad = linear.weight.grad
                        if w_grad is not None:
                            w_norm = w_grad.detach().float().norm().item()
                            log_dict[f"adaln/grad_norm/block{i}"] = w_norm
                            w_grads.append(w_norm)
                        b_grad = linear.bias.grad
                        if b_grad is not None:
                            b_norm = b_grad.detach().float().norm().item()
                            log_dict[f"adaln/bias_grad_norm/block{i}"] = b_norm
                            b_grads.append(b_norm)
                    if w_grads:
                        w_tensor = torch.tensor(w_grads)
                        log_dict["adaln/grad_norm_min"] = w_tensor.min().item()
                        log_dict["adaln/grad_norm_median"] = w_tensor.median().item()
                        log_dict["adaln/grad_norm_max"] = w_tensor.max().item()
                    if b_grads:
                        b_tensor = torch.tensor(b_grads)
                        log_dict["adaln/bias_grad_norm_min"] = b_tensor.min().item()
                        log_dict["adaln/bias_grad_norm_median"] = b_tensor.median().item()
                        log_dict["adaln/bias_grad_norm_max"] = b_tensor.max().item()
                    # Weight norms: moving away from ~0 confirms escape from identity trap
                    for i, blk in enumerate(predictor.predictor_blocks):
                        w = blk.adaLN_modulation[1].weight
                        log_dict[f"adaln/weight_norm/block{i}"] = w.detach().float().norm().item()
                log_dict.update(stats)
            if record_z and rollout_steps_used == 1:
                z = world_model.module.predictor._last_z
                if z is not None:
                    cf_input = gt_feats
                    with torch.no_grad(), torch.amp.autocast(
                        "cuda", dtype=amp_dtype, enabled=use_amp
                    ):
                        zero_z = torch.zeros_like(z)
                        cf_pred = world_model.module.forward_pred(
                            cf_input, actions, override_z=zero_z
                        )
                        cf_loss = world_model.module.compute_loss(
                            cf_pred, gt_feats, shift=1
                        )["loss"]
                    log_dict["train/cf_zero_z_loss"] = cf_loss.item()
                    log_dict["train/cf_zero_z_delta"] = cf_loss.item() - loss.item()
            # Per-step rollout diagnostics: for each rollout step we log
            # train/rollout/step{i}/l2_loss and cos_loss (what the model
            # achieves autoregressively) plus tf_l2_loss (teacher-forced
            # anchor, for comparison). Gap between tf and rollout at step i
            # is a direct measure of compounding error.
            for k, v in loss_dict.items():
                if isinstance(k, str) and k.startswith("step"):
                    log_dict[f"train/rollout/{k}"] = v
            wandb.log(log_dict, step=global_step)

        if is_main and cfg.viz and (global_step + 1) % cfg.viz_interval == 0:
            viz_images = []
            for s_idx in range(min(4, images.shape[0])):
                viz_images.append(_make_sample_viz(
                    images, actions, s_idx
                ))
            wandb.log({"train/samples": viz_images}, step=global_step)

        if (global_step + 1) % cfg.eval_interval == 0:
            val_loss = run_validation(global_step + 1)
            if is_main and val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    torch.save({
                        "step": global_step + 1,
                        "model_state_dict": world_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "dataset_statistics": dataset_statistics,
                        "config": cfg.to_dict(),
                    }, save_dir / "best_model.pt")
                    logging.info(f"  Saved best model (val_loss={val_loss:.6f})")

        if is_main and save_dir and global_step + 1 == 5000:
            torch.save({
                "step": global_step + 1,
                "model_state_dict": world_model.module.state_dict(),
                "config": cfg.to_dict(),
            }, save_dir / "checkpoint_step5000_early.pt")
            logging.info("  Saved early diagnostic checkpoint at step 5000")

        if is_main and save_dir and (global_step + 1) % cfg.save_interval == 0:
            torch.save({
                "step": global_step + 1,
                "model_state_dict": world_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "dataset_statistics": dataset_statistics,
                "config": cfg.to_dict(),
            }, save_dir / f"checkpoint_step{global_step + 1}.pt")

    # ── Final ────────────────────────────────────────────────────────────
    logging.info("\nFinal validation ...")
    run_validation(total_steps)
    if is_main and save_dir:
        torch.save({
            "step": total_steps,
            "model_state_dict": world_model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "dataset_statistics": dataset_statistics,
            "config": cfg.to_dict(),
        }, save_dir / "final_model.pt")
        logging.info(f"Saved final checkpoint to {save_dir / 'final_model.pt'}")

    logging.info("Training complete.")
    if is_main:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
