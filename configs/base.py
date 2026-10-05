"""
Base: AdaLN ViT predictor trained with the autoregressive rollout loss.

Pipeline:
    Encoder   : frozen DINOv2-ViT-S/14 ("dino") or DINOv3-ViT-S/16 ("dinov3")
    Predictor : 6-layer AdaLN ViT, RoPE, causal mask; action -> AdaLN modulation
    Loss      : L2 + 0.5 * (1 - cos)   on rollout predictions (Eqs. 1-2)

Rollout counter-curriculum (cfg.rollout_schedule): K ramps 2 -> 4 -> 6 -> 8
while the per-GPU batch size and learning rate are annealed. Each entry is
(start_step, K, batch_per_gpu, lr); an entry becomes active once
step >= start_step, and the first entry also covers steps before it.

Action = (local_x, local_y, sin_yaw, cos_yaw).

Run:
    torchrun --nproc_per_node 4 train.py --config configs/base.py --encoder dino --name base
"""

from ml_collections import ConfigDict


def get_config(encoder_type: str = "dino"):
    """
    Args:
        encoder_type: "dino" (DINOv2), "dinov3", or "vjepa". Selects the frozen image encoder.
    Returns:
        ConfigDict with all hyperparameters.
    """
    cfg = ConfigDict()

    # ── Encoder ──────────────────────────────────────────────────────────
    cfg.encoder_type = encoder_type  # "dino" (DINOv2) | "dinov3" | "vjepa"

    _EMBED_DIMS = {
        "vits16": 384,
        "vitb16": 768,
        "vitl16": 1024,
        "vitg16": 1536,
    }

    if encoder_type == "dino":
        cfg.enc_model_name = "dinov2_vits14"
        cfg.embed_dim = 384
        cfg.img_size = 224
        cfg.patch_size = 14
        cfg.grid_size = 16
        cfg.frozen_patterns = ("*dino_model*",)
    elif encoder_type == "dinov3":
        cfg.enc_model_name = "dinov3_vits16"
        cfg.embed_dim = _EMBED_DIMS[cfg.enc_model_name.split("_")[1]]
        cfg.img_size = 256
        cfg.patch_size = 16
        cfg.grid_size = 16  # 256 // 16
        cfg.frozen_patterns = ("*dino_model*",)

    elif encoder_type == "vjepa":
        cfg.enc_model_name = "vjepa2_vit_large"
        cfg.embed_dim = 1024
        cfg.img_size = 256
        cfg.patch_size = 16
        cfg.grid_size = 16
        cfg.frozen_patterns = ("*vjepa2_model*",)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    # ── Predictor (AdaLN) ────────────────────────────────────────────────
    cfg.pred_type = "AdaLN"
    cfg.pred_embed_dim = 384
    cfg.pred_depth = 6
    cfg.pred_num_heads = 12
    cfg.use_rope = True
    cfg.is_causal = True
    cfg.use_silu = False
    cfg.init_scale_factor_adaln = 1

    # ── Action ──────────────────────────────────────────────────────────
    cfg.action_dim = 4
    cfg.action_horizon = 1
    cfg.action_encoder_inpred = True
    cfg.action_conditioning = "adaln"  # "adaln" | "token" | "both"
    cfg.action_norm = True               # normalize encoded actions before fusion

    # ── Proprio (disabled) ───────────────────────────────────────────────
    cfg.use_proprio = False
    cfg.proprio_dim = 0
    cfg.proprio_emb_dim = 0
    cfg.proprio_tokens = 0
    cfg.proprio_encoding = "feature"
    cfg.proprio_encoder_inpred = True

    # ── World model flags ────────────────────────────────────────────────
    cfg.batchify_video = True
    cfg.dup_image = False
    cfg.normalize_reps = True

    # ── Data ─────────────────────────────────────────────────────────────
    # 14,240 training trajectories (GNM image portion of CAST)
    cfg.window_size = 14
    cfg.subsample_length = 50  # random windows per trajectory; None = all
    cfg.force_recompute_dataset_statistics = False  # stats are computed once and cached next to the dataset

    # CAST datasets
    cfg.data_dir = "./data"    # contains cast_filtered_dataset/ (RLDS)
    cfg.dataset_names = [
        "cast_filtered_dataset",
    ]
    cfg.sample_weights = [1]

    cfg.batch_size = 64
    cfg.shuffle_buffer_size = 5000
    cfg.traj_transform_threads = 16
    cfg.traj_read_threads = 16

    cfg.img_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=["random_brightness", "random_contrast"],
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    cfg.loss = ConfigDict()
    cfg.loss.l2_loss_weight = 1.0
    cfg.loss.l1_loss_weight = 0.0
    cfg.loss.cos_loss_weight = 0.5
    cfg.loss.smooth_l1_loss_weight = 0.0
    cfg.loss.proprio_loss = False

    # ── Rollout schedule ─────────────────────────────────────────────────
    cfg.rollout_schedule = [
        (2000,  2,  128, 1e-4),   # counter-curriculum: start at 2-step, penalizes identity immediately
        (6000,  4,  64, 5e-5),
        (10000,  6,  32, 1e-5),
        (14000,  8,  32, 5e-6),
    ]
    cfg.ctxt_window = 4
    cfg.rollout_seed_frames = 4
    cfg.tf_loss_weight = 0.0
    cfg.residual_prediction = False       # predict z_{t+1}, not z_{t+1} - z_t
    cfg.contrast_loss_weight      = 0     # ACT only (see configs/act.py)
    cfg.contrast_rollout_schedule = [   # K ramps in sync with main schedule, stays <= main K
        (2000,  2),
        (6000,  4),
        (10000,  6),
        (14000,  8),
    ]
    cfg.contrast_seed_frames  = 4
    cfg.contrast_ctxt_window  = 4
    cfg.contrast_warmup_steps = 100     # gate until predictor is non-degenerate
    cfg.infonce_temperature   = 0.1

    # ── Optimisation ─────────────────────────────────────────────────────
    cfg.num_steps = 18000
    cfg.lr = 5e-4
    cfg.weight_decay = 0.01
    cfg.warmup_steps = 500
    cfg.grad_clip = 1.0
    cfg.mixed_precision = "bf16"

    # ── Logging / checkpointing ──────────────────────────────────────────
    cfg.log_interval = 50
    cfg.eval_interval = 1000
    cfg.viz = False
    cfg.viz_interval = 1000
    cfg.save_interval = 2000
    cfg.save_dir = "./checkpoints"
    cfg.wandb_project = "monotone-latent-wm"

    return cfg


if __name__ == "__main__":
    cfg = get_config("dinov3")
    print(cfg)
