"""
MCR: Monotone Cost Ranking fine-tuning stage (Eqs. 3-4).

Resumes from a converged Base (or TF / ACT) checkpoint and fine-tunes with a
pairwise margin-ranking loss over graded AR(1)-correlated action perturbations,
plus a rollout anchor on the GT actions:

    L_MCR = L_rollout + L_mcr,     m_ij = max(0.02, 0.5 * |d_i - d_j|)

  - Low LR (1e-5), no warm-up; rollout fixed at K = 8.
  - 3 perturbations per sample at sigma in {0.05, 0.15, 0.35}, rho = 0.7.

NOTE: cfg.num_steps is the *absolute* final step. Training resumes at the
checkpoint's step, so set num_steps = <checkpoint step> + 1000
(e.g. 19000 for an 18k-step Base checkpoint).

Run:
    torchrun --nproc_per_node 4 train.py --config configs/mcr.py --encoder dino --name mcr \
        --resume <base_run_dir>/final_model.pt --skip_optimizer_state

Watch in W&B:
    mcr/mean_viol -> 0          (cost surface becoming monotone)
    mcr/gt_cost   stays low     (GT is still the minimum)
"""

from ml_collections import ConfigDict


def get_config(encoder_type: str = "dino"):
    cfg = ConfigDict()

    # ── Encoder (dinov3, must match checkpoint) ──────────────────────────
    cfg.encoder_type = encoder_type

    _EMBED_DIMS = {
        "vits16": 384,
        "vitb16": 768,
        "vitl16": 1024,
        "vitg16": 1536,
    }

    if encoder_type == "dinov3":
        cfg.enc_model_name = "dinov3_vits16"
        cfg.embed_dim = _EMBED_DIMS[cfg.enc_model_name.split("_")[1]]
        cfg.img_size = 256
        cfg.patch_size = 16
        cfg.grid_size = 16
        cfg.frozen_patterns = ("*dino_model*",)
    elif encoder_type == "dino":
        cfg.enc_model_name = "dinov2_vits14"
        cfg.embed_dim = 384
        cfg.img_size = 224
        cfg.patch_size = 14
        cfg.grid_size = 16
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

    # ── Predictor ────────────────────────────────────────────────────────
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
    cfg.action_conditioning = "adaln"
    cfg.action_norm = True

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
    cfg.window_size = 14
    cfg.subsample_length = 50
    cfg.force_recompute_dataset_statistics = False  # already computed

    cfg.data_dir = "./data"    # contains cast_filtered_dataset/ (RLDS)
    cfg.dataset_names = ["cast_filtered_dataset"]
    cfg.sample_weights = [1]

    cfg.batch_size = 8
    cfg.shuffle_buffer_size = 5000
    cfg.traj_transform_threads = 16
    cfg.traj_read_threads = 16

    cfg.img_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=["random_brightness", "random_contrast"],
    )

    # ── Loss weights ─────────────────────────────────────────────────────
    cfg.loss = ConfigDict()
    cfg.loss.l2_loss_weight = 1.0
    cfg.loss.l1_loss_weight = 0.0
    cfg.loss.cos_loss_weight = 0.5
    cfg.loss.smooth_l1_loss_weight = 0.0
    cfg.loss.proprio_loss = False

    # ── Rollout schedule — frozen at 8-step (past all thresholds) ────────
    # Single entry at step 0 so the schedule always resolves to 8 steps / 1e-5.
    cfg.rollout_schedule = [
        (0, 8, 8, 1e-5),
    ]
    cfg.ctxt_window = 4
    cfg.rollout_seed_frames = 4
    cfg.tf_loss_weight = 0.0
    cfg.residual_prediction = False

    # ── Contrastive loss (off for MCR) ─────────────────────────────────────
    cfg.contrast_loss_weight = 0
    cfg.contrast_rollout_schedule = [(0, 8)]
    cfg.contrast_seed_frames = 4
    cfg.contrast_ctxt_window = 4
    cfg.contrast_warmup_steps = 0
    cfg.infonce_temperature = 0.1
    cfg.contrast_negatives = "cyclic"

    # ── Monotone Cost Ranking (MCR) ───────────────────────────────────────────
    cfg.mcr_weight = 1.0          # ranking loss weight in total loss
    cfg.mcr_warmup_steps = 0      # active from the first step
    cfg.mcr_rollout_steps = 8     # match base training horizon
    cfg.mcr_ctxt_window = 4
    cfg.mcr_seed_frames = 4
    cfg.mcr_n_perturb = 3         # 1 per std level — (1+3)*4 = 16 forward passes
    cfg.mcr_perturb_stds = (0.05, 0.15, 0.35)   # small / mid / large noise
    cfg.mcr_temporal_corr = 0.7  # AR(1) correlation of perturbation noise
    cfg.mcr_margin_scale = 0.5   # margin grows with trajectory distance gap
    cfg.mcr_margin_min = 0.02    # floor margin for any ordered pair
    cfg.mcr_anchor_weight = 1.0  # weight of anchor (GT rollout) loss within the MCR loss
    # Raise mcr_anchor_weight if mcr/gt_cost starts rising.

    # ── Optimisation ─────────────────────────────────────────────────────
    # Absolute final step: Base checkpoint step (18k) + 1k MCR steps.
    cfg.num_steps = 19000
    cfg.lr = 1e-5
    cfg.weight_decay = 0.01
    cfg.warmup_steps = 0       # no warmup — continuing a converged model
    cfg.grad_clip = 1.0
    cfg.mixed_precision = "bf16"

    # ── Logging / checkpointing ──────────────────────────────────────────
    cfg.log_interval = 50
    cfg.eval_interval = 500
    cfg.viz = False
    cfg.viz_interval = 1000
    cfg.save_interval = 1000
    cfg.save_dir = "./checkpoints"
    cfg.wandb_project = "monotone-latent-wm"

    return cfg


if __name__ == "__main__":
    cfg = get_config("dinov3")
    print(cfg)
