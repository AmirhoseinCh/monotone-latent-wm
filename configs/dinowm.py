"""
DINO-WM baseline: ViT predictor with per-patch action concatenation.

This is the `pred_type="vit"` variant: it uses DINO-WM's original ViT backbone
(vit.py) wrapped by ViTPredictorWM, which encodes the action with an MLP and
concatenates it onto every patch token — the actual DINO-WM action mechanism.

Pipeline:
    Encoder    : frozen DINOv2-ViT-S/14            -> (B, T, 1, H, W, 384)
    Predictor  : ViTPredictorWM (vit.py backbone)  -> next-frame patch features
                 transformer dim = embed_dim + action_emb_dim = 384 + 10 = 394
                 learned positional embeddings + block-causal mask
    Loss       : pure L2 latent-consistency (DINO-WM recipe)
    Training   : single-step teacher forcing, no rollout curriculum

Faithfulness notes vs. the DINO-WM paper:
  * Backbone, learned pos-embeddings, block-causal mask, per-patch action
    concat — all match. No RoPE deviation (vit.py uses learned pos-embeddings).
  * Resolution 224 -> 16x16 grid (paper used 196 -> 14x14). Harmless.
  * Single global LR (paper used predictor 5e-5 / action-encoder 5e-4).
  * No frameskip knob — set it in the Cast transform if available.

Predictor-construction reminders (see build_predictor / ViTPredictorWM):
  * num_frames is built from cfg.window_size (NOT ctxt_window): the
    teacher-forcing fallback in rollout_loss can feed the full window.
  * The training loop's `record_z` must be False on this path — the
    counterfactual-z probe is AdaLN-only and the wrapper will raise.
  * `adaln/*` W&B metrics are meaningless on this path; ignore them.

Action dim = 4 (local_x, local_y, sin_yaw, cos_yaw). Dataset: Cast.
"""

from ml_collections import ConfigDict


def get_config(encoder_type: str = "dino"):
    """
    Args:
        encoder_type: "dino" (DINOv2), "dinov3", or "vjepa".
    Returns:
        ConfigDict with all hyperparameters.
    """
    cfg = ConfigDict()

    # ── Encoder ──────────────────────────────────────────────────────────
    cfg.encoder_type = encoder_type

    _EMBED_DIMS = {
        "vits16": 384,
        "vitb16": 768,
        "vitl16": 1024,
        "vitg16": 1536,
    }

    if encoder_type == "dino":
        # DINO-WM encoder: frozen DINOv2-ViT-S/14.
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

    # ── Predictor (DINO-WM vit.py backbone via ViTPredictorWM) ───────────
    cfg.pred_type = "vit"          # selects ViTPredictorWM in build_predictor
    cfg.pred_depth = 6             # DINO-WM: 6 layers
    cfg.pred_num_heads = 16        # DINO-WM: 16 heads (heads/dim_head decoupled)
    cfg.pred_dim_head = 64         # inner attn dim = pred_num_heads * pred_dim_head
    cfg.pred_mlp_dim = 2048        # DINO-WM predictor MLP hidden dim
    # NOTE: the transformer model dim is embed_dim + action_emb_dim (=394),
    # set inside ViTPredictorWM — there is no separate predictor embed dim
    # on this path.

    # AdaLN-only fields — IGNORED when pred_type == "vit". Kept so the AdaLN
    # branch of build_predictor and any cfg lookups still resolve.
    cfg.pred_embed_dim = 384
    cfg.use_rope = True
    cfg.is_causal = True
    cfg.use_silu = False
    cfg.init_scale_factor_adaln = 1
    cfg.action_conditioning = "concat"   # informational; vit path concats per-patch
    cfg.action_norm = False

    # ── Action ──────────────────────────────────────────────────────────
    cfg.action_dim = 4
    cfg.action_horizon = 1
    cfg.action_encoder_inpred = True     # action encoded inside the predictor
    cfg.action_emb_dim = 10              # DINO-WM Table 12: "Action emb dim 10"

    # ── Proprio (disabled — ViTPredictorWM has no proprio path) ──────────
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
    # window_size also sets the predictor's num_frames (pos-embedding +
    # causal-mask size). ctxt_window = 4 mirrors DINO-WM's H=4.
    cfg.window_size = 8
    cfg.subsample_length = 50
    cfg.force_recompute_dataset_statistics = False  # stats are computed once and cached next to the dataset

    # CAST datasets
    cfg.data_dir = "./data"    # contains cast_filtered_dataset/ (RLDS)
    cfg.dataset_names = [
        "cast_filtered_dataset",
    ]
    cfg.sample_weights = [1]


    cfg.batch_size = 32            # DINO-WM batch size
    cfg.shuffle_buffer_size = 5000
    cfg.traj_transform_threads = 16
    cfg.traj_read_threads = 16

    cfg.img_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        augment_order=["random_brightness", "random_contrast"],
    )

    # ── Loss ─────────────────────────────────────────────────────────────
    # DINO-WM: pure L2 latent-consistency loss. Cosine OFF.
    cfg.loss = ConfigDict()
    cfg.loss.l2_loss_weight = 1.0
    cfg.loss.l1_loss_weight = 0.0
    cfg.loss.cos_loss_weight = 0.0
    cfg.loss.smooth_l1_loss_weight = 0.0
    cfg.loss.proprio_loss = False

    # ── Rollout schedule (single-step teacher forcing, no curriculum) ────
    cfg.rollout_schedule = [
        (7000, 1, 32, 5e-5),
    ]
    cfg.ctxt_window = 4            # DINO-WM history length H
    cfg.rollout_seed_frames = 4    # seed buffer with full H frames of GT context
    cfg.tf_loss_weight = 0.0
    cfg.residual_prediction = False     # DINO-WM predicts z_{t+1} directly;
                                        # vit.py is residual internally — do
                                        # NOT also add the skip in forward_pred

    # Contrastive loss OFF — not part of DINO-WM.
    cfg.contrast_loss_weight = 0.0
    cfg.contrast_rollout_schedule = [
        (7000, 1),
    ]
    cfg.contrast_seed_frames = 4
    cfg.contrast_ctxt_window = 4
    cfg.contrast_warmup_steps = 100
    cfg.infonce_temperature = 0.1

    # ── Optimisation ─────────────────────────────────────────────────────
    # lr = 5e-5 matches DINO-WM's predictor lr. DINO-WM trained ~100 epochs;
    # scale num_steps to roughly 100 * (dataset_size / batch_size).
    cfg.num_steps = 7000
    cfg.lr = 5e-5
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
    cfg = get_config("dino")
    print(cfg)
