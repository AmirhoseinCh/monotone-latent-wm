"""
ViTPredictorWM — DINO-WM-faithful predictor wrapper.

The original DINO-WM `vit.py` (ViTPredictor) is only the *backbone*: it adds a
learned positional embedding, runs a block-causal transformer, and returns a
tensor. It has no action input. DINO-WM's actual predictor encodes the action
with an MLP and concatenates it onto every patch token *before* the
transformer — that fusion logic is not in vit.py.

This wrapper adds exactly that, and exposes the same forward signature and
return contract as VisionTransformerAdaLN, so CastWorldModelLang.forward_pred
needs no changes:

    forward(x, actions, proprio=None, language=None, override_z=None)
        x:        (B, T, V, H, W, embed_dim)      V == 1
        actions:  (B, T, action_dim)
        returns:  (pred, None, None)  with pred (B, T, V*H*W, embed_dim)

Faithful to DINO-WM:
  * action encoded by an MLP, concatenated per-patch
    -> transformer model dim = embed_dim + action_emb_dim;
  * learned positional embeddings + block-causal mask from vit.py;
  * predicted next-frame features = the obs-portion of the transformer output
    (the action channels of the output are discarded).

Not supported (DINO-WM does not need them; the wrapper raises if asked):
  * proprio, language, override_z.

Note on vit.py globals: ViTPredictor sets module-level NUM_FRAMES / NUM_PATCHES
at construction. Construct exactly ONE predictor per process (this wrapper does).
"""

import torch
import torch.nn as nn

from lmwm.models.vit import ViTPredictor


class ViTPredictorWM(nn.Module):
    def __init__(
        self,
        *,
        num_patches,            # patches per frame, e.g. cfg.grid_size ** 2
        num_frames,             # MAX frames per window — pass cfg.window_size,
                                # not ctxt_window (the teacher-forcing fallback
                                # in rollout_loss feeds the full window)
        embed_dim,              # frozen-encoder feature dim (DINOv2-S/14 = 384)
        depth,
        heads,
        mlp_dim=2048,           # DINO-WM predictor MLP hidden dim
        dim_head=64,            # decoupled from heads; inner = heads * dim_head
        action_dim=4,
        action_emb_dim=10,      # DINO-WM Table 12: "Action emb dim 10"
        action_hidden_dim=128,
        dropout=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_emb_dim = action_emb_dim
        self.num_patches = num_patches

        # Transformer operates on obs features + per-patch action features.
        model_dim = embed_dim + action_emb_dim

        # DINO-WM-style MLP action encoder: action_dim -> action_emb_dim.
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, action_emb_dim),
        )

        # Original DINO-WM backbone. `dim` need not equal heads*dim_head —
        # vit.Attention decouples them (to_qkv: dim -> heads*dim_head*3).
        self.vit = ViTPredictor(
            num_patches=num_patches,
            num_frames=num_frames,
            dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            pool="mean",          # unused by ViTPredictor.forward; value irrelevant
            use_sdpa=use_sdpa,
        )

        # ── Attributes the training script's W&B logging block reads ──────
        # The logging code is written against the AdaLN predictor. These
        # attributes keep it from crashing. NOTE: any `adaln/*` metrics it
        # emits are meaningless for this predictor — ignore them. The
        # AdaLN block-grad logging is hasattr-guarded on `predictor_blocks`,
        # which this wrapper does not have, so that part is simply skipped.
        self.action_encoder_inpred = True   # `action_encoder(actions)` is valid
        self.action_norm = False            # no encoded-action normalization
        self.use_language = False           # skips the language logging branch
        self.action_conditioning = "concat" # informational only
        self.record_z = False               # counterfactual-z probe is AdaLN-only
        self._last_z = None

    def forward(self, x, actions, proprio=None, language=None, override_z=None):
        if proprio is not None:
            raise NotImplementedError(
                "ViTPredictorWM has no proprio path; set cfg.use_proprio=False."
            )
        if language is not None:
            raise NotImplementedError(
                "ViTPredictorWM has no language path; set cfg.use_language=False, "
                "or use the AdaLN predictor for language conditioning."
            )
        if override_z is not None:
            raise NotImplementedError(
                "override_z is AdaLN-only. Ensure the training loop's `record_z` "
                "is False when cfg.pred_type == 'vit'."
            )

        B, T, V, H, W, D = x.shape
        assert V == 1, f"expected V==1, got V={V}"
        assert D == self.embed_dim, f"expected embed_dim={self.embed_dim}, got {D}"
        N = V * H * W
        assert N == self.num_patches, (
            f"num_patches mismatch: predictor built for {self.num_patches}, got "
            f"{N} (H={H}, W={W}). Rebuild with num_patches=cfg.grid_size**2."
        )

        # Patch tokens: (B, T, N, embed_dim)
        obs = x.reshape(B, T, N, D)

        # Encode action, broadcast to every patch: (B, T, N, action_emb_dim)
        act = self.action_encoder(actions)                      # (B, T, action_emb_dim)
        act = act.unsqueeze(2).expand(B, T, N, self.action_emb_dim)

        # Per-patch concat -> (B, T, N, embed_dim + action_emb_dim)
        tokens = torch.cat([obs, act], dim=-1)

        # Flatten time+patches, FRAME-MAJOR (row-major reshape gives
        # [frame0 patches, frame1 patches, ...]), which is exactly the
        # layout vit.py's block-causal mask assumes.
        tokens = tokens.reshape(B, T * N, D + self.action_emb_dim)

        out = self.vit(tokens)                                  # (B, T*N, D+Ae)
        out = out.reshape(B, T, N, D + self.action_emb_dim)

        # Predicted next-frame features = obs-portion of the output. The
        # transformer is residual, so this is input_obs + learned delta;
        # the action channels are conditioning only and are discarded.
        pred = out[..., :D]                                     # (B, T, N, embed_dim)
        return pred, None, None
