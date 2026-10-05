"""
Latent world model: frozen DINO-family encoder + action-conditioned predictor.

Contains the model wrapper, all training objectives (rollout loss, MCR loss,
ACT contrastive loss) and the encoder/predictor factories. Shared by training,
evaluation, the plotting tools and robot deployment.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lmwm.encoders.dino_encoder import DinoV2EncoderPt, DinoV3EncoderPt
from lmwm.encoders.vjepa_encoder import VJEPA2EncoderPt


def _sample_correlated_noise(B, K, P, action_dim, temporal_corr, device):
    """AR(1) temporally-correlated noise. Returns (B, K, P, action_dim)."""
    noise = torch.zeros(B, K, P, action_dim, device=device)
    noise[:, 0] = torch.randn(B, P, action_dim, device=device)
    decay = (1.0 - temporal_corr ** 2) ** 0.5
    for t in range(1, K):
        noise[:, t] = (temporal_corr * noise[:, t - 1]
                       + decay * torch.randn(B, P, action_dim, device=device))
    return noise

class LatentWorldModel(nn.Module):
    """
    Latent world model:
        encoder   -> frozen DINOv2 / DINOv3 / V-JEPA2   (B, T, 1, H, W, D)
        predictor -> AdaLN ViT (ours) or ViT with concatenated actions (DINO-WM)
    """

    def __init__(self, encoder, predictor, cfg):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

        self.enc_type = cfg.encoder_type
        self.grid_size = cfg.grid_size
        self.normalize_reps = cfg.normalize_reps

        self.cfgs_loss = {
            "l2_loss_weight": cfg.loss.l2_loss_weight,
            "l1_loss_weight": cfg.loss.l1_loss_weight,
            "cos_loss_weight": cfg.loss.cos_loss_weight,
            "smooth_l1_loss_weight": cfg.loss.smooth_l1_loss_weight,
        }
        self.residual_prediction = getattr(cfg, "residual_prediction", False)

    # ── Encode observations ──────────────────────────────────────────────
    @torch.no_grad()
    def encode_obs(self, images):
        """(B, T, C, H, W) uint8 in [0, 255] -> (B, T, 1, gH, gW, D) features."""
        B, T, C, H, W = images.shape
        self.encoder.eval()
        flat = rearrange(images, "b t c h w -> (b t) c h w")
        embs = self.encoder(flat)
        embs = rearrange(embs, "(b t) d h w -> b t 1 h w d", b=B, t=T)
        if self.normalize_reps:
            embs = F.layer_norm(embs, (embs.size(-1),))
        return embs

    # ── Forward predictor ────────────────────────────────────────────────
    def forward_pred(self, video_features, actions, lang_emb=None, override_z=None):
        """(B, T, 1, gH, gW, D) + (B, T, A) -> (B, T, 1, gH, gW, D) next-step features.

        lang_emb is unused by the paper models and kept only for API compatibility.
        """
        pred_video, _, _ = self.predictor(
            video_features,
            actions,
            proprio=None,
            language=lang_emb,
            override_z=override_z,
        )
        pred_video = rearrange(
            pred_video,
            "b t (v h w) d -> b t v h w d",
            v=1, h=self.grid_size, w=self.grid_size,
        )
        # Residual skip: add input features in raw (pre-norm) space so the predictor
        # learns delta z_{t+1} - z_t. Must come before normalize_reps.
        if self.residual_prediction:
            pred_video = pred_video + video_features
        if self.normalize_reps:
            pred_video = F.layer_norm(pred_video, (pred_video.size(-1),))
        return pred_video

    def forward(self, images, actions):
        video_features = self.encode_obs(images)
        pred_video_features = self.forward_pred(video_features, actions)
        return pred_video_features, video_features

    # ── Per-step prediction loss: L2 + cosine (Eq. 1) ─────────────────────────────────────────────────
    def compute_loss(self, pred_video_features, video_features, shift=1):
        B, T, V, H, W, C = pred_video_features.shape
        pred_flat = pred_video_features.reshape(B, T, V * H * W, C)
        gt_flat = video_features.reshape(B, T, V * H * W, C)
        if shift == 0:
            pred_, gt_ = pred_flat, gt_flat
        else:
            pred_, gt_ = pred_flat[:, :-shift], gt_flat[:, shift:]

        l2_loss = F.mse_loss(pred_, gt_, reduction="none").mean(dim=-1)
        l1_loss = F.l1_loss(pred_, gt_, reduction="none").mean(dim=-1)
        smooth_l1_loss = F.smooth_l1_loss(pred_, gt_, reduction="none").mean(dim=-1)
        cos_loss = -(
            pred_ * gt_
            / (pred_.norm(dim=-1, keepdim=True) * gt_.norm(dim=-1, keepdim=True) + 1e-8)
        ).sum(dim=-1)
        loss = (
            self.cfgs_loss["l2_loss_weight"] * l2_loss
            + self.cfgs_loss["l1_loss_weight"] * l1_loss
            + self.cfgs_loss["smooth_l1_loss_weight"] * smooth_l1_loss
            + self.cfgs_loss["cos_loss_weight"] * cos_loss
        )
        return {
            "loss": loss.mean(),
            "l2_loss": l2_loss.mean(),
            "l1_loss": l1_loss.mean(),
            "smooth_l1_loss": smooth_l1_loss.mean(),
            "cos_loss": cos_loss.mean(),
        }

    # ── Autoregressive rollout loss (Eq. 2) ─────────────
    def rollout_loss(self, video_features, actions, lang_emb=None,
                     rollout_steps=1, ctxt_window=3,
                     tf_loss_weight=0.0, seed_frames=1):
        """
        Autoregressive rollout loss with optional teacher-forcing anchor.

        At each rollout step we compute TWO losses against the same GT frame:
          - rollout_loss: from predictions built on the rolled buffer
                          (mix of GT seed frames + model's own predictions)
          - tf_loss:      from predictions built on the GT prefix only
                          (anchors single-step prediction quality)

        Per-step combined loss = (1 - w) * rollout + w * tf, where w = tf_loss_weight.
        Set tf_loss_weight=0.0 to get the original pure-rollout behaviour.

        Args:
            video_features: (B, T, V, H, W, D) — frozen encoder outputs (GT)
            actions:        (B, T, action_dim)
            lang_emb:       (B, language_dim) or None — static across frames
            rollout_steps:  number of autoregressive steps
            ctxt_window:    max context frames fed to predictor per step
            tf_loss_weight: 0..1 weight on the teacher-forced anchor
            seed_frames:    number of GT frames used to seed the rolled buffer
                            (1 = hardest; 2-3 gives real temporal context)

        Returns:
            dict with "loss" (scalar), aggregated component losses, AND
            per-step breakdowns under keys "step{i}/<component>" for i in
            range(rollout_steps).
        """
        B, T, V, H, W, D = video_features.shape
        seed_frames = max(1, int(seed_frames))
        # Need: seed_frames GT + rollout_steps future GT frames.
        max_start = T - rollout_steps - seed_frames
        if max_start < 0:
            logging.warning(
                f"Not enough frames for rollout (T={T}, seed={seed_frames}, "
                f"rollout_steps={rollout_steps}) — falling back to teacher-forcing"
            )
            return self.compute_loss(
                self.forward_pred(video_features, actions, lang_emb),
                video_features,
                shift=1,
            )

        t_start = 0
        # Seed buffer with `seed_frames` GT frames.
        rolled = [video_features[:, t_start : t_start + seed_frames].detach()]

        total_loss = torch.tensor(0.0, device=video_features.device)
        agg = {"l2_loss": 0.0, "l1_loss": 0.0,
               "smooth_l1_loss": 0.0, "cos_loss": 0.0}
        per_step = {}   # e.g. per_step["step0/l2_loss"] -> float

        for step in range(rollout_steps):
            # Absolute time index of the GT target for this step:
            #   seed_frames GT frames at positions [0, seed_frames),
            #   then rollout step 0 predicts position seed_frames,
            #   step 1 predicts seed_frames+1, etc.
            tgt_idx = t_start + seed_frames + step
            gt_frame = video_features[:, tgt_idx : tgt_idx + 1]

            # ── (a) Rollout branch: context = seeded GT + own predictions ──
            buf = torch.cat(rolled, dim=1)                      # (B, n_ctx, V, H, W, D)
            ctx_start = max(0, buf.shape[1] - ctxt_window)
            ctx_feats = buf[:, ctx_start:]
            act_start = t_start + ctx_start
            ctx_acts = actions[:, act_start : act_start + ctx_feats.shape[1]]

            pred_next = self.forward_pred(ctx_feats, ctx_acts, lang_emb)
            pred_last = pred_next[:, -1:]
            roll_step = self.compute_loss(pred_last, gt_frame, shift=0)

            # ── (b) Teacher-forcing branch: context = pure GT prefix ───────
            if tf_loss_weight > 0.0:
                # Same ctxt_window, but fed GT frames at the same absolute times
                # as (a) used the rolled buffer. That keeps action alignment identical.
                tf_ctx_feats = video_features[
                    :, act_start : act_start + ctx_feats.shape[1]
                ].detach()
                tf_pred_next = self.forward_pred(tf_ctx_feats, ctx_acts, lang_emb)
                tf_pred_last = tf_pred_next[:, -1:]
                tf_step = self.compute_loss(tf_pred_last, gt_frame, shift=0)
                step_loss = (1.0 - tf_loss_weight) * roll_step["loss"] \
                          + tf_loss_weight * tf_step["loss"]
            else:
                tf_step = None
                step_loss = roll_step["loss"]

            total_loss = total_loss + step_loss

            # Aggregated (averaged over steps at end).
            for k in agg:
                agg[k] += roll_step[k].item()

            # Per-step logging — rollout-branch numbers are what we care about
            # for diagnosing long-horizon quality.
            for k in ("l2_loss", "cos_loss"):
                per_step[f"step{step}/{k}"] = roll_step[k].item()
            if tf_step is not None:
                per_step[f"step{step}/tf_l2_loss"] = tf_step["l2_loss"].item()

            # Append prediction to rolled buffer (gradients flow through).
            rolled.append(pred_last)

        n = float(rollout_steps)
        out = {
            "loss": total_loss / n,
            "l2_loss": torch.tensor(agg["l2_loss"] / n),
            "l1_loss": torch.tensor(agg["l1_loss"] / n),
            "smooth_l1_loss": torch.tensor(agg["smooth_l1_loss"] / n),
            "cos_loss": torch.tensor(agg["cos_loss"] / n),
        }
        out.update(per_step)
        return out

    # ── Action-contrastive InfoNCE loss (ACT variant, Eq. 5) ──────────────
    def contrastive_action_loss(self, gt_feats, actions, lang_emb=None,
                            rollout_steps=3, ctxt_window=4, seed_frames=2,
                            temperature=0.1, n_perm_negs=3):
        
        """Multi-step order-sensitive InfoNCE (the ACT variant studied in the paper).

        Positive: rollout under GT actions.
        Negatives: rollouts under temporal permutations of the same actions
        (same scene, same action multiset, wrong order) plus the reversed sequence.
        """
        B, T, V, H, W, D = gt_feats.shape
        device = gt_feats.device
        seed_frames = max(1, int(seed_frames))
        if B < 2 or T < seed_frames + rollout_steps:
            return torch.tensor(0.0, device=device), 0.0, 0.0

        def _rollout(acts):
            rolled = [gt_feats[:, :seed_frames].detach()]
            for step in range(rollout_steps):
                buf       = torch.cat(rolled, dim=1)
                ctx_start = max(0, buf.shape[1] - ctxt_window)
                ctx_feats = buf[:, ctx_start:]
                ctx_acts  = acts[:, ctx_start:ctx_start + ctx_feats.shape[1]]
                pred_next = self.forward_pred(ctx_feats, ctx_acts, lang_emb)
                rolled.append(pred_next[:, -1:])
            return rolled[-1]

        # The actions that actually drive the rollout: positions seed-1 .. seed-1+K
        # (matches rollout_loss action alignment). Permute ONLY those K steps.
        a_start = seed_frames - 1
        a_end   = a_start + rollout_steps
        K = rollout_steps

        pred_pos  = _rollout(actions)
        tgt_idx   = seed_frames + rollout_steps - 1
        gt_target = gt_feats[:, tgt_idx:tgt_idx + 1]

        def _pool(t):
            t = t.reshape(B, -1, D)
            return F.normalize(F.normalize(t, dim=-1).mean(dim=1), dim=-1)

        g     = _pool(gt_target)
        p_pos = _pool(pred_pos)

        neg_pools = []
        # (a) temporal-permutation negatives — same multiset, wrong order
        for _ in range(n_perm_negs):
            acts_perm = actions.clone()
            for b in range(B):
                # only permute the K driving steps; identity perm is rejected
                while True:
                    idx = torch.randperm(K, device=device)
                    if K < 2 or not torch.equal(idx, torch.arange(K, device=device)):
                        break
                acts_perm[b, a_start:a_end] = actions[b, a_start:a_end][idx]
            neg_pools.append(_pool(_rollout(acts_perm)))

        # (b) reversed negative — hard case the oracle showed the WM fails
        acts_rev = actions.clone()
        acts_rev[:, a_start:a_end] = torch.flip(actions[:, a_start:a_end], dims=[1])
        neg_pools.append(_pool(_rollout(acts_rev)))

        # (c) keep ONE cross-scene negative — still useful for action-image grounding
        # shift = torch.randint(1, B, (1,)).item()
        # neg_pools.append(_pool(_rollout(actions[torch.arange(B, device=device).roll(shift)])))

        logit_pos = (g * p_pos).sum(-1, keepdim=True) / temperature      # (B,1)
        # each negative: per-sample dot product (same-scene), so diagonal-style
        logit_negs = [ (g * pn).sum(-1, keepdim=True) / temperature for pn in neg_pools ]
        logits  = torch.cat([logit_pos] + logit_negs, dim=1)             # (B, 1+n_neg)
        targets = torch.zeros(B, dtype=torch.long, device=device)
        loss    = F.cross_entropy(logits, targets)

        sim_pos = logit_pos.squeeze(1).mean().item() * temperature
        # report the permutation negative specifically — this is the metric that matters
        sim_off = logit_negs[0].squeeze(1).mean().item() * temperature
        return loss, sim_pos, sim_off

    # ── Monotone Cost Ranking (MCR) loss ──────────────────────────────────
    def mcr_loss(self, gt_feats, actions, lang_emb=None,
                            rollout_steps=6, ctxt_window=4, seed_frames=4,
                            n_perturb=8, perturb_stds=(0.05, 0.15, 0.35),
                            temporal_corr=0.7, margin_scale=0.5, margin_min=0.02,
                            rollout_loss_weight=1.0):
        """
        Monotone Cost Ranking (MCR) loss (Eqs. 3-4).

        Generates n_perturb perturbed action sequences around GT at graded noise
        magnitudes, dead-reckons each to a trajectory, and trains the WM cost to be
        MONOTONE in trajectory-distance-from-GT via a pairwise margin-ranking loss.

        Returns: (loss, mean_rank_violation, gt_cost) for logging.
        """
        B, T, V, H, W, D = gt_feats.shape
        device = gt_feats.device
        seed_frames = max(1, int(seed_frames))
        K = rollout_steps
        if B < 1 or T < seed_frames + K:
            return torch.tensor(0.0, device=device), 0.0, 0.0

        a_start = seed_frames - 1
        a_end   = a_start + K

        def _rollout(acts):
            rolled = [gt_feats[:, :seed_frames].detach()]
            for step in range(K):
                buf       = torch.cat(rolled, dim=1)
                ctx_start = max(0, buf.shape[1] - ctxt_window)
                ctx_feats = buf[:, ctx_start:]
                ctx_acts  = acts[:, ctx_start:ctx_start + ctx_feats.shape[1]]
                pred_next = self.forward_pred(ctx_feats, ctx_acts, lang_emb)
                rolled.append(pred_next[:, -1:])
            return rolled[-1]

        def _pool(t):
            t = t.reshape(B, -1, D)
            return F.normalize(F.normalize(t, dim=-1).mean(dim=1), dim=-1)

        tgt_idx   = seed_frames + K - 1
        gt_target = gt_feats[:, tgt_idx:tgt_idx + 1]
        g         = _pool(gt_target)

        # ── (1) Anchor: keep the WM an accurate predictor ────────────────────
        # Without this, the surface gets smoothly ordered around a drifting WM.
        pred_pos    = _rollout(actions)
        anchor_loss = self.compute_loss(pred_pos, gt_target, shift=0)["loss"]

        # ── (2) Build graded perturbations of the GT driving actions ─────────
        gt_drive = actions[:, a_start:a_end]                       # (B, K, A)
        cand_acts  = [actions]                                     # idx 0 = GT itself
        for std in perturb_stds:
            per_std = max(1, n_perturb // len(perturb_stds))
            for _ in range(per_std):
                # correlated noise — same shape CEM searches with
                noise = _sample_correlated_noise(B, K, 1, gt_drive.shape[-1],
                                                temporal_corr, device).squeeze(2)
                acts_p = actions.clone()
                acts_p[:, a_start:a_end] = (gt_drive + std * noise).clamp(-1., 1.)
                cand_acts.append(acts_p)

        # ── (3) Cost + trajectory distance for every candidate ───────────────
        costs   = []   # (n_cand,) list of (B,) WM feature costs — DIFFERENTIABLE
        tdists  = []   # (n_cand,) list of (B,) trajectory distances — DETACHED
        for ca in cand_acts:
            pred  = _rollout(ca)
            p     = _pool(pred)
            cost  = 1.0 - (g * p).sum(-1)                          # (B,) cosine cost
            costs.append(cost)
            with torch.no_grad():
                d = self._traj_dist(ca[:, a_start:a_end], gt_drive)  # (B,) detached
            tdists.append(d)

        costs  = torch.stack(costs,  dim=1)                        # (B, n_cand) grad
        tdists = torch.stack(tdists, dim=1)                        # (B, n_cand) detached

        # ── (4) Pairwise margin-ranking: dist_i > dist_j  ⇒  cost_i > cost_j ──
        # margin scales with the trajectory-distance GAP between the pair.
        n_cand = costs.shape[1]
        ci = costs.unsqueeze(2)                                    # (B, n_cand, 1)
        cj = costs.unsqueeze(1)                                    # (B, 1, n_cand)
        di = tdists.unsqueeze(2)
        dj = tdists.unsqueeze(1)

        dist_gap = di - dj                                         # (B, n_cand, n_cand)
        pair_mask = (dist_gap > 1e-4).float()                      # i strictly farther
        margin    = (margin_min + margin_scale * dist_gap).clamp(min=0.0)
        # want cost_i >= cost_j + margin  → penalize  margin + cost_j - cost_i  > 0
        violation = F.relu(margin + cj - ci) * pair_mask
        denom     = pair_mask.sum().clamp(min=1.0)
        rank_loss = violation.sum() / denom

        loss = rollout_loss_weight * anchor_loss + rank_loss

        with torch.no_grad():
            mean_viol = (violation.sum() / denom).item()
            gt_cost   = costs[:, 0].mean().item()
        return loss, mean_viol, gt_cost

    @staticmethod
    def _traj_dist(acts_a, acts_b):
        """Dead-reckon both action sequences, return mean L2 path distance (B,)."""
        def _reckon(acts):
            B, K, _ = acts.shape
            pos = torch.zeros(B, 2, device=acts.device)
            theta = torch.zeros(B, device=acts.device)
            positions = []
            for t in range(K):
                dx_l, dy_l = acts[:, t, 0], acts[:, t, 1]
                cos_t, sin_t = theta.cos(), theta.sin()
                pos = torch.stack([
                    pos[:, 0] + cos_t * dx_l - sin_t * dy_l,
                    pos[:, 1] + sin_t * dx_l + cos_t * dy_l,
                ], dim=1)
                positions.append(pos)
                theta = theta + torch.atan2(acts[:, t, 2], acts[:, t, 3])
            return torch.stack(positions, dim=1)  # (B, K, 2)
        pa = _reckon(acts_a)
        pb = _reckon(acts_b)
        return (pa - pb).norm(dim=-1).mean(dim=1)  # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════════════════

def build_encoder(cfg):
    if cfg.encoder_type == "dino":
        return DinoV2EncoderPt(
            model_name=cfg.enc_model_name, freeze=True, img_norm_type="imagenet",
        )
    elif cfg.encoder_type == "dinov3":
        return DinoV3EncoderPt(
            model_name=cfg.enc_model_name, freeze=True, img_norm_type="imagenet",
        )
    elif cfg.encoder_type == "vjepa":
        return VJEPA2EncoderPt(
            model_name=cfg.enc_model_name, freeze=True, img_norm_type="imagenet",
            force_resolution=cfg.img_size,
        )
    raise ValueError(f"Unknown encoder_type: {cfg.encoder_type}")


def build_predictor(cfg):
    if cfg.pred_type == "AdaLN":
        from lmwm.models.AdaLN_vit import vit_predictor_AdaLN
        return vit_predictor_AdaLN(
            img_size=(cfg.img_size, cfg.img_size),
            patch_size=cfg.patch_size,
            num_frames=cfg.window_size,
            tubelet_size=1,
            embed_dim=cfg.embed_dim,
            predictor_embed_dim=cfg.pred_embed_dim,
            depth=cfg.pred_depth,
            num_heads=cfg.pred_num_heads,
            mlp_ratio=4.0,
            use_rope=cfg.use_rope,
            is_causal=cfg.is_causal,
            use_silu=cfg.use_silu,
            action_dim=cfg.action_dim,
            action_encoder_inpred=cfg.action_encoder_inpred,
            action_norm=getattr(cfg, "action_norm", False),
            action_conditioning=getattr(cfg, "action_conditioning", "adaln"),
            proprio_dim=cfg.proprio_dim,
            use_proprio=cfg.use_proprio,
            proprio_encoding=cfg.proprio_encoding,
            proprio_emb_dim=cfg.proprio_emb_dim,
            proprio_tokens=cfg.proprio_tokens,
            proprio_encoder_inpred=cfg.proprio_encoder_inpred,
            init_scale_factor_adaln=cfg.init_scale_factor_adaln,
        )
    elif cfg.pred_type == "vit":
        # DINO-WM baseline predictor.
        from lmwm.models.vit_predictor_wm import ViTPredictorWM
        return ViTPredictorWM(
            num_patches=cfg.grid_size ** 2,
            num_frames=cfg.window_size,        # NOT ctxt_window
            embed_dim=cfg.embed_dim,
            depth=cfg.pred_depth,
            heads=cfg.pred_num_heads,
            mlp_dim=cfg.pred_mlp_dim,
            dim_head=cfg.pred_dim_head,
            action_dim=cfg.action_dim,
            action_emb_dim=cfg.action_emb_dim,
            use_sdpa=True,
        )
    raise ValueError(f"Unknown pred_type: {cfg.pred_type}")


def build_world_model(cfg):
    return LatentWorldModel(build_encoder(cfg), build_predictor(cfg), cfg)
