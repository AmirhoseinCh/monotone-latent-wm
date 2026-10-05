#!/usr/bin/env python3
"""
Evaluate a trained latent world model with CEM planning on the GNM / CAST
held-out split.

Navigation metrics (over the K-step planned trajectory):
    AOE   Average Orientation Error (deg)
    MAOE  Max Orientation Error (deg): worst-case per trajectory
    ADE   Average Displacement Error (m)
    MADE  Max Displacement Error (m): worst-case per trajectory

Planning: CEM with the world model as the dynamics function.
    - Sample N candidate action sequences (optionally from a unicycle prior)
    - Roll out the WM in latent space; score by cosine cost to the goal features
    - Refit mean/std to the elites, repeat for the given number of iterations
    - Unnormalize best actions -> trajectory -> metrics

The model config is read from config.py next to the checkpoint.

Usage (paper settings, see scripts/eval.sh):
    python eval_planning.py --checkpoint <run_dir>/final_model.pt --encoder dino \\
        --dataset_name cast_filtered_dataset --split val --plan_horizon 6 --unicycle
"""

import argparse
import importlib.util
import json
import os
import queue
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

from lmwm.world_model import LatentWorldModel, build_world_model
from lmwm.data.dataset import make_dataset_from_rlds, make_test_dataset
from lmwm.data.utils.cast_transforms import gnm_action_angle_dataset_transform
from lmwm.data.utils.data_utils import NormalizationType
from lmwm.data.spec import ModuleSpec
from lmwm.data.torch_rlds_dataset import TorchRLDSDataset, collate_fn


# ═══════════════════════════════════════════════════════════════════════════════
# Prefetch helper
# ═══════════════════════════════════════════════════════════════════════════════

class PrefetchLoader:
    """Runs the DataLoader in a background thread so CPU preprocessing overlaps
    with GPU computation.  Works with RLDS/TF datasets (not picklable) because
    threading shares the process address space instead of forking.
    """

    def __init__(self, iterable, maxsize: int = 2):
        self._queue = queue.Queue(maxsize=maxsize)
        self._sentinel = object()
        self._thread = threading.Thread(
            target=self._fill, args=(iterable,), daemon=True
        )
        self._thread.start()

    def _fill(self, iterable):
        for batch in iterable:
            self._queue.put(batch)
        self._queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        return item


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory / metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_xy_trajectory_gt(act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (T, 4) action sequence [local_x, local_y, sin_yaw, cos_yaw] → (T+1, 2) xy."""
    heading = 0.0
    xy = [np.array([0.0, 0.0])]
    headings = [0.0]
    for t in range(act.shape[0]):
        dx_l, dy_l = act[t, 0], act[t, 1]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        xy.append(xy[-1] + np.array([cos_h * dx_l - sin_h * dy_l,
                                     sin_h * dx_l + cos_h * dy_l]))
        heading += np.arctan2(act[t, 2], act[t, 3])
        headings.append(heading)
    return np.array(xy), np.array(headings)


def unnormalize_bound_action(action: np.ndarray, unnorm_stats: dict) -> np.ndarray:
    """Undo BOUNDS normalization [-1, 1] → metric space."""
    p01 = np.array(unnorm_stats["p01"]).squeeze()
    p99 = np.array(unnorm_stats["p99"]).squeeze()
    if "mask" in unnorm_stats:
        mask = np.array(unnorm_stats["mask"]).squeeze().astype(bool)
    else:
        mask = np.ones_like(p01, dtype=bool)
    action = action[..., :len(mask)].copy()
    action = np.where(mask, (action + 1) * (p99 - p01) / 2 + p01, action)
    return action


def _renorm_gt_to_ckpt(
    acts: torch.Tensor,      # (..., A) normalized with gt_stats (all dims BOUNDS, no mask)
    gt_act_stats: dict,      # fresh dataset stats — no mask
    ckpt_act_stats: dict,    # checkpoint stats — may have mask=[T,T,F,F]
    device: torch.device,
) -> torch.Tensor:
    """Re-normalize actions from gt-stats space to ckpt-stats space.

    Loader uses fresh (gt) stats to normalize GT actions; the WM was trained with
    checkpoint stats (and possibly a sin/cos pass-through mask).  This converts
    seed actions so they match the distribution the model saw during training.
    """
    A = acts.shape[-1]
    def _t(arr):
        return torch.tensor(np.array(arr).flatten()[:A], dtype=torch.float32, device=device)

    p01_gt, p99_gt   = _t(gt_act_stats["p01"]),   _t(gt_act_stats["p99"])
    p01_ck, p99_ck   = _t(ckpt_act_stats["p01"]),  _t(ckpt_act_stats["p99"])

    raw  = (acts.float() + 1.0) * (p99_gt - p01_gt) / 2.0 + p01_gt
    norm = (2.0 * (raw - p01_ck) / (p99_ck - p01_ck + 1e-8) - 1.0).clamp(-1., 1.)

    _mask = ckpt_act_stats.get("mask", None)
    if _mask is not None:
        mask_t = torch.tensor(np.array(_mask).flatten()[:A].astype(bool), device=device)
        # Unmasked dims (sin/cos pass-through): use raw value clamped to [-1, 1]
        norm = torch.where(mask_t, norm, raw.clamp(-1., 1.))

    return norm.to(acts.dtype)


def compute_nav_metrics(
    xy_pred: np.ndarray, xy_gt: np.ndarray
) -> Dict[str, float]:
    """
    Compute AOE / MAOE / ADE / MADE between predicted and GT trajectories.

    Args:
        xy_pred: (T+1, 2) predicted cumulative trajectory (origin = [0, 0])
        xy_gt:   (T+1, 2) GT cumulative trajectory (origin = [0, 0])
    """
    wp_pred = xy_pred[1:]  # (T, 2) — skip the shared origin
    wp_gt = xy_gt[1:]

    # Displacement error per waypoint
    dists = np.linalg.norm(wp_pred - wp_gt, axis=-1)  # (T,)
    ade = float(dists.mean())
    made = float(dists.max())

    # Orientation error: angle between direction-to-waypoint vectors
    pred_n = np.linalg.norm(wp_pred, axis=-1, keepdims=True).clip(1e-8)
    gt_n = np.linalg.norm(wp_gt, axis=-1, keepdims=True).clip(1e-8)
    cos_sim = np.clip((wp_pred / pred_n * wp_gt / gt_n).sum(-1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(cos_sim))  # (T,)
    aoe = float(angles_deg.mean())
    maoe = float(angles_deg.max())

    return {"AOE": aoe, "MAOE": maoe, "ADE": ade, "MADE": made}


def compute_xy_trajectory_batch(acts_B: np.ndarray, headings_B: np.ndarray = None):
    """Vectorized dead-reckoning for a batch of action sequences.

    acts_B: (B, T, 4) — [local_x, local_y, sin_yaw, cos_yaw]
    Returns: (B, T+1, 2) cumulative XY positions, (B, T+1) heading at each step.
    """
    B, T = acts_B.shape[:2]
    xy = np.zeros((B, T + 1, 2))
    heading = np.zeros(B) if headings_B is None else headings_B[:, 0].copy()
    all_headings = np.zeros((B, T + 1))
    all_headings[:, 0] = heading
    for t in range(T):
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        dx_l, dy_l = acts_B[:, t, 0], acts_B[:, t, 1]
        xy[:, t + 1, 0] = xy[:, t, 0] + cos_h * dx_l - sin_h * dy_l
        xy[:, t + 1, 1] = xy[:, t, 1] + sin_h * dx_l + cos_h * dy_l
        heading += np.arctan2(acts_B[:, t, 2], acts_B[:, t, 3])
        all_headings[:, t + 1] = heading
    return xy, all_headings

def rotate_to_initial_heading(xy, headings):
    """Rotate trajectories so the initial heading faces "forward" (x-axis).

    Supports single trajectories (T, 2) with headings (T,) and batches
    (B, T, 2) with headings (B, T).
    """
    xy = np.asarray(xy)
    headings = np.asarray(headings)

    if xy.ndim == 2:
        h0 = float(headings.reshape(-1)[0])
        c, s = np.cos(-h0), np.sin(-h0)
        x = xy[:, 0]
        y = xy[:, 1]
        xy_rot = np.stack([c * x - s * y, s * x + c * y], axis=-1)
        headings_rot = (headings.reshape(-1) - h0) % (2 * np.pi)
        return xy_rot, headings_rot

    if xy.ndim != 3:
        raise ValueError(f"xy must be (T, 2) or (B, T, 2); got {xy.shape}")

    # Batch rotation: per-example initial heading.
    h0 = headings[:, 0].reshape(-1, 1)  # (B, 1)
    c = np.cos(-h0)
    s = np.sin(-h0)
    x = xy[..., 0]
    y = xy[..., 1]
    xy_rot = np.stack([c * x - s * y, s * x + c * y], axis=-1)
    headings_rot = (headings - h0) % (2 * np.pi)
    return xy_rot, headings_rot

def compute_nav_metrics_batch(
    xy_pred_B: np.ndarray, xy_gt_B: np.ndarray
) -> Dict[str, np.ndarray]:
    """Vectorized AOE/MAOE/ADE/MADE for a batch.

    xy_pred_B, xy_gt_B: (B, T+1, 2)
    Returns: dict of (B,) float arrays.
    """
    wp_pred = xy_pred_B[:, 1:]   # (B, T, 2)
    wp_gt   = xy_gt_B[:, 1:]

    dists = np.linalg.norm(wp_pred - wp_gt, axis=-1)   # (B, T)
    ade   = dists.mean(axis=-1)
    made  = dists.max(axis=-1)

    pred_n  = np.linalg.norm(wp_pred, axis=-1, keepdims=True).clip(1e-8)
    gt_n    = np.linalg.norm(wp_gt,   axis=-1, keepdims=True).clip(1e-8)
    cos_sim = np.clip((wp_pred / pred_n * wp_gt / gt_n).sum(-1), -1.0, 1.0)
    angles  = np.degrees(np.arccos(cos_sim))            # (B, T)
    aoe     = angles.mean(axis=-1)
    maoe    = angles.max(axis=-1)

    return {"AOE": aoe, "MAOE": maoe, "ADE": ade, "MADE": made}


def compute_nav_metrics_batch_yaw(
    xy_pred_B: np.ndarray,
    xy_gt_B: np.ndarray,
    headings_pred_B: np.ndarray,
    headings_gt_B: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Vectorized AOE/MAOE/ADE/MADE using heading deltas for orientation.

    xy_pred_B, xy_gt_B: (B, T+1, 2)
    headings_pred_B, headings_gt_B: (B, T+1) in radians
    Returns: dict of (B,) float arrays.
    """
    wp_pred = xy_pred_B[:, 1:]   # (B, T, 2)
    wp_gt   = xy_gt_B[:, 1:]

    dists = np.linalg.norm(wp_pred - wp_gt, axis=-1)   # (B, T)
    ade   = dists.mean(axis=-1)
    made  = dists.max(axis=-1)

    # Orientation error: absolute yaw difference per waypoint.
    yaw_pred = headings_pred_B[:, 1:]
    yaw_gt   = headings_gt_B[:, 1:]
    yaw_diff = (yaw_pred - yaw_gt + np.pi) % (2 * np.pi) - np.pi
    angles   = np.degrees(np.abs(yaw_diff))
    aoe      = angles.mean(axis=-1)
    maoe     = angles.max(axis=-1)

    return {"AOE": aoe, "MAOE": maoe, "ADE": ade, "MADE": made}


# ─── PCA feature visualization ────────────────────────────────────────────────

class PCAVisualizer:
    """Projects [..., D] patch features to an RGB image via PCA."""

    def __init__(self, features: np.ndarray, n_samples: int = 100_000,
                 n_components: int = 3, whiten: bool = False,
                 normalize: bool = True) -> None:
        from sklearn import decomposition
        features = np.array(features).reshape([-1, features.shape[-1]])
        if features.shape[0] > n_samples:
            idx = np.random.randint(0, features.shape[0], n_samples)
            features = features[idx]
        pca = decomposition.PCA(n_components=n_components, whiten=whiten)
        pca.fit(features)
        self.pca = pca
        self.normalize = normalize

    def __call__(self, features: np.ndarray) -> np.ndarray:
        features = np.array(features)
        out = self.pca.transform(
            features.reshape([-1, features.shape[-1]])
        ).reshape(features.shape[:-1] + (3,))
        if self.normalize:
            norms = np.linalg.norm(out, ord=2, axis=-1, keepdims=True).clip(min=1e-3)
            out = out / norms * 0.5 + 0.5
        return out.clip(0.0, 1.0)


def plot_last_example(
    img_first: np.ndarray,       # (C, H, W) uint8
    img_goal: np.ndarray,        # (C, H, W) uint8
    img_middle: np.ndarray,       # (C, H, W) uint8
    xy_gt: np.ndarray,           # (T+1, 2) metric trajectory (from unnorm actions)
    xy_pred: np.ndarray,         # (T+1, 2) metric trajectory (from unnorm actions)
    xy_state: np.ndarray,        # (T, 2) metric trajectory from observation/proprio
    headings_gt: np.ndarray,     # (T+1,) heading at each GT waypoint (radians)
    headings_pred: np.ndarray,   # (T+1,) heading at each predicted waypoint (radians)
    headings_state: np.ndarray,  # (T,)   heading at each state waypoint (radians)
    gt_actions: np.ndarray,      # (T, >=4) unnormalized metric actions
    pred_actions: np.ndarray,    # (T, >=4) unnormalized metric actions
    language: str,
    sample_actions: Optional[np.ndarray] = None,  # (T, N, A) unnormalized metric actions
    sample_costs: Optional[np.ndarray] = None,    # (N,) planner costs (lower is better)
    feat_first: Optional[np.ndarray] = None,       # (gH, gW, D) encoder features of first/seed frame
    feat_goal_gt: Optional[np.ndarray] = None,    # (gH, gW, D) encoder features of actual goal
    feat_goal_pred: Optional[np.ndarray] = None,  # (gH, gW, D) WM-predicted final features
    iter_xy_trajs: Optional[list] = None,            # list of (T+1, 2) arrays, one per planner iter
    iter_costs: Optional[np.ndarray] = None,         # (n_iters,) min cost at each planner iter
    first_iter_sample_xys: Optional[list] = None,   # list of (T+1, 2) arrays, one per first-iter sample
    save_path: Optional[str] = None,
) -> None:
    """Plot first/goal frames, top-down trajectory, and per-dim action comparison."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def to_hwc(img: np.ndarray) -> np.ndarray:
        img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    def draw_heading_arrows(ax, xy, headings, color, arrow_len):
        for ti in range(len(xy)):
            ax.annotate(
                "",
                xy=(xy[ti, 0] + arrow_len * np.cos(headings[ti]),
                    xy[ti, 1] + arrow_len * np.sin(headings[ti])),
                xytext=(xy[ti, 0], xy[ti, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            )

    has_iter = iter_xy_trajs is not None and len(iter_xy_trajs) > 0
    fig = plt.figure(figsize=(25, 14 if has_iter else 10))
    gs = gridspec.GridSpec(
        3 if has_iter else 2, 5, figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=([2, 2, 1.2] if has_iter else [2, 2]),
    )

    # ── Row 0: frames + metric trajectory + PCA features ─────────────────
    ax_first     = fig.add_subplot(gs[0, 0])
    ax_middle    = fig.add_subplot(gs[0, 1])
    ax_goal      = fig.add_subplot(gs[0, 2])
    ax_traj      = fig.add_subplot(gs[0, 3])
    ax_gt_pca    = fig.add_subplot(gs[0, 4])
    ax_first_pca = fig.add_subplot(gs[1, 3])
    ax_pred_pca  = fig.add_subplot(gs[1, 4])


    ax_first.imshow(to_hwc(img_first))
    ax_first.set_title("First frame  (t = 0)", fontsize=11)
    ax_first.axis("off")

    ax_middle.imshow(to_hwc(img_middle))
    ax_middle.set_title("Middle frame  (t = {xy_gt.shape[0] // 2})", fontsize=11)
    ax_middle.axis("off")

    ax_goal.imshow(to_hwc(img_goal))
    ax_goal.set_title(f"Goal frame  (t = {xy_gt.shape[0] - 1})", fontsize=11)
    ax_goal.axis("off")

    # Arrow length: ~30% of the mean step size across all three trajectories
    all_steps = np.concatenate([
        np.linalg.norm(np.diff(xy_gt,   axis=0), axis=-1),
        np.linalg.norm(np.diff(xy_pred, axis=0), axis=-1),
        np.linalg.norm(np.diff(xy_state, axis=0), axis=-1) if len(xy_state) > 1 else [],
    ])
    arrow_len = float(np.median(all_steps[all_steps > 1e-6])) * 0.5 if len(all_steps) else 0.1

    # Metric trajectory
    ax_traj.plot(xy_gt[:, 0],     xy_gt[:, 1],     "g.-",  linewidth=2, markersize=4,  label="GT actions")
    ax_traj.plot(xy_pred[:, 0],   xy_pred[:, 1],   "r.-",  linewidth=2, markersize=4,  label="Planned")
    ax_traj.plot(xy_state[:, 0],  xy_state[:, 1],  "b--o", linewidth=1, markersize=4,  label="GT state")
    ax_traj.plot(0, 0, "ko", markersize=6, zorder=5, label="Start")
    ax_traj.plot(xy_gt[-1, 0],    xy_gt[-1, 1],    "g*",  markersize=7, zorder=5)
    ax_traj.plot(xy_pred[-1, 0],  xy_pred[-1, 1],  "r*",  markersize=7, zorder=5)
    ax_traj.plot(xy_state[-1, 0], xy_state[-1, 1], "b*",  markersize=7, zorder=5)

    if first_iter_sample_xys is not None:
        for _xy_s in first_iter_sample_xys:
            ax_traj.plot(_xy_s[:, 0], _xy_s[:, 1],
                         color="orange", alpha=0.15, linewidth=0.7, zorder=1)

    if has_iter:
        n_iters = len(iter_xy_trajs)
        _iter_cmap = plt.cm.Blues
        for _it, _xy_it in enumerate(iter_xy_trajs):
            _alpha = 0.3 + 0.7 * (_it / max(n_iters - 1, 1))
            _color = _iter_cmap(0.3 + 0.7 * (_it / max(n_iters - 1, 1)))
            ax_traj.plot(
                _xy_it[:, 0], _xy_it[:, 1],
                color=_color, alpha=_alpha, linewidth=1.2, zorder=2,
                label=f"Iter {_it}" if _it in (0, n_iters - 1) else None,
            )

    if headings_gt is not None:
        draw_heading_arrows(ax_traj, xy_gt,   headings_gt,    "green",  arrow_len)
    if headings_pred is not None:
        draw_heading_arrows(ax_traj, xy_pred, headings_pred,  "red",    arrow_len)
    if headings_state is not None:
        draw_heading_arrows(ax_traj, xy_state, headings_state, "blue",  arrow_len)

    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    # ax_traj.legend(fontsize=9)
    ax_traj.set_xlabel("x_fwd (m)")
    ax_traj.set_ylabel("y_left (m)")
    ax_traj.set_title("Top-down trajectory (metric)", fontsize=11)

    

    # ── Row 1: per-dim action comparison ─────────────────────────────────
    dim_names = ["x_fwd (m)", "y_left (m)", "Δyaw)"]
    # set the delta yaw instead of sin and cos for easier interpretability
    gt_actions[:, 2] = np.arctan2(gt_actions[:, 2], gt_actions[:, 3])
    pred_actions[:, 2] = np.arctan2(pred_actions[:, 2], pred_actions[:, 3])
    samples = None
    sample_costs_1d = None
    best_idx = None
    if sample_actions is not None:
        samples = np.array(sample_actions)
        if samples.shape[-1] >= 4:
            samples[..., 2] = np.arctan2(samples[..., 2], samples[..., 3])
        if sample_costs is not None:
            sample_costs_1d = np.array(sample_costs).reshape(-1)
        if samples.ndim == 3 and gt_actions.ndim == 2:
            # L2 distance to GT action sequence (x, y, yaw)
            gt_ref = gt_actions[:, :3]
            dists = ((samples[:, :, :3] - gt_ref[:, None, :]) ** 2).sum(axis=(0, 2))
            if dists.size:
                best_idx = int(np.argmin(dists))
    steps = np.arange(gt_actions.shape[0])
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        if samples is not None:
            n_samp = samples.shape[1]
            if sample_costs_1d is not None and len(sample_costs_1d) == n_samp:
                ranks = np.argsort(np.argsort(sample_costs_1d))
                colors = plt.cm.Reds(ranks / max(n_samp - 1, 1))
            else:
                colors = np.repeat(["gray"], n_samp)
            for s in range(n_samp):
                ax.plot(
                    steps,
                    samples[:, s, i],
                    color=colors[s],
                    alpha=0.2,
                    lw=0.7,
                )
            if best_idx is not None:
                ax.plot(
                    steps,
                    samples[:, best_idx, i],
                    color="black",
                    alpha=0.9,
                    lw=2.0,
                    label="Best L2" if i == 0 else None,
                )
            for t in range(samples.shape[0]):
                ax.scatter(
                    np.full(n_samp, t),
                    samples[t, :, i],
                    s=10,
                    alpha=0.25,
                    c=colors,
                    label="Samples" if (t == 0 and i == 0) else None,
                )
        ax.plot(steps, gt_actions[:, i],   "g.-", linewidth=2, markersize=4, label="GT")
        ax.plot(steps, pred_actions[:, i], "r.-", linewidth=2, markersize=4, label="Planned")
        ax.set_title(dim_names[i], fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    # ── Row 2: per-iteration cost ──────────────────────────────────────────
    if has_iter:
        ax_cost = fig.add_subplot(gs[2, :3])
        iters = np.arange(1, len(iter_xy_trajs) + 1)
        if iter_costs is not None and len(iter_costs) == len(iter_xy_trajs):
            ax_cost.plot(iters, iter_costs, "o-", color="steelblue", linewidth=2, markersize=5)
            ax_cost.set_ylabel("min cost")
        ax_cost.set_xlabel("CEM iteration")
        ax_cost.set_title("Cost vs. planner iteration", fontsize=10)
        ax_cost.grid(True, alpha=0.3)
        ax_cost.set_xticks(iters)

    # ── PCA feature visualization ──────────────────────────────────────────
    _pca_feats = [f for f in (feat_first, feat_goal_gt, feat_goal_pred) if f is not None]
    if len(_pca_feats) >= 2:
        gH, gW, D = _pca_feats[0].shape
        all_feats = np.concatenate([f.reshape(-1, D) for f in _pca_feats], axis=0)
        if not np.all(np.isfinite(all_feats)):
            print("  [warn] features contain NaN/inf — skipping PCA visualization")
            _pca_feats = []  # fall through to the axis-off block below
    if len(_pca_feats) >= 2:
        pca_vis = PCAVisualizer(all_feats)
        for ax, feat, title in (
            (ax_gt_pca,    feat_goal_gt,  "GT goal (PCA)"),
            (ax_first_pca, feat_first,    "First frame (PCA)"),
            (ax_pred_pca,  feat_goal_pred, "Pred goal (PCA)"),
        ):
            if feat is not None:
                ax.imshow(pca_vis(feat))
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    else:
        ax_gt_pca.axis("off")
        ax_first_pca.axis("off")
        ax_pred_pca.axis("off")

    max_chars = 90
    label = language if len(language) <= max_chars else language[:max_chars] + "…"
    fig.suptitle(f'"{label}"', fontsize=12, style="italic")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CEM planning  (mirrors notebook 07_world_model_single_inference_rollout.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cem_unroll(
    model: LatentWorldModel,
    z_seed_N: torch.Tensor,                    # (N, S, 1, gH, gW, D) — S >= 1 seed frames
    actions_TNA: torch.Tensor,                 # (T, N, A) — CEM-sampled actions
    lang_N: Optional[torch.Tensor],            # (N, D_lang) or None
    ctxt_window: int,
    seed_acts_SNA: Optional[torch.Tensor] = None,  # (S-1, N, A) GT actions within seed
    goal_N: Optional[torch.Tensor] = None,     # (N, 1, *spatial, D) — enables trajectory cost
    traj_cost_fn: Optional[str] = None,        # cost_fn for intermediate steps
    traj_weight: float = 0.0,                  # weight on mean intermediate cost
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Autoregressive WM rollout for N candidate action sequences.

    z_seed_N must already be expanded to N (use .expand or .repeat_interleave
    before calling). S seed frames are used to prime the context buffer before
    the T CEM-planned steps begin (S=1 reproduces the original single-frame
    behaviour).

    When goal_N and traj_weight > 0 are provided, cost is also accumulated at
    every intermediate step (steps 0..T-2) and returned as a (N,) tensor.
    Early steps have less WM rollout drift, giving a denser and cleaner signal
    that is especially helpful for orientation planning.

    Returns: (pred_final, intermed_cost)
        pred_final:    (N, 1, 1, gH, gW, D) — final predicted frame.
        intermed_cost: (N,) mean cost over intermediate steps scaled by
                       traj_weight, or None when traj_weight == 0 / goal_N is None.
    """
    T, N = actions_TNA.shape[:2]
    S = z_seed_N.shape[1]
    *spatial, D = z_seed_N.shape[2:]   # (1, gH, gW, D)
    buf_len = S + T

    # Pre-allocate feature buffer and fill with S seed frames.
    feat_buf = z_seed_N.new_empty(N, buf_len, *spatial, D)
    feat_buf[:, :S] = z_seed_N

    # Pre-allocate action buffer; fill S-1 GT actions within the seed context.
    A = actions_TNA.shape[2]
    act_buf = actions_TNA.new_empty(N, buf_len, A)
    if seed_acts_SNA is not None and S > 1:
        # seed_acts_SNA: (S-1, N, A) → transpose to (N, S-1, A)
        act_buf[:, :S - 1] = seed_acts_SNA.permute(1, 0, 2)

    pred_next: Optional[torch.Tensor] = None
    intermed_cost: Optional[torch.Tensor] = None
    use_traj = (goal_N is not None and traj_weight > 0.0 and T > 1)

    for t in range(T):
        act_buf[:, S - 1 + t] = actions_TNA[t]   # CEM action at buffer position S-1+t

        ctx_end   = S + t
        ctx_start = max(0, ctx_end - ctxt_window)
        ctx_feats = feat_buf[:, ctx_start:ctx_end]   # (N, <=ctxt, *spatial, D)
        ctx_acts  = act_buf[:, ctx_start:ctx_end]    # (N, <=ctxt, A)

        pred      = model.forward_pred(ctx_feats, ctx_acts, lang_N)
        pred_next = pred[:, -1:]                     # (N, 1, *spatial, D)

        if t < T - 1:                                # last write is unread — skip it
            feat_buf[:, S + t] = pred_next[:, 0]
            if use_traj:
                # Accumulate scalar cost vs goal — no extra feature memory needed.
                c = cem_cost(pred_next, goal_N, traj_cost_fn)
                intermed_cost = c if intermed_cost is None else intermed_cost + c

    if intermed_cost is not None:
        intermed_cost = intermed_cost * (traj_weight / (T - 1))

    return pred_next, intermed_cost  # (N, 1, *spatial, D), (N,) or None


def cem_cost(
    predicted_NTvHWD: torch.Tensor,  # (N, 1, 1, gH, gW, D) — final frame only
    goal_N: torch.Tensor,            # (N, 1, 1, gH, gW, D) — already N-matched
    cost_fn: Optional[str] = None,
) -> torch.Tensor:
    """
    Cost between the last predicted frame and the goal frame.

    goal_N must already be expanded/replicated to match N (the leading dim of
    predicted_NTvHWD). This allows the same function to handle both the
    single-example (N candidates) and batched (B*N candidates) cases.

    cost_fn: "cosine" | "mse" | "l1" | None (combined cosine+mse, default).
    Returns: (N,) — lower is better.
    """
    final = predicted_NTvHWD[:, -1].flatten(1)  # (N, v*gH*gW*D)
    g     = goal_N.flatten(1)                    # (N, v*gH*gW*D)

    if cost_fn == "cosine":
        return 1.0 - F.cosine_similarity(final, g, dim=1)
    if cost_fn == "mse":
        return ((final - g) ** 2).mean(dim=1)
    if cost_fn == "l1":
        return torch.abs(final - g).mean(dim=1)
    # default: combined cosine + mse
    return (1.0 - F.cosine_similarity(final, g, dim=1)
            + ((final - g) ** 2).mean(dim=1))


# ─── Yaw-constrained action sampling helpers ─────────────────────────────────

def _yaw_sincos_stats(p01: np.ndarray, p99: np.ndarray, sincos_masked: bool = False) -> dict:
    """BOUNDS normalisation constants for sin_yaw (dim 2) and cos_yaw (dim 3).

    sincos_masked=True when action_normalization_mask=[T,T,F,F]: sin/cos pass through
    raw (already in [-1,1]) so BOUNDS normalization was skipped during training.
    """
    return {
        "p01_sin": float(p01[2]), "p99_sin": float(p99[2]),
        "p01_cos": float(p01[3]), "p99_cos": float(p99[3]),
        "sincos_masked": sincos_masked,
    }


def _expand_yaw_to_sincos(acts_3d: torch.Tensor, ys: dict) -> torch.Tensor:
    """(…, 3) [x_norm, y_norm, yaw_rad] → (…, 4) [x_norm, y_norm, sin_out, cos_out].

    When ys["sincos_masked"] is False (default): applies BOUNDS normalisation so
    the output matches the training pipeline (sin²+cos²=1 in raw space).
    When True: sin/cos were not BOUNDS-normalized during training, so output raw
    sin/cos directly (still on the unit circle).
    """
    yaw = acts_3d[..., 2]
    if ys.get("sincos_masked", False):
        sin_n = torch.sin(yaw).clamp(-1., 1.)
        cos_n = torch.cos(yaw).clamp(-1., 1.)
    else:
        sin_n = (2.0 * (torch.sin(yaw) - ys["p01_sin"])
                 / (ys["p99_sin"] - ys["p01_sin"] + 1e-8) - 1.0).clamp(-1., 1.)
        cos_n = (2.0 * (torch.cos(yaw) - ys["p01_cos"])
                 / (ys["p99_cos"] - ys["p01_cos"] + 1e-8) - 1.0).clamp(-1., 1.)
    return torch.cat([acts_3d[..., :2], sin_n.unsqueeze(-1), cos_n.unsqueeze(-1)], dim=-1)


def _sincos_to_yaw(acts_4d: torch.Tensor, ys: dict) -> torch.Tensor:
    """(…, 4) [x_norm, y_norm, sin_norm, cos_norm] → (…, 3) [x_norm, y_norm, yaw_rad]."""
    sin_n = acts_4d[..., 2]
    cos_n = acts_4d[..., 3]
    if ys.get("sincos_masked", False):
        sin_r = sin_n
        cos_r = cos_n
    else:
        sin_r = (sin_n + 1.0) * (ys["p99_sin"] - ys["p01_sin"]) / 2.0 + ys["p01_sin"]
        cos_r = (cos_n + 1.0) * (ys["p99_cos"] - ys["p01_cos"]) / 2.0 + ys["p01_cos"]
    yaw = torch.atan2(sin_r, cos_r)
    return torch.cat([acts_4d[..., :2], yaw.unsqueeze(-1)], dim=-1)


def _unicycle_to_3d(samples: torch.Tensor, us: dict) -> torch.Tensor:
    """(…, 2) [u_metric, delta_yaw] → (…, 3) [x_norm, y_norm, delta_yaw].

    Applies the unicycle kinematic constraint: dx = u·cos(yaw), dy = u·sin(yaw),
    then BOUNDS-normalises x and y using the dataset p01/p99 percentiles stored in
    *us* so the output is compatible with _expand_yaw_to_sincos.
    """
    u, yaw = samples[..., 0], samples[..., 1]
    x_m = u * torch.cos(yaw)
    y_m = u * torch.sin(yaw)
    p01x, p99x = us["p01_x"], us["p99_x"]
    p01y, p99y = us["p01_y"], us["p99_y"]
    x_n = (2.0 * (x_m - p01x) / (p99x - p01x + 1e-8) - 1.0).clamp(-1., 1.)
    y_n = (2.0 * (y_m - p01y) / (p99y - p01y + 1e-8) - 1.0).clamp(-1., 1.)
    return torch.stack([x_n, y_n, yaw], dim=-1)


def _3d_to_unicycle(samples_3d: torch.Tensor, us: dict) -> torch.Tensor:
    """(…, 3) [x_norm, y_norm, yaw_rad] → (…, 2) [u_metric, yaw_rad].

    Inverse of _unicycle_to_3d: unnormalises x/y then computes speed magnitude.
    """
    x_n, y_n, yaw = samples_3d[..., 0], samples_3d[..., 1], samples_3d[..., 2]
    p01x, p99x = us["p01_x"], us["p99_x"]
    p01y, p99y = us["p01_y"], us["p99_y"]
    x_m = (x_n + 1.0) * (p99x - p01x) / 2.0 + p01x
    y_m = (y_n + 1.0) * (p99y - p01y) / 2.0 + p01y
    u = (x_m.pow(2) + y_m.pow(2)).sqrt()
    return torch.stack([u, yaw], dim=-1)


def _compute_action_l2_cost(
    action_samples_BTNA: torch.Tensor,
    gt_actions_BTA: torch.Tensor,
    yaw_stats: Optional[dict],
) -> torch.Tensor:
    """Mean L2 cost between sampled actions and GT actions.

    action_samples_BTNA: (B, T, N, A) in internal action space.
    gt_actions_BTA:      (B, T, 4) in ckpt-normalized action space.
    Returns: (B, N) mean L2 per step.
    """
    B, T, N, A = action_samples_BTNA.shape
    acts_BTN = action_samples_BTNA.permute(0, 2, 1, 3).reshape(B * N, T, A)
    if yaw_stats is not None:
        acts_BTN = _expand_yaw_to_sincos(acts_BTN, yaw_stats)
    acts_BN_T_A = acts_BTN.reshape(B, N, T, -1)
    gt = gt_actions_BTA
    if gt.shape[-1] != acts_BN_T_A.shape[-1]:
        gt = gt[..., :acts_BN_T_A.shape[-1]]
    diff = (acts_BN_T_A - gt[:, None]).float()
    l2 = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8).mean(dim=-1)
    return l2


def _prepare_action_corr(
    action_corr: Optional[np.ndarray],
    A: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Prepare a Cholesky factor for cross-dimension correlation.

    action_corr: (A, A) correlation matrix. If A==4 and action_corr is 3x3,
    it will be placed in the top-left with the last dim independent.
    """
    if action_corr is None:
        return None
    corr = torch.as_tensor(action_corr, device=device, dtype=torch.float32)
    if corr.ndim != 2:
        print(f"Warning: action_corr must be 2D, got shape {tuple(corr.shape)}; ignoring.")
        return None
    if corr.shape[0] == 3 and corr.shape[1] == 3 and A == 4:
        corr_full = torch.eye(A, device=device, dtype=torch.float32)
        corr_full[:3, :3] = corr
        corr = corr_full
    if corr.shape[0] != A or corr.shape[1] != A:
        print(f"Warning: action_corr shape {tuple(corr.shape)} does not match A={A}; ignoring.")
        return None
    corr = 0.5 * (corr + corr.T)
    corr = corr.clone()
    corr.fill_diagonal_(1.0)
    jitter = 1e-4
    for _ in range(3):
        try:
            return torch.linalg.cholesky(corr + jitter * torch.eye(A, device=device))
        except RuntimeError:
            jitter *= 10.0
    print("Warning: action_corr is not positive definite; ignoring.")
    return None


def _prepare_cluster_stats(
    cluster_means: Optional[np.ndarray],
    cluster_stds: Optional[np.ndarray],
    plan_horizon: int,
    A: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if cluster_means is None or cluster_stds is None:
        return None, None
    means = np.array(cluster_means)
    stds = np.array(cluster_stds)
    if means.ndim != 3 or stds.ndim != 3:
        print("Warning: cluster stats must be (K, T, A); ignoring.")
        return None, None
    k, t_src, a_src = means.shape
    if stds.shape != means.shape:
        print("Warning: cluster stds shape does not match means; ignoring.")
        return None, None

    # Adjust time dimension
    if t_src < plan_horizon:
        pad = plan_horizon - t_src
        means = np.concatenate([means, np.repeat(means[:, -1:, :], pad, axis=1)], axis=1)
        stds = np.concatenate([stds, np.repeat(stds[:, -1:, :], pad, axis=1)], axis=1)
    elif t_src > plan_horizon:
        means = means[:, :plan_horizon, :]
        stds = stds[:, :plan_horizon, :]

    # Adjust action dimension
    if a_src < A:
        pad = A - a_src
        means = np.concatenate([means, np.zeros((k, plan_horizon, pad))], axis=2)
        stds = np.concatenate([stds, np.zeros((k, plan_horizon, pad)) + 0.1], axis=2)
    elif a_src > A:
        means = means[:, :, :A]
        stds = stds[:, :, :A]

    return means, stds


def _sample_correlated_noise(
    B: int, T: int, N: int, A: int,
    rho: float,
    device: torch.device,
    corr_chol: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AR(1) temporally-correlated noise with unit stationary variance.

    rho=0.0 → i.i.d. (original); rho≈0.8 → smooth steering-like trajectories.
    noise[t] = rho * noise[t-1] + sqrt(1-rho²) * eps[t]
    """
    # rho can be scalar or (A,) per-dim
    if isinstance(rho, (int, float)):
        rho_vec = torch.full((A,), float(rho), device=device)
    else:
        rho_vec = torch.as_tensor(rho, device=device, dtype=torch.float32)
    if (rho_vec == 0).all():
        noise = torch.randn(B, T, N, A, device=device)
        if corr_chol is not None:
            noise = noise @ corr_chol.T
        return noise
    innov_scale = torch.sqrt(1.0 - rho_vec * rho_vec)   # (A,)
    noise = torch.zeros(B, T, N, A, device=device)
    prev  = torch.zeros(B, N, A, device=device)
    for t in range(T):
        eps  = torch.randn(B, N, A, device=device)
        if corr_chol is not None:
            eps = eps @ corr_chol.T
        prev = rho_vec * prev + innov_scale * eps
        noise[:, t] = prev
    return noise


@torch.no_grad()
def cem_plan(
    model: LatentWorldModel,
    z_init: torch.Tensor,             # (1, 1, 1, gH, gW, D)
    goal: torch.Tensor,               # (1, 1, 1, gH, gW, D)
    lang_emb: Optional[torch.Tensor], # (1, D_lang) or None
    plan_horizon: int,
    n_samples: int,
    n_iters: int,
    n_elites: int,
    var_scale: float,
    momentum: float,
    cost_fn: Optional[str],
    ctxt_window: int,
    device: torch.device,
    temporal_corr = 0.0,              # AR(1) rho: scalar or per-dim list
    action_corr: Optional[np.ndarray] = None,  # cross-dim correlation (A x A)
) -> np.ndarray:
    """
    CEM planning loop using cem_unroll and cem_cost.
    Returns best action sequence (plan_horizon, 4) in normalized [-1, 1].
    """
    A = model.predictor.action_dim if hasattr(model.predictor, "action_dim") else 4
    T, N = plan_horizon, n_samples

    z_init_N = z_init.expand(N, -1, -1, -1, -1, -1).contiguous()  # (N, 1, 1, gH, gW, D)
    goal_N   = goal.expand(N, -1, -1, -1, -1, -1).contiguous()    # (N, 1, 1, gH, gW, D)
    lang_N   = lang_emb.expand(N, -1).contiguous() if lang_emb is not None else None

    action_min, action_max = -1.0, 1.0
    mean = torch.zeros(T, A, device=device)
    std  = var_scale * torch.ones(T, A, device=device)

    corr_chol = _prepare_action_corr(action_corr, A, device)

    for _ in range(n_iters):
        # Sample (T, N, A) candidates with temporal correlation; slot 0 = current mean
        noise          = _sample_correlated_noise(1, T, N, A, temporal_corr, device, corr_chol=corr_chol).squeeze(0)
        action_samples = mean.unsqueeze(1) + std.unsqueeze(1) * noise  # (T, N, A)
        action_samples = action_samples.clamp(min=action_min, max=action_max)
        action_samples[:, 0] = mean.clamp(min=action_min, max=action_max)

        # Unroll and score
        predicted, _ = cem_unroll(model, z_init_N, action_samples, lang_N, ctxt_window)
        cost      = cem_cost(predicted, goal_N, cost_fn)  # (N,)

        # Elite selection (minimize cost)
        elite_idxs    = torch.topk(-cost, n_elites, dim=0).indices  # (E,)
        elite_actions = action_samples[:, elite_idxs]               # (T, E, A)

        # Update mean/std with momentum
        new_mean = elite_actions.mean(dim=1)
        new_std  = elite_actions.std(dim=1).clamp(min=0.01)
        mean     = (new_mean * (1 - momentum) + mean * momentum).clamp(
            min=action_min, max=action_max
        )
        std      = new_std  * (1 - momentum) + std  * momentum

    return mean.cpu().numpy()  # (T, A)


@torch.no_grad()
def cem_plan_batch(
    model: LatentWorldModel,
    z_seed_B: torch.Tensor,             # (B, S, 1, gH, gW, D)
    goal_B: torch.Tensor,               # (B, 1, 1, gH, gW, D)
    lang_emb_B: Optional[torch.Tensor], # (B, D_lang) or None
    plan_horizon: int,
    n_samples: int,
    n_iters: int,
    n_elites: int,
    var_scale: float,
    momentum: float,
    cost_fn: Optional[str],
    ctxt_window: int,
    device: torch.device,
    action_prior_mean: Optional[torch.Tensor] = None,  # (A,) normalized training mean
    seed_acts_B: Optional[torch.Tensor] = None,         # (B, S-1, A) GT seed actions
    yaw_stats: Optional[dict] = None,          # normalization constants for sin/cos dims
    action_prior_std: Optional[torch.Tensor] = None,   # (A_int,) dataset std in internal space
    traj_weight: float = 0.0,                  # weight on intermediate-step cost (0 = final only)
    temporal_corr = 0.0,                        # AR(1) rho: scalar or per-dim list
    action_corr: Optional[np.ndarray] = None,   # cross-dim correlation (A x A)
    gt_action_sample_B: Optional[torch.Tensor] = None,  # (B, T, 4) ckpt-normalized GT actions
    action_cluster_mean_B: Optional[torch.Tensor] = None,  # (B, T, A) internal action space
    action_cluster_std_B: Optional[torch.Tensor] = None,   # (B, T, A) internal action space
    return_samples: bool = False,               # return final-iter action samples for plotting
    gt_action_cost_B: Optional[torch.Tensor] = None,    # (B, T, 4) ckpt-normalized GT actions
    return_iter_info: bool = False,             # return per-iter best means and min costs
    unicycle_stats: Optional[dict] = None,      # if set, sample (u, delta_yaw) and derive (dx, dy)
) -> np.ndarray:
    """
    Batched CEM planning for B examples simultaneously.

    Each example gets N independent candidates, all rolled out together as a
    single (B*N)-batch through the world model. Elite selection and mean/std
    updates are performed independently per example.

    When yaw_stats is provided the planner works in a 3D space
    [x_norm, y_norm, yaw_rad] and projects to [x_norm, y_norm, sin_norm, cos_norm]
    before each model call, enforcing the sin²+cos²=1 constraint.

    When unicycle_stats is provided the internal space shrinks to 2D [u_metric,
    delta_yaw].  dx = u·cos(yaw) and dy = u·sin(yaw) are derived before the
    model call via _unicycle_to_3d, which then proceeds through the normal
    _expand_yaw_to_sincos projection.

    Returns: (B, plan_horizon, 4) best action sequences in normalized [-1, 1].
    When gt_action_cost_B is provided, CEM optimizes mean L2 to GT actions
    instead of WM feature cost (planner lower-bound diagnostic).
    """
    B = z_seed_B.shape[0]
    S = z_seed_B.shape[1]
    A_model = model.predictor.action_dim if hasattr(model.predictor, "action_dim") else 4
    A = 2 if unicycle_stats is not None else (3 if yaw_stats is not None else A_model)
    T, N = plan_horizon, n_samples
    n_elites = min(n_elites, N)  # guard: elites can't exceed samples

    use_gt_cost = gt_action_cost_B is not None

    if not use_gt_cost:
        z_seed_BN = z_seed_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
        goal_BN   = goal_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
        lang_BN   = (lang_emb_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
                     if lang_emb_B is not None else None)
        seed_acts_BN = None
        if seed_acts_B is not None and S > 1:
            seed_acts_BN = (seed_acts_B.repeat_interleave(N, dim=0)
                            .to(torch.bfloat16)
                            .permute(1, 0, 2))  # (S-1, B*N, A)
    else:
        z_seed_BN = goal_BN = lang_BN = seed_acts_BN = None

    # Initialize mean in internal action space.
    if action_prior_mean is not None:
        if unicycle_stats is not None:
            # prior is already [u_metric, yaw_rad] — use directly
            prior_internal = action_prior_mean[:2].to(device)
        elif yaw_stats is not None:
            ys = yaw_stats
            if ys.get("sincos_masked", False):
                sin_r = action_prior_mean[2]
                cos_r = action_prior_mean[3]
            else:
                sin_r = (action_prior_mean[2] + 1.0) * (ys["p99_sin"] - ys["p01_sin"]) / 2.0 + ys["p01_sin"]
                cos_r = (action_prior_mean[3] + 1.0) * (ys["p99_cos"] - ys["p01_cos"]) / 2.0 + ys["p01_cos"]
            prior_internal = torch.cat([action_prior_mean[:2], torch.atan2(sin_r, cos_r).unsqueeze(0)])
        else:
            prior_internal = action_prior_mean
        mean = prior_internal.unsqueeze(0).unsqueeze(0).expand(B, T, A).clone()
    else:
        mean = torch.zeros(B, T, A, device=device)

    # Std: use dataset std when provided, else fall back to var_scale.
    if action_prior_std is not None:
        std = action_prior_std.unsqueeze(0).unsqueeze(0).expand(B, T, A).clone()
    else:
        std = var_scale * torch.ones(B, T, A, device=device)

    if action_cluster_mean_B is not None:
        mean = action_cluster_mean_B.to(device)
    if action_cluster_std_B is not None:
        std = action_cluster_std_B.to(device).clamp(min=0.01)

    def _clamp(t):
        if unicycle_stats is not None:
            return torch.cat([t[..., :1].clamp(0., unicycle_stats["u_max"]),
                              t[..., 1:2].clamp(-np.pi, np.pi)], dim=-1)
        if yaw_stats is not None:
            return torch.cat([t[..., :2].clamp(-1., 1.),
                              t[..., 2:3].clamp(-np.pi, np.pi)], dim=-1)
        return t.clamp(-1., 1.)

    corr_chol = _prepare_action_corr(action_corr, A, device)
    gt_internal = None
    gt_sample_idx = None
    if gt_action_sample_B is not None and N > 1:
        if unicycle_stats is not None:
            gt_3d = (_sincos_to_yaw(gt_action_sample_B, yaw_stats) if yaw_stats is not None
                     else gt_action_sample_B[..., :3])
            gt_internal = _3d_to_unicycle(gt_3d.to(device), unicycle_stats)
        elif yaw_stats is not None:
            gt_internal = _sincos_to_yaw(gt_action_sample_B, yaw_stats)
        else:
            gt_internal = gt_action_sample_B[..., :A]
        gt_internal = gt_internal.to(device)
        gt_sample_idx = N - 1
    last_samples = None
    first_samples_4d = None  # (B, T, N, 4) first-iter samples projected to 4D
    last_cost = None
    _iter_means = [] if return_iter_info else None
    _iter_costs = [] if return_iter_info else None
    _all_iter_samps = [] if return_iter_info else None  # all-iter samples projected to 4D
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for iter_j in range(n_iters):
            noise = _sample_correlated_noise(B, T, N, A, temporal_corr, device, corr_chol=corr_chol)
            action_samples = mean.unsqueeze(2) + std.unsqueeze(2) * noise  # (B, T, N, A)
            action_samples = _clamp(action_samples.reshape(B * T * N, A)).reshape(B, T, N, A)
            action_samples[:, :, 0] = _clamp(mean)
            if gt_internal is not None and gt_sample_idx is not None:
                action_samples[:, :, gt_sample_idx] = _clamp(gt_internal)
            if return_samples:
                last_samples = action_samples
            if return_iter_info:
                _fs = action_samples
                if unicycle_stats is not None:
                    _fs = _unicycle_to_3d(_fs, unicycle_stats)
                if yaw_stats is not None:
                    _fs = _expand_yaw_to_sincos(_fs, yaw_stats)
                _all_iter_samps.append(_fs.detach().float().cpu())
                if iter_j == 0:
                    first_samples_4d = _all_iter_samps[0]

            if use_gt_cost:
                # convert to 3D before L2 comparison (unicycle samples are 2D)
                _samps = (_unicycle_to_3d(action_samples, unicycle_stats) if unicycle_stats is not None
                          else action_samples)
                cost = _compute_action_l2_cost(_samps, gt_action_cost_B, yaw_stats)
            else:
                # Reshape to (B*N, T, A) then project to 4D for the model.
                acts_BTN = action_samples.permute(0, 2, 1, 3).reshape(B * N, T, A)
                if unicycle_stats is not None:
                    acts_BTN = _unicycle_to_3d(acts_BTN, unicycle_stats)  # (B*N, T, 3)
                if yaw_stats is not None:
                    acts_BTN = _expand_yaw_to_sincos(acts_BTN, yaw_stats)  # (B*N, T, 4)
                actions_T_BN = acts_BTN.permute(1, 0, 2).to(torch.bfloat16)

                predicted, i_cost = cem_unroll(model, z_seed_BN, actions_T_BN, lang_BN, ctxt_window,
                                              seed_acts_SNA=seed_acts_BN,
                                              goal_N=goal_BN, traj_cost_fn=cost_fn,
                                              traj_weight=traj_weight)
                cost = cem_cost(predicted, goal_BN, cost_fn).float().reshape(B, N)
                if i_cost is not None:
                    cost = cost + i_cost.float().reshape(B, N)
            last_cost = cost
            # last iter:
            # if iter_j == n_iters - 1:
            #     n_elites = 1  # for cleaner output; we only return the final mean, not the elites themselves
            elite_idxs = torch.topk(-cost, n_elites, dim=1).indices            # (B, E)
            idx_exp    = elite_idxs[:, None, :, None].expand(B, T, n_elites, A)
            elite_acts = action_samples.gather(2, idx_exp)                      # (B, T, E, A)

            new_mean = elite_acts.mean(dim=2)
            new_std  = elite_acts.std(dim=2).clamp(min=0.01)
            mean = _clamp(new_mean * (1 - momentum) + mean * momentum)
            std  = new_std  * (1 - momentum) + std  * momentum

            if _iter_costs is not None:
                _iter_costs.append(cost.min(dim=1).values.float().cpu())
                _iter_means.append(mean.detach().float().cpu())
            

    # Project back to 4D normalized for the downstream pipeline.
    if unicycle_stats is not None:
        mean_3d = _unicycle_to_3d(mean, unicycle_stats)
        best = (_expand_yaw_to_sincos(mean_3d, yaw_stats).cpu().numpy() if yaw_stats is not None
                else mean_3d.cpu().numpy())
    else:
        best = (_expand_yaw_to_sincos(mean, yaw_stats).cpu().numpy() if yaw_stats is not None
                else mean.cpu().numpy())
    best_cost = last_cost.min(dim=1).values.float().cpu().numpy() if last_cost is not None else None
    iter_info = None
    if return_iter_info and _iter_means:
        iter_info = (
            torch.stack(_iter_means, dim=0).numpy(),                              # (n_iters, B, T, A_internal)
            torch.stack(_iter_costs, dim=0).numpy(),                              # (n_iters, B)
            first_samples_4d.numpy() if first_samples_4d is not None else None,  # (B, T, N, 4)
            [s.numpy() for s in _all_iter_samps] if _all_iter_samps else None,   # list[n_iters] of (B,T,N,4)
        )
    if return_samples and last_samples is not None:
        if unicycle_stats is not None:
            samp_3d = _unicycle_to_3d(last_samples, unicycle_stats)
            samp = (_expand_yaw_to_sincos(samp_3d, yaw_stats).cpu().numpy() if yaw_stats is not None
                    else samp_3d.cpu().numpy())
        elif yaw_stats is not None:
            samp = _expand_yaw_to_sincos(last_samples, yaw_stats).cpu().numpy()
        else:
            samp = last_samples.cpu().numpy()
        cost_out = last_cost.cpu().numpy() if last_cost is not None else None
        return best, samp, cost_out, iter_info
    return best, None, best_cost, iter_info  # (B, T, A), None, (B,), (iter_info or None)


@torch.no_grad()
def mppi_plan_batch(
    model: LatentWorldModel,
    z_seed_B: torch.Tensor,             # (B, S, 1, gH, gW, D)
    goal_B: torch.Tensor,               # (B, 1, 1, gH, gW, D)
    lang_emb_B: Optional[torch.Tensor], # (B, D_lang) or None
    plan_horizon: int,
    n_samples: int,
    n_iters: int,
    temperature: float,
    var_scale: float,
    momentum: float,
    std_min: float,
    std_max: float,
    cost_fn: Optional[str],
    ctxt_window: int,
    device: torch.device,
    action_prior_mean: Optional[torch.Tensor] = None,  # (A,) normalized training mean
    seed_acts_B: Optional[torch.Tensor] = None,         # (B, S-1, A) GT seed actions
    yaw_stats: Optional[dict] = None,          # normalization constants for sin/cos dims
    action_prior_std: Optional[torch.Tensor] = None,   # (A_int,) dataset std in internal space
    traj_weight: float = 0.0,                  # weight on intermediate-step cost (0 = final only)
    temporal_corr = 0.0,                        # AR(1) rho: scalar or per-dim list
    action_corr: Optional[np.ndarray] = None,   # cross-dim correlation (A x A)
    gt_action_sample_B: Optional[torch.Tensor] = None,  # (B, T, 4) ckpt-normalized GT actions
    action_cluster_mean_B: Optional[torch.Tensor] = None,  # (B, T, A) internal action space
    action_cluster_std_B: Optional[torch.Tensor] = None,   # (B, T, A) internal action space
    return_samples: bool = False,               # return final-iter action samples for plotting
    gt_action_cost_B: Optional[torch.Tensor] = None,    # (B, T, 4) ckpt-normalized GT actions
    return_iter_info: bool = False,             # return per-iter best means and min costs
    unicycle_stats: Optional[dict] = None,      # if set, sample (u, delta_yaw) and derive (dx, dy)
) -> np.ndarray:
    """
    Batched MPPI planning for B examples simultaneously.

    Unlike CEM's hard elite selection, MPPI soft-weights ALL N candidates by
    exp(temperature * (min_cost - cost_i)) so every sample contributes to the
    update — avoiding variance collapse from small elite sets.

    When yaw_stats is provided the planner works in a 3D space
    [x_norm, y_norm, yaw_rad] and projects to [x_norm, y_norm, sin_norm, cos_norm]
    before each model call, enforcing the sin²+cos²=1 constraint.

    When unicycle_stats is provided the internal space is 2D [u_metric, delta_yaw].

    Returns: (B, plan_horizon, 4) best action sequences in normalized [-1, 1].
    When gt_action_cost_B is provided, MPPI optimizes mean L2 to GT actions
    instead of WM feature cost (planner lower-bound diagnostic).
    """
    B = z_seed_B.shape[0]
    S = z_seed_B.shape[1]
    A_model = model.predictor.action_dim if hasattr(model.predictor, "action_dim") else 4
    A = 2 if unicycle_stats is not None else (3 if yaw_stats is not None else A_model)
    T, N = plan_horizon, n_samples

    use_gt_cost = gt_action_cost_B is not None

    if not use_gt_cost:
        z_seed_BN = z_seed_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
        goal_BN   = goal_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
        lang_BN   = (lang_emb_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
                     if lang_emb_B is not None else None)
        seed_acts_BN = None
        if seed_acts_B is not None and S > 1:
            seed_acts_BN = (seed_acts_B.repeat_interleave(N, dim=0)
                            .to(torch.bfloat16)
                            .permute(1, 0, 2))  # (S-1, B*N, A)
    else:
        z_seed_BN = goal_BN = lang_BN = seed_acts_BN = None

    # Initialize mean in internal action space.
    if action_prior_mean is not None:
        if unicycle_stats is not None:
            prior_internal = action_prior_mean[:2].to(device)
        elif yaw_stats is not None:
            ys = yaw_stats
            if ys.get("sincos_masked", False):
                sin_r = action_prior_mean[2]
                cos_r = action_prior_mean[3]
            else:
                sin_r = (action_prior_mean[2] + 1.0) * (ys["p99_sin"] - ys["p01_sin"]) / 2.0 + ys["p01_sin"]
                cos_r = (action_prior_mean[3] + 1.0) * (ys["p99_cos"] - ys["p01_cos"]) / 2.0 + ys["p01_cos"]
            prior_internal = torch.cat([action_prior_mean[:2], torch.atan2(sin_r, cos_r).unsqueeze(0)])
        else:
            prior_internal = action_prior_mean
        mean = prior_internal.unsqueeze(0).unsqueeze(0).expand(B, T, A).clone()
    else:
        mean = torch.zeros(B, T, A, device=device)

    # Std: use dataset std when provided, else fall back to var_scale.
    if action_prior_std is not None:
        std = action_prior_std.unsqueeze(0).unsqueeze(0).expand(B, T, A).clone()
    else:
        std = var_scale * torch.ones(B, T, A, device=device)

    if action_cluster_mean_B is not None:
        mean = action_cluster_mean_B.to(device)
    if action_cluster_std_B is not None:
        std = action_cluster_std_B.to(device).clamp(min=0.01)

    def _clamp(t):
        if unicycle_stats is not None:
            return torch.cat([t[..., :1].clamp(0., unicycle_stats["u_max"]),
                              t[..., 1:2].clamp(-np.pi, np.pi)], dim=-1)
        if yaw_stats is not None:
            return torch.cat([t[..., :2].clamp(-1., 1.),
                              t[..., 2:3].clamp(-np.pi, np.pi)], dim=-1)
        return t.clamp(-1., 1.)

    corr_chol = _prepare_action_corr(action_corr, A, device)
    gt_internal = None
    gt_sample_idx = None
    if gt_action_sample_B is not None and N > 1:
        if unicycle_stats is not None:
            gt_3d = (_sincos_to_yaw(gt_action_sample_B, yaw_stats) if yaw_stats is not None
                     else gt_action_sample_B[..., :3])
            gt_internal = _3d_to_unicycle(gt_3d.to(device), unicycle_stats)
        elif yaw_stats is not None:
            gt_internal = _sincos_to_yaw(gt_action_sample_B, yaw_stats)
        else:
            gt_internal = gt_action_sample_B[..., :A]
        gt_internal = gt_internal.to(device)
        gt_sample_idx = N - 1
    last_samples = None
    first_samples_4d = None  # (B, T, N, 4) first-iter samples projected to 4D
    last_cost = None
    _iter_means = [] if return_iter_info else None
    _iter_costs = [] if return_iter_info else None
    _all_iter_samps = [] if return_iter_info else None  # all-iter samples projected to 4D
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for iter_j in range(n_iters):
            noise = _sample_correlated_noise(B, T, N, A, temporal_corr, device, corr_chol=corr_chol)
            action_samples = mean.unsqueeze(2) + std.unsqueeze(2) * noise  # (B, T, N, A)
            action_samples = _clamp(action_samples.reshape(B * T * N, A)).reshape(B, T, N, A)
            action_samples[:, :, 0] = _clamp(mean)  # slot 0 = deterministic anchor
            if gt_internal is not None and gt_sample_idx is not None:
                action_samples[:, :, gt_sample_idx] = _clamp(gt_internal)
            if return_iter_info:
                _fs = action_samples
                if unicycle_stats is not None:
                    _fs = _unicycle_to_3d(_fs, unicycle_stats)
                if yaw_stats is not None:
                    _fs = _expand_yaw_to_sincos(_fs, yaw_stats)
                _all_iter_samps.append(_fs.detach().float().cpu())
                if iter_j == 0:
                    first_samples_4d = _all_iter_samps[0]
            if return_samples:
                last_samples = action_samples

            if use_gt_cost:
                _samps = (_unicycle_to_3d(action_samples, unicycle_stats) if unicycle_stats is not None
                          else action_samples)
                cost = _compute_action_l2_cost(_samps, gt_action_cost_B, yaw_stats)
            else:
                # Reshape to (B*N, T, A) then project to 4D for the model.
                acts_BTN = action_samples.permute(0, 2, 1, 3).reshape(B * N, T, A)
                if unicycle_stats is not None:
                    acts_BTN = _unicycle_to_3d(acts_BTN, unicycle_stats)  # (B*N, T, 3)
                if yaw_stats is not None:
                    acts_BTN = _expand_yaw_to_sincos(acts_BTN, yaw_stats)  # (B*N, T, 4)
                actions_T_BN = acts_BTN.permute(1, 0, 2).to(torch.bfloat16)

                predicted, i_cost = cem_unroll(model, z_seed_BN, actions_T_BN, lang_BN, ctxt_window,
                                              seed_acts_SNA=seed_acts_BN,
                                              goal_N=goal_BN, traj_cost_fn=cost_fn,
                                              traj_weight=traj_weight)
                cost = cem_cost(predicted, goal_BN, cost_fn).float().reshape(B, N)  # (B, N)
                if i_cost is not None:
                    cost = cost + i_cost.float().reshape(B, N)
            last_cost = cost

            # Soft MPPI weights: all N samples contribute proportionally.
            # Dividing by cost_std makes temperature scale-invariant across
            # different cost magnitudes (temperature=1 = moderate, 3 = sharp).
            min_cost  = cost.min(dim=1, keepdim=True).values            # (B, 1) stability shift
            cost_std  = cost.std(dim=1, keepdim=True).clamp(min=1e-6)   # (B, 1)
            score = torch.exp(temperature * (min_cost - cost) / cost_std)  # (B, N)
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)    # (B, N) normalized

            s = score[:, None, :, None]                                # (B, 1, N, 1)
            new_mean = (s * action_samples).sum(dim=2)                 # (B, T, A)
            delta    = action_samples - new_mean.unsqueeze(2)          # (B, T, N, A)
            new_std  = torch.sqrt((s * delta.pow(2)).sum(dim=2)).clamp(min=std_min, max=std_max)

            mean = _clamp(new_mean * (1 - momentum) + mean * momentum)
            std  = (new_std  * (1 - momentum) + std  * momentum).clamp(min=std_min, max=std_max)
            if _iter_costs is not None:
                _iter_costs.append(cost.min(dim=1).values.float().cpu())
                _iter_means.append(mean.detach().float().cpu())

    # Project back to 4D normalized for the downstream pipeline.
    if unicycle_stats is not None:
        mean_3d = _unicycle_to_3d(mean, unicycle_stats)
        best = (_expand_yaw_to_sincos(mean_3d, yaw_stats).cpu().numpy() if yaw_stats is not None
                else mean_3d.cpu().numpy())
    else:
        best = (_expand_yaw_to_sincos(mean, yaw_stats).cpu().numpy() if yaw_stats is not None
                else mean.cpu().numpy())
    best_cost = last_cost.min(dim=1).values.float().cpu().numpy() if last_cost is not None else None
    iter_info = None
    if return_iter_info and _iter_means:
        iter_info = (
            torch.stack(_iter_means, dim=0).numpy(),                              # (n_iters, B, T, A_internal)
            torch.stack(_iter_costs, dim=0).numpy(),                              # (n_iters, B)
            first_samples_4d.numpy() if first_samples_4d is not None else None,  # (B, T, N, 4)
            [s.numpy() for s in _all_iter_samps] if _all_iter_samps else None,   # list[n_iters] of (B,T,N,4)
        )
    if return_samples and last_samples is not None:
        if unicycle_stats is not None:
            samp_3d = _unicycle_to_3d(last_samples, unicycle_stats)
            samp = (_expand_yaw_to_sincos(samp_3d, yaw_stats).cpu().numpy() if yaw_stats is not None
                    else samp_3d.cpu().numpy())
        elif yaw_stats is not None:
            samp = _expand_yaw_to_sincos(last_samples, yaw_stats).cpu().numpy()
        else:
            samp = last_samples.cpu().numpy()
        cost_out = last_cost.cpu().numpy() if last_cost is not None else None
        return best, samp, cost_out, iter_info
    return best, None, best_cost, iter_info  # (B, T, A), None, (B,), (iter_info or None)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_raw_language(raw) -> List[str]:
    if isinstance(raw, list):
        return [r.decode("utf-8") if isinstance(r, (bytes, bytearray)) else str(r) for r in raw]
    if isinstance(raw, np.ndarray):
        return [r.decode("utf-8") if isinstance(r, (bytes, bytearray)) else str(r)
                for r in raw.tolist()]
    return [str(raw)]


def _to_numpy_tree(x):
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def eval_dataset(
    model: LatentWorldModel,
    loader: DataLoader,
    ds_stats: dict,
    device: torch.device,
    plan_horizon: int,
    cem_samples: int,
    cem_iters: int,
    cem_elites: int,
    cem_var_scale: float,
    cem_momentum: float,
    cost_fn: Optional[str],
    ctxt_window: int,
    num_examples: Optional[int] = None,
    planner: str = "cem",
    mppi_temperature: float = 1.0,
    mppi_std_min: float = 0.01,
    mppi_std_max: float = 2.0,
    orientation_metric: str = "vector",
    output_plot: Optional[str] = None,
    debug: bool = False,
    checkpoint_json: Optional[str] = None,
    save_every: int = 100,
    seed_frames: int = 1,
    gt_baseline: bool = False,
    traj_weight: float = 0.0,
    temporal_corr: float = 0.0,
    action_corr: Optional[np.ndarray] = None,
    include_gt_sample: bool = False,
    action_cluster_means: Optional[np.ndarray] = None,
    action_cluster_stds: Optional[np.ndarray] = None,
    action_cluster_counts: Optional[np.ndarray] = None,
    cluster_index: Optional[int] = None,
    gt_ds_stats: Optional[dict] = None,    # fresh dataset stats for GT unnorm; falls back to ds_stats
    ckpt_ds_stats: Optional[dict] = None,  # checkpoint stats for seed-act renorm; falls back to ds_stats
    plot_samples: bool = False,            # plot final-iter samples in action plot
    gt_action_cost: bool = False,          # also evaluate planner with L2 GT action cost
    unicycle: bool = False,                # sample (u, delta_yaw) and derive (dx, dy) via unicycle model
) -> Tuple[Dict, List, Optional[Dict]]:
    """Run evaluation on one dataset split until exhausted.

    Returns (aggregate_metrics, per_example_list, aggregate_gt_l2_or_None).
    """
    import time

    model.eval()
    all_metrics: Dict[str, List[float]] = {"AOE": [], "MAOE": [], "ADE": [], "MADE": []}
    all_metrics_gt_l2: Optional[Dict[str, List[float]]] = None
    if gt_action_cost:
        all_metrics_gt_l2 = {"AOE": [], "MAOE": [], "ADE": [], "MADE": []}
    records: List[dict] = []
    all_cem_costs: List[float] = []
    n_done = 0
    _last_save_n = 0

    t_data   = 0.0  # data loading + H2D transfer
    t_encode = 0.0  # encode_obs
    t_cem    = 0.0  # cem_plan_batch
    t_post   = 0.0  # unnorm + trajectory + metrics

    try:
        _total = num_examples // loader.batch_size if num_examples is not None else len(loader)
    except TypeError:
        _total = None
    pbar = tqdm(PrefetchLoader(loader), desc="  examples", ncols=100,
                total=_total)

    fallback_ds = list(ds_stats.keys())[0]

    # Compute the normalized training mean to initialize the planner near the
    # actual action distribution (p01/p99 are asymmetric so mean=0 ≠ median).
    _astats = ds_stats[fallback_ds]["action"]
    _p01 = np.array(_astats["p01"])
    _p99 = np.array(_astats["p99"])
    _raw_mean = np.array(_astats["mean"])

    # Detect whether sin/cos dims were left un-normalized during training.
    _mask_arr = _astats.get("mask", None)
    if _mask_arr is not None:
        _mask_arr = np.array(_mask_arr).flatten().astype(bool)
        sincos_masked = len(_mask_arr) >= 4 and not _mask_arr[2] and not _mask_arr[3]
    else:
        sincos_masked = False
    print(f"Detected sin/cos masked={sincos_masked} based on action mask: {_mask_arr}")
    if sincos_masked:
        # x,y: BOUNDS-normalize; sin/cos: pass through raw (already in [-1, 1])
        _prior_norm = np.zeros(4, dtype=np.float32)
        _prior_norm[:2] = np.clip(
            2.0 * (_raw_mean.flatten()[:2] - _p01.flatten()[:2])
            / (_p99.flatten()[:2] - _p01.flatten()[:2] + 1e-8) - 1.0,
            -1.0, 1.0,
        )
        _prior_norm[2:4] = _raw_mean.flatten()[2:4]
    else:
        _prior_norm = np.clip(
            2.0 * (_raw_mean - _p01) / (_p99 - _p01 + 1e-8) - 1.0, -1.0, 1.0
        ).astype(np.float32)
    action_prior_mean = torch.tensor(_prior_norm.reshape(-1), device=device)  # always (A,)
    print(f"  action_prior_mean: {action_prior_mean.tolist()}  (NaN={action_prior_mean.isnan().any().item()})")

    # Dataset std in 3D internal space [x_norm, y_norm, yaw_rad].
    # Some datasets store action stats as scalars / ragged arrays; flatten everything
    # first so indexing by dim is safe.
    _p01_flat      = np.array(_astats["p01"]).flatten()
    _p99_flat      = np.array(_astats["p99"]).flatten()
    _raw_mean_flat = np.array(_astats["mean"]).flatten()
    _std_raw       = _astats.get("std", None)
    _raw_std_flat  = (np.array(_std_raw).flatten() if _std_raw is not None
                      else (_p99_flat - _p01_flat) / 5.15)
    _has_per_dim   = (len(_raw_mean_flat) >= 4 and len(_raw_std_flat) >= 4
                      and len(_p01_flat) >= 4)

    # x / y normalised std
    if _has_per_dim:
        _xy_std_norm = np.clip(
            2.0 * _raw_std_flat[:2] / (_p99_flat[:2] - _p01_flat[:2] + 1e-8), 0.05, 2.0
        )
    else:
        _xy_std_norm = np.array([0.4, 0.4])  # fallback ~40 % of normalised range

    # Yaw std via delta method:  std_yaw² ≈ (cos_mean/R²)²·var_sin + (sin_mean/R²)²·var_cos
    if _has_per_dim:
        _mean_sin_r  = float(_raw_mean_flat[2])
        _mean_cos_r  = float(_raw_mean_flat[3])
        _R_sq        = _mean_sin_r**2 + _mean_cos_r**2 + 1e-8
        _yaw_std_rad = float(np.sqrt(
            (_mean_cos_r**2 * _raw_std_flat[2]**2
             + _mean_sin_r**2 * _raw_std_flat[3]**2) / _R_sq**2
        ))
    elif len(_p01_flat) >= 3 and len(_p99_flat) >= 3:
        _yaw_std_rad = float((_p99_flat[2] - _p01_flat[2]) / 5.15)  # sin percentile proxy
    else:
        _yaw_std_rad = 0.3  # conservative default (~17°)
    _yaw_std_rad = float(np.clip(_yaw_std_rad, 0.05, np.pi))

    _prior_std_3d = np.array([_xy_std_norm[0], _xy_std_norm[1], _yaw_std_rad], dtype=np.float32)
    action_prior_std = torch.tensor(_prior_std_3d * cem_var_scale, device=device)  # (3,)
    print(f"  action_prior_std:  {action_prior_std.tolist()}  (scaled by cem_var_scale={cem_var_scale})")
    print(f"  raw stats — p01: {_p01_flat.tolist()}, p99: {_p99_flat.tolist()}, mean: {_raw_mean_flat.tolist()}, std: {_raw_std_flat.tolist()}")
    yaw_stats = (_yaw_sincos_stats(_p01_flat, _p99_flat, sincos_masked=sincos_masked)
                 if len(_p01_flat) >= 4 else None)
    if yaw_stats is None:
        action_prior_std = None   # 3D std only valid when yaw_stats is available

    # Build unicycle sampling stats: internal space is [u_metric, delta_yaw] (2D).
    unicycle_stats_dict = None
    if unicycle:
        if yaw_stats is not None:
            _u_mean = float(np.sqrt(_raw_mean_flat[0]**2 + _raw_mean_flat[1]**2))
            _yaw_mean = float(np.arctan2(float(_raw_mean_flat[2]), float(_raw_mean_flat[3])))
            _u_std_base = float(np.sqrt(_raw_std_flat[0]**2 + _raw_std_flat[1]**2))
            _u_max = float(np.sqrt(max(_p99_flat[0], 0.0)**2 + _p99_flat[1]**2))
            unicycle_stats_dict = {
                "p01_x": float(_p01_flat[0]),
                "p99_x": float(_p99_flat[0]),
                "p01_y": float(_p01_flat[1]),
                "p99_y": float(_p99_flat[1]),
                "u_max": _u_max,
            }
            action_prior_mean = torch.tensor([_u_mean, _yaw_mean], dtype=torch.float32, device=device)
            action_prior_std  = torch.tensor([_u_std_base, _yaw_std_rad], dtype=torch.float32, device=device) * cem_var_scale
            print(f"  unicycle mode ON: u_max={_u_max:.3f}  u_mean={_u_mean:.3f}  u_std={_u_std_base * cem_var_scale:.4f}"
                  f"  yaw_mean={_yaw_mean:.4f} rad  yaw_std={_yaw_std_rad * cem_var_scale:.4f} rad")
        else:
            print("  Warning: --unicycle requires 4D action stats (sin/cos yaw encoding); unicycle disabled.")

    cluster_means = None
    cluster_stds = None
    if action_cluster_means is not None and action_cluster_stds is not None:
        cluster_means, cluster_stds = _prepare_cluster_stats(
            action_cluster_means, action_cluster_stds, plan_horizon, A=3 if yaw_stats is not None else 4
        )
        if cluster_means is not None:
            print(f"Using action clusters: K={cluster_means.shape[0]}  T={cluster_means.shape[1]}")

    # Data for the final-example plot (updated each batch, used after the loop)
    _last_img_first = _last_img_goal = _last_xy_gt = _last_xy_pred = None
    _last_xy_state = None
    _last_gt_headings = _last_pred_headings = _last_state_headings = None
    _last_gt_actions = _last_pred_actions = _last_lang = None
    _last_sample_actions = None
    _last_sample_costs = None
    _last_first_iter_sample_xys = None
    _last_gt_sample_rank = None
    _last_gt_sample_cost = None
    _last_gt_sample_n = None
    _last_feat_first = None
    _last_feat_goal_gt = None
    _last_feat_goal_pred = None
    _last_iter_xy_trajs: Optional[list] = None  # per-CEM-iter best traj for last example
    _last_iter_costs: Optional[np.ndarray] = None  # per-CEM-iter min cost for last example

    _t0 = time.perf_counter()
    for batch in pbar:
        if num_examples is not None and n_done >= num_examples:
            break

        # ── Data / H2D ────────────────────────────────────────────────────
        t_data += time.perf_counter() - _t0

        _t1 = time.perf_counter()
        images = batch["observation"]["image_primary"].to(device)  # (B, T, C, H, W)
        actions = batch["action"]
        if actions.dim() == 4:   # (B, T, action_horizon, A) → (B, T, A)
            actions = actions[:, :, 0, :]
        B = images.shape[0]

        raw_language = _normalize_raw_language(batch.get("raw_language", [""] * B))
        ds_names_raw = batch.get("dataset_name", [None] * B)
        ds_names = []
        for name in ds_names_raw:
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8")
            elif name is not None:
                name = str(name)
            ds_names.append(name if name in ds_stats else fallback_ds)

        # Skip windows that don't have a goal frame at seed_frames-1+plan_horizon.
        # We check BEFORE encoding so we don't waste an encoder forward.
        if images.shape[1] < seed_frames + plan_horizon:
            n_done += B
            _t0 = time.perf_counter()
            continue

        # Skip batches whose action tensor contains NaN (corrupted dataset entries).
        if actions.isnan().any():
            print(f"  [warn] skipping batch {n_done}..{n_done+B}: NaN in actions")
            n_done += B
            _t0 = time.perf_counter()
            continue

        # WM needs seed_frames context frames + the goal frame.
        goal_idx  = seed_frames - 1 + plan_horizon
        seed_idxs = list(range(seed_frames))
        images_zg = images[:, seed_idxs + [goal_idx]]                 # (B, S+1, C, H, W)
        zg_feats = model.encode_obs(images_zg)                         # (B, S+1, 1, gH, gW, D)
        lang_emb = None                                                # paper models are image-goal only
        if debug:
            torch.cuda.synchronize(device)
        t_encode += time.perf_counter() - _t1

        # ── CEM ───────────────────────────────────────────────────────────
        z_seed    = zg_feats[:, :seed_frames]                          # (B, S, 1, gH, gW, D)
        goal      = zg_feats[:, seed_frames:seed_frames + 1]          # (B, 1, 1, gH, gW, D)
        seed_acts = (actions[:, :seed_frames - 1].to(device)          # (B, S-1, A)
                     if seed_frames > 1 else None)

        # Loader normalized actions with gt_stats; WM expects ckpt-normalized actions.
        if seed_acts is not None and gt_ds_stats is not None:
            _gt_as   = gt_ds_stats[list(gt_ds_stats.keys())[0]]["action"]
            _ckpt_src = ckpt_ds_stats if ckpt_ds_stats is not None else ds_stats
            _ckpt_as = _ckpt_src[list(_ckpt_src.keys())[0]]["action"]
            seed_acts = _renorm_gt_to_ckpt(seed_acts, _gt_as, _ckpt_as, device)

        act_start = seed_frames - 1
        gt_acts_cost_B = None
        gt_acts_sample_B = None
        if include_gt_sample or (gt_action_cost and not gt_baseline):
            _gt_acts_raw = actions[:, act_start:act_start + plan_horizon].to(device)
            if gt_ds_stats is not None:
                _gt_as   = gt_ds_stats[list(gt_ds_stats.keys())[0]]["action"]
                _ckpt_src = ckpt_ds_stats if ckpt_ds_stats is not None else ds_stats
                _ckpt_as = _ckpt_src[list(_ckpt_src.keys())[0]]["action"]
                _gt_acts_raw = _renorm_gt_to_ckpt(_gt_acts_raw, _gt_as, _ckpt_as, device)
            if include_gt_sample:
                gt_acts_sample_B = _gt_acts_raw
            if gt_action_cost and not gt_baseline:
                gt_acts_cost_B = _gt_acts_raw

        if goal.shape[1] == 0:                  # defensive (guard above should cover this)
            n_done += B
            _t0 = time.perf_counter()
            continue

        _t2 = time.perf_counter()
        sample_acts_B = None
        sample_costs_B = None
        pred_acts_oracle_B = None
        cluster_mean_B = None
        cluster_std_B = None
        if cluster_means is not None and cluster_stds is not None:
            k = cluster_means.shape[0]
            if cluster_index is not None:
                idx = int(np.clip(cluster_index, 0, k - 1))
                idxs = np.full(B, idx, dtype=np.int32)
            elif action_cluster_counts is not None and np.sum(action_cluster_counts) > 0:
                probs = action_cluster_counts / np.sum(action_cluster_counts)
                idxs = np.random.choice(k, size=B, p=probs)
            else:
                idxs = np.random.randint(0, k, size=B)
            cluster_mean_B = torch.tensor(cluster_means[idxs], device=device, dtype=torch.float32)
            cluster_std_B = torch.tensor(cluster_stds[idxs], device=device, dtype=torch.float32)
        pred_final = None  # set in gt_baseline path; used for PCA feature visualization
        _iter_info = None  # set by planner; None in gt_baseline path
        if gt_baseline:
            # Use ground-truth actions directly — measures WM ceiling, not planner quality.
            act_start_gt = seed_frames - 1
            pred_acts_B = actions[:B, act_start_gt:act_start_gt + plan_horizon].cpu().numpy()

            # WM oracle: roll out the WM with GT actions and measure feature similarity
            # to the goal. High cosine sim means the WM can distinguish good from bad
            # actions; low sim means WM rollout error is the fundamental bottleneck.
            if not hasattr(eval_dataset, "_wm_cosim_accum"):
                eval_dataset._wm_cosim_accum = []
            gt_acts_dev = actions[:B, act_start_gt:act_start_gt + plan_horizon].to(device)
            # Convert to ckpt-normalized for WM oracle (loader used gt_stats)
            if gt_ds_stats is not None:
                _gt_as    = gt_ds_stats[list(gt_ds_stats.keys())[0]]["action"]
                _ckpt_src = ckpt_ds_stats if ckpt_ds_stats is not None else ds_stats
                _ckpt_as  = _ckpt_src[list(_ckpt_src.keys())[0]]["action"]
                gt_acts_dev = _renorm_gt_to_ckpt(gt_acts_dev, _gt_as, _ckpt_as, device)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seed_acts_gt = (actions[:B, :seed_frames - 1].to(device)
                                if seed_frames > 1 else None)
                if seed_acts_gt is not None and gt_ds_stats is not None:
                    seed_acts_gt = _renorm_gt_to_ckpt(seed_acts_gt, _gt_as, _ckpt_as, device)
                seed_acts_gt_perm = (seed_acts_gt.permute(1, 0, 2)
                                     if seed_acts_gt is not None else None)
                pred_final, _ = cem_unroll(
                    model,
                    z_seed.to(torch.bfloat16),
                    gt_acts_dev.permute(1, 0, 2).to(torch.bfloat16),  # (T, B, A)
                    lang_emb.to(torch.bfloat16) if lang_emb is not None else None,
                    ctxt_window,
                    seed_acts_SNA=seed_acts_gt_perm,
                )  # (B, 1, 1, gH, gW, D)
                goal_flat = goal.to(torch.bfloat16).flatten(1).float()   # (B, *)
                pred_flat = pred_final.flatten(1).float()                # (B, *)
                cosim = F.cosine_similarity(pred_flat, goal_flat, dim=1)  # (B,)
            eval_dataset._wm_cosim_accum.extend(cosim.cpu().tolist())
        elif planner == "mppi":
            mppi_out = mppi_plan_batch(
                model=model,
                z_seed_B=z_seed,
                goal_B=goal,
                lang_emb_B=lang_emb,
                plan_horizon=plan_horizon,
                n_samples=cem_samples,
                n_iters=cem_iters,
                temperature=mppi_temperature,
                var_scale=cem_var_scale,
                momentum=cem_momentum,
                std_min=mppi_std_min,
                std_max=mppi_std_max,
                cost_fn=cost_fn,
                ctxt_window=ctxt_window,
                device=device,
                action_prior_mean=action_prior_mean,
                seed_acts_B=seed_acts,
                yaw_stats=yaw_stats,
                action_prior_std=action_prior_std,
                traj_weight=traj_weight,
                temporal_corr=temporal_corr,
                action_corr=action_corr,
                gt_action_sample_B=gt_acts_sample_B,
                action_cluster_mean_B=cluster_mean_B,
                action_cluster_std_B=cluster_std_B,
                return_samples=plot_samples and output_plot is not None,
                return_iter_info=output_plot is not None,
                unicycle_stats=unicycle_stats_dict,
            )
            pred_acts_B, sample_acts_B, sample_costs_B, _iter_info = mppi_out
            if gt_action_cost and gt_acts_cost_B is not None:
                mppi_oracle = mppi_plan_batch(
                    model=model,
                    z_seed_B=z_seed,
                    goal_B=goal,
                    lang_emb_B=lang_emb,
                    plan_horizon=plan_horizon,
                    n_samples=cem_samples,
                    n_iters=cem_iters,
                    temperature=mppi_temperature,
                    var_scale=cem_var_scale,
                    momentum=cem_momentum,
                    std_min=mppi_std_min,
                    std_max=mppi_std_max,
                    cost_fn=cost_fn,
                    ctxt_window=ctxt_window,
                    device=device,
                    action_prior_mean=action_prior_mean,
                    seed_acts_B=seed_acts,
                    yaw_stats=yaw_stats,
                    action_prior_std=action_prior_std,
                    traj_weight=traj_weight,
                    temporal_corr=temporal_corr,
                    action_corr=action_corr,
                    gt_action_sample_B=None,
                    action_cluster_mean_B=None,
                    action_cluster_std_B=None,
                    return_samples=False,
                    gt_action_cost_B=gt_acts_cost_B,
                    unicycle_stats=unicycle_stats_dict,
                )
                pred_acts_oracle_B = mppi_oracle[0]
        else:
            cem_out = cem_plan_batch(
                model=model,
                z_seed_B=z_seed,
                goal_B=goal,
                lang_emb_B=lang_emb,
                plan_horizon=plan_horizon,
                n_samples=cem_samples,
                n_iters=cem_iters,
                n_elites=cem_elites,
                var_scale=cem_var_scale,
                momentum=cem_momentum,
                cost_fn=cost_fn,
                ctxt_window=ctxt_window,
                device=device,
                action_prior_mean=action_prior_mean,
                seed_acts_B=seed_acts,
                yaw_stats=yaw_stats,
                action_prior_std=action_prior_std,
                traj_weight=traj_weight,
                temporal_corr=temporal_corr,
                action_corr=action_corr,
                gt_action_sample_B=gt_acts_sample_B,
                action_cluster_mean_B=cluster_mean_B,
                action_cluster_std_B=cluster_std_B,
                return_samples=plot_samples and output_plot is not None,
                return_iter_info=output_plot is not None,
                unicycle_stats=unicycle_stats_dict,
            )
            pred_acts_B, sample_acts_B, sample_costs_B, _iter_info = cem_out
            if gt_action_cost and gt_acts_cost_B is not None:
                cem_oracle = cem_plan_batch(
                    model=model,
                    z_seed_B=z_seed,
                    goal_B=goal,
                    lang_emb_B=lang_emb,
                    plan_horizon=plan_horizon,
                    n_samples=cem_samples,
                    n_iters=cem_iters,
                    n_elites=cem_elites,
                    var_scale=cem_var_scale,
                    momentum=cem_momentum,
                    cost_fn=cost_fn,
                    ctxt_window=ctxt_window,
                    device=device,
                    action_prior_mean=action_prior_mean,
                    seed_acts_B=seed_acts,
                    yaw_stats=yaw_stats,
                    action_prior_std=action_prior_std,
                    traj_weight=traj_weight,
                    temporal_corr=temporal_corr,
                    action_corr=action_corr,
                    gt_action_sample_B=None,
                    action_cluster_mean_B=None,
                    action_cluster_std_B=None,
                    return_samples=False,
                    gt_action_cost_B=gt_acts_cost_B,
                    unicycle_stats=unicycle_stats_dict,
                )
                pred_acts_oracle_B = cem_oracle[0]
        if debug:
            torch.cuda.synchronize(device)
        t_cem += time.perf_counter() - _t2

        # ── Post-processing ───────────────────────────────────────────────
        _t3 = time.perf_counter()
        b_end = min(B, num_examples - n_done) if num_examples is not None else B
        min_cost_B = None
        if sample_costs_B is not None:
            _sc = np.asarray(sample_costs_B)
            min_cost_B = _sc.min(axis=-1) if _sc.ndim == 2 else _sc
        if include_gt_sample and sample_costs_B is not None and b_end > 0:
            gt_sample_idx = (cem_samples - 1) if cem_samples > 1 else None
            if gt_sample_idx is not None:
                costs_last = np.array(sample_costs_B[b_end - 1]).reshape(-1)
                if gt_sample_idx < costs_last.size:
                    ranks = np.argsort(np.argsort(costs_last))
                    _last_gt_sample_rank = int(ranks[gt_sample_idx])
                    _last_gt_sample_cost = float(costs_last[gt_sample_idx])
                    _last_gt_sample_n = int(costs_last.size)

        # GT actions start at frame seed_frames-1 (the last seed frame), matching
        # the CEM's plan_horizon actions that drive frames [seed_frames-1 .. goal_idx-1].
        gt_acts_norm_B   = actions[:b_end, act_start:act_start + plan_horizon].cpu().numpy()  # (b_end, H, A)
        pred_acts_norm_B = pred_acts_B[:b_end]
        pred_acts_oracle_norm_B = (pred_acts_oracle_B[:b_end]
                       if pred_acts_oracle_B is not None else None)
        gt_unnorm = np.empty_like(gt_acts_norm_B)
        pred_unnorm = np.empty_like(pred_acts_norm_B)
        pred_oracle_unnorm = (np.empty_like(pred_acts_oracle_norm_B)
                      if pred_acts_oracle_norm_B is not None else None)

        # State-derived trajectory: proprio observations from seed_frames-1 to goal_idx (inclusive).
        proprio_norm_full = batch["observation"]["proprio"][:b_end, act_start:act_start + plan_horizon + 1].cpu().numpy()  # (b, T+1, 3)
        proprio_metric_full = np.empty_like(proprio_norm_full)

        _gt_stats  = gt_ds_stats if gt_ds_stats is not None else ds_stats
        _gt_fb     = list(_gt_stats.keys())[0]   # gt_stats has one key (the eval dataset)
        _ckpt_fb   = list(ds_stats.keys())[0]    # ckpt_stats may have multiple; pick first
        for b in range(b_end):
            # gt_ds_stats only contains the eval dataset — use its single key
            _gt_key   = ds_names[b] if ds_names[b] in _gt_stats else _gt_fb
            _ckpt_key = ds_names[b] if ds_names[b] in ds_stats  else _ckpt_fb
            gt_unnorm[b]   = unnormalize_bound_action(gt_acts_norm_B[b],
                                                      _gt_stats[_gt_key]["action"])
            pred_unnorm[b] = unnormalize_bound_action(pred_acts_norm_B[b],
                                                      ds_stats[_ckpt_key]["action"])
            if pred_oracle_unnorm is not None:
                pred_oracle_unnorm[b] = unnormalize_bound_action(
                    pred_acts_oracle_norm_B[b], ds_stats[_ckpt_key]["action"]
                )
            # Use per-sample dataset's proprio stats; fall back to action stats if proprio absent
            p_stats = _gt_stats[_gt_key].get("proprio", _gt_stats[_gt_key]["action"])
            proprio_metric_full[b] = unnormalize_bound_action(proprio_norm_full[b], p_stats)


        # print("prop shape", proprio_metric_full.shape, proprio_metric_full[..., 2])
        # Dead-reckon in window-start's robot frame (heading=0 at t=0).
        # Actions are already in local robot frame at each step, so no initial heading offset needed.
        # This matches xy_state_B which is also rotated to heading=0 via rotate_to_initial_heading.
        xy_gt_B,   headings_gt_B   = compute_xy_trajectory_batch(gt_unnorm)    # (b, H+1, 2), (b, H+1)
        xy_pred_B, headings_pred_B = compute_xy_trajectory_batch(pred_unnorm)
        xy_pred_oracle_B, headings_pred_oracle_B = (None, None)
        if pred_oracle_unnorm is not None:
            xy_pred_oracle_B, headings_pred_oracle_B = compute_xy_trajectory_batch(pred_oracle_unnorm)

        proprio_metric_B  = proprio_metric_full[..., :2]                             # (b, T, 2) x_fwd, y_left
        xy_state_B        = proprio_metric_B - proprio_metric_B[:, :1, :]            # window-relative, episode-frame-0
        xy_state_B, headings_state_B = rotate_to_initial_heading(xy_state_B, proprio_metric_full[..., 2])  # → window-start frame
        

        # Keep last example's data for the post-loop plot
        if output_plot is not None:
            last_b = b_end - 1
            goal_t = seed_frames - 1 + plan_horizon          # absolute frame index of goal
            mid_t  = (seed_frames - 1 + goal_t) // 2        # midpoint between plan start and goal
            _last_img_first  = images[last_b, seed_frames - 1].cpu().numpy()  # (C, H, W) — plan start
            _last_img_middle = images[last_b, mid_t].cpu().numpy()            # (C, H, W)
            _last_img_goal   = images[last_b, goal_t].cpu().numpy()           # (C, H, W)
            _last_xy_gt      = xy_gt_B[last_b]
            _last_xy_pred    = xy_pred_B[last_b]
            _last_xy_state   = xy_state_B[last_b]
            _last_gt_headings    = headings_gt_B[last_b]
            _last_pred_headings  = headings_pred_B[last_b]
            _last_state_headings = headings_state_B[last_b]
            _last_gt_actions = gt_unnorm[last_b]                      # (T, A) metric
            _last_pred_actions = pred_unnorm[last_b]                  # (T, A) metric
            _last_lang       = raw_language[last_b] if last_b < len(raw_language) else ""
            if plot_samples and sample_acts_B is not None:
                _ckpt_key = ds_names[last_b] if ds_names[last_b] in ds_stats else _ckpt_fb
                _last_sample_actions = unnormalize_bound_action(
                    sample_acts_B[last_b], ds_stats[_ckpt_key]["action"]
                )
                if sample_costs_B is not None:
                    _last_sample_costs = sample_costs_B[last_b]

            # PCA feature visualization: capture first, GT goal, and predicted goal features
            _last_feat_first    = z_seed[last_b, 0, 0].cpu().float().numpy()  # (gH, gW, D)
            _last_feat_goal_gt  = goal[last_b, 0, 0].cpu().float().numpy()    # (gH, gW, D)
            if pred_final is not None and last_b < pred_final.shape[0]:
                _last_feat_goal_pred = pred_final[last_b, 0, 0].cpu().float().numpy()
            else:
                _pa = torch.tensor(
                    pred_acts_B[last_b:last_b + 1], dtype=torch.float32, device=device
                )  # (1, T, A)
                _sa_p = (
                    seed_acts[last_b:last_b + 1].permute(1, 0, 2).to(torch.bfloat16)
                    if seed_acts is not None else None
                )
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _pf, _ = cem_unroll(
                        model,
                        z_seed[last_b:last_b + 1].to(torch.bfloat16),
                        _pa.to(torch.bfloat16).permute(1, 0, 2),
                        lang_emb[last_b:last_b + 1].to(torch.bfloat16)
                        if lang_emb is not None else None,
                        ctxt_window,
                        seed_acts_SNA=_sa_p,
                    )
                _last_feat_goal_pred = _pf[0, 0, 0].cpu().float().numpy()

            # Per-CEM-iteration best trajectory and first-iter samples for the last example
            if _iter_info is not None:
                _iter_means_np  = _iter_info[0]
                _iter_costs_np  = _iter_info[1]
                _first_samps_np = _iter_info[2]
                _ckpt_key = ds_names[last_b] if ds_names[last_b] in ds_stats else _ckpt_fb
                _last_iter_xy_trajs = []
                for _it in range(_iter_means_np.shape[0]):
                    _mean_it = _iter_means_np[_it, last_b]  # (T, A_internal)
                    # Expand to 4D for unnorm (unicycle → 3D → sin/cos if needed)
                    _mean_t = torch.tensor(_mean_it[None], dtype=torch.float32)
                    if unicycle_stats_dict is not None:
                        _mean_t = _unicycle_to_3d(_mean_t, unicycle_stats_dict)
                    if yaw_stats is not None:
                        _mean_4d = _expand_yaw_to_sincos(_mean_t, yaw_stats).numpy()[0]  # (T, 4)
                    else:
                        _mean_4d = _mean_t.numpy()[0]
                    _unnorm_it = unnormalize_bound_action(_mean_4d, ds_stats[_ckpt_key]["action"])
                    _xy_it, _ = compute_xy_trajectory_gt(_unnorm_it)
                    _last_iter_xy_trajs.append(_xy_it)
                _last_iter_costs = _iter_costs_np[:, last_b]  # (n_iters,)
                # First-iter sample trajectories: (T, N, 4) → list of (T+1, 2)
                _last_first_iter_sample_xys = None
                if _first_samps_np is not None:
                    _fs_b = _first_samps_np[last_b]  # (T, N, 4)
                    _last_first_iter_sample_xys = []
                    for _n in range(_fs_b.shape[1]):
                        _unnorm_n = unnormalize_bound_action(_fs_b[:, _n, :], ds_stats[_ckpt_key]["action"])
                        _xy_n, _ = compute_xy_trajectory_gt(_unnorm_n)
                        _last_first_iter_sample_xys.append(_xy_n)
            else:
                _last_iter_xy_trajs = None
                _last_iter_costs = None
                _last_first_iter_sample_xys = None

        if orientation_metric == "yaw":
            m_B = compute_nav_metrics_batch_yaw(
                xy_pred_B, xy_gt_B, headings_pred_B, headings_gt_B
            )
        else:
            m_B = compute_nav_metrics_batch(xy_pred_B, xy_gt_B)
        for k, v in m_B.items():
            all_metrics[k].extend(v.tolist())

        m_oracle_B = None
        if pred_oracle_unnorm is not None and all_metrics_gt_l2 is not None:
            if orientation_metric == "yaw":
                m_oracle_B = compute_nav_metrics_batch_yaw(
                    xy_pred_oracle_B, xy_gt_B, headings_pred_oracle_B, headings_gt_B
                )
            else:
                m_oracle_B = compute_nav_metrics_batch(xy_pred_oracle_B, xy_gt_B)
            for k, v in m_oracle_B.items():
                all_metrics_gt_l2[k].extend(v.tolist())

        if min_cost_B is not None:
            all_cem_costs.extend(min_cost_B[:b_end].tolist())
        for b in range(b_end):
            rec = {
                "idx": n_done + b,
                "language": raw_language[b] if b < len(raw_language) else "",
                "dataset": ds_names[b] if b < len(ds_names) else fallback_ds,
                **{k: round(float(m_B[k][b]), 6) for k in m_B},
                "xy_pred": xy_pred_B[b].tolist(),
                "xy_gt": xy_gt_B[b].tolist(),
                "headings_pred": headings_pred_B[b].tolist(),
                "headings_gt": headings_gt_B[b].tolist(),
            }
            if min_cost_B is not None and b < len(min_cost_B):
                rec["cem_cost"] = round(float(min_cost_B[b]), 6)
            if m_oracle_B is not None:
                rec.update({f"{k}_gt_l2": round(float(m_oracle_B[k][b]), 6) for k in m_oracle_B})
            records.append(rec)
        n_done += b_end
        t_post += time.perf_counter() - _t3

        if checkpoint_json is not None and (n_done - _last_save_n) >= save_every:
            _agg_ckpt = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
                for k, v in all_metrics.items() if v
            }
            _ckpt = {"n_done": n_done, "aggregate": _agg_ckpt, "per_example": records}
            if all_metrics_gt_l2 is not None:
                _agg_ckpt_gt_l2 = {
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
                    for k, v in all_metrics_gt_l2.items() if v
                }
                _ckpt["aggregate_gt_l2"] = _agg_ckpt_gt_l2
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_json)), exist_ok=True)
            with open(checkpoint_json, "w") as _f:
                json.dump(_ckpt, _f)
            _last_save_n = n_done

        t_total = t_data + t_encode + t_cem + t_post
        pbar.set_postfix(
            AOE=f"{np.mean(all_metrics['AOE']):.1f}°",
            ADE=f"{np.mean(all_metrics['ADE']):.4f}m",
            cost=f"{np.mean(all_cem_costs):.3f}" if all_cem_costs else "n/a",
            data=f"{t_data/t_total*100:.0f}%",
            enc=f"{t_encode/t_total*100:.0f}%",
            cem=f"{t_cem/t_total*100:.0f}%",
            post=f"{t_post/t_total*100:.0f}%",
        )

        _t0 = time.perf_counter()

    if output_plot is not None and _last_img_first is not None:
        plot_last_example(
            img_first=_last_img_first,
            img_middle=_last_img_middle,
            img_goal=_last_img_goal,
            xy_gt=_last_xy_gt,
            xy_pred=_last_xy_pred,
            xy_state=_last_xy_state,
            headings_gt=_last_gt_headings,
            headings_pred=_last_pred_headings,
            headings_state=_last_state_headings,
            gt_actions=_last_gt_actions,
            pred_actions=_last_pred_actions,
            language=_last_lang,
            sample_actions=_last_sample_actions,
            sample_costs=_last_sample_costs,
            feat_first=_last_feat_first,
            feat_goal_gt=_last_feat_goal_gt,
            feat_goal_pred=_last_feat_goal_pred,
            iter_xy_trajs=_last_iter_xy_trajs,
            iter_costs=_last_iter_costs,
            first_iter_sample_xys=_last_first_iter_sample_xys,
            save_path=output_plot,
        )
        if _last_gt_sample_rank is not None and _last_gt_sample_n is not None:
            print(
                "  GT sample cost rank (0=best): "
                f"{_last_gt_sample_rank}/{_last_gt_sample_n - 1}  "
                f"cost={_last_gt_sample_cost:.4f}"
            )

    if gt_baseline and hasattr(eval_dataset, "_wm_cosim_accum") and eval_dataset._wm_cosim_accum:
        cosims = np.array(eval_dataset._wm_cosim_accum)
        print(f"\n  WM oracle (GT actions → predicted vs actual goal feature):")
        print(f"    cosine similarity  mean={cosims.mean():.4f}  "
              f"median={np.median(cosims):.4f}  "
              f"p10={np.percentile(cosims, 10):.4f}")
        print(f"    Interpretation: >0.95 = WM is accurate, planner is the bottleneck")
        print(f"                    <0.80 = WM rollout error is the fundamental bottleneck")
        del eval_dataset._wm_cosim_accum  # reset for potential re-use

    t_total = t_data + t_encode + t_cem + t_post
    planner_label = planner.upper()
    print(f"\n  Latency breakdown (total {t_total:.1f}s over {n_done} examples):")
    print(f"    data/H2D : {t_data:6.1f}s  ({t_data/t_total*100:4.1f}%)")
    print(f"    encode   : {t_encode:6.1f}s  ({t_encode/t_total*100:4.1f}%)")
    print(f"    {planner_label:<9}: {t_cem:6.1f}s  ({t_cem/t_total*100:4.1f}%)")
    print(f"    post     : {t_post:6.1f}s  ({t_post/t_total*100:4.1f}%)")
    if n_done > 0:
        print(f"    throughput: {n_done/t_total:.1f} examples/s  "
              f"({t_total/n_done*1000:.0f} ms/example)")

    agg = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "median": float(np.median(v)),
        }
        for k, v in all_metrics.items() if v
    }
    if all_cem_costs:
        agg["cem_cost"] = {
            "mean": float(np.mean(all_cem_costs)),
            "std": float(np.std(all_cem_costs)),
            "median": float(np.median(all_cem_costs)),
        }
        print(f"\n  CEM cost (final iter, min over samples):  "
              f"mean={agg['cem_cost']['mean']:.4f}  "
              f"median={agg['cem_cost']['median']:.4f}  "
              f"std={agg['cem_cost']['std']:.4f}")
    agg_gt_l2 = None
    if all_metrics_gt_l2 is not None:
        agg_gt_l2 = {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "median": float(np.median(v)),
            }
            for k, v in all_metrics_gt_l2.items() if v
        }
    return agg, records, agg_gt_l2


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builder helper
# ═══════════════════════════════════════════════════════════════════════════════

def _build_test_loader(
    cfg,
    data_dir,
    split_name,
    dataset_name,
    stats,
    collate_fn,
    max_windows_per_traj=None,
    batch_size=1,
    window_size=None,
):
    """Build a non-overlapping, reproducible test loader for one split."""
    std_fn = ModuleSpec.create(
        gnm_action_angle_dataset_transform,
        action_horizon=cfg.action_horizon,
    )
    chunk_start_idx, chunk_stride = 0, window_size  # start from frame 0, non-overlapping
    tf_ds, _ = make_test_dataset(
        name=dataset_name,
        data_dir=data_dir,
        normalize=True,
        split=split_name,
        standardize_fn=std_fn,
        dataset_statistics=stats[dataset_name],
        window_size=window_size if window_size is not None else cfg.window_size,
        image_obs_keys={"primary": "image"},
        proprio_obs_key="state",
        language_key="language_instruction",
        resize_size={"primary": (cfg.img_size, cfg.img_size)},
        max_windows_per_traj=max_windows_per_traj,
        chunk_start_idx=chunk_start_idx,
        chunk_stride=chunk_stride,
    )
    pt_ds = TorchRLDSDataset(tf_ds, text_processor=None, train=False)
    loader = DataLoader(
        pt_ds, batch_size=batch_size, num_workers=0,
        collate_fn=collate_fn, pin_memory=True,
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a latent world model with CEM planning (AOE/MAOE/ADE/MADE)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint (best_model.pt or checkpoint_stepN.pt)")
    parser.add_argument("--encoder", default="dino",
                        choices=["dino", "dinov3", "vjepa"],
                        help="Encoder type used during training")
    parser.add_argument("--data_dir", default=None,
                        help="Root directory containing the RLDS dataset (default: cfg.data_dir)")
    parser.add_argument("--dataset_name", default="cast_filtered_dataset",
                        help="dataset name")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test_seen", "test_unseen", "test_all"],
                        help="Which split(s) to evaluate using non-overlapping deterministic windows. "
                             "test_all evaluates both test_seen and test_unseen.")
    parser.add_argument("--plan_horizon", type=int, default=6,
                        help="Number of waypoints to predict (must be < window_size)")
    parser.add_argument("--cem_samples", type=int, default=128,
                        help="CEM candidate trajectories per iteration")
    parser.add_argument("--cem_iters", type=int, default=6,
                        help="CEM refinement iterations")
    parser.add_argument("--cem_elites", type=int, default=4,
                        help="CEM elite candidates per iteration")
    parser.add_argument("--cem_var_scale", type=float, default=2.0,
                        help="Initial std for CEM action distribution (scales action_prior_std)")
    parser.add_argument("--unicycle", action="store_true", default=False,
                        help="Use unicycle motion model for CEM/MPPI: sample (u, delta_yaw) "
                             "and derive dx=u*cos(yaw), dy=u*sin(yaw)")
    parser.add_argument("--cem_momentum", type=float, default=0.1,
                        help="CEM momentum for mean/std update (0 = no momentum)")
    parser.add_argument("--cost_fn", default=None,
                        choices=["cosine", "mse", "l1", None],
                        help="CEM cost function (None = combined cosine+mse)")
    parser.add_argument("--ctxt_window", type=int, default=None,
                        help="Context window for cem_unroll (default: from config)")
    parser.add_argument("--output_json", default=None,
                        help="Path to save JSON results (optional)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Examples per forward pass — higher = faster but more GPU memory (default: 8)")
    parser.add_argument("--max_windows_per_traj", type=int, default=None,
                        help="Keep at most N windows per trajectory (e.g. 1 for fast debug runs)")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Stop after this many examples (default: full split)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for CEM sampling and language instruction selection (default: 42)")
    parser.add_argument("--planner", default="cem", choices=["cem", "mppi"],
                        help="Planning algorithm: cem (hard elites) or mppi (soft temperature weighting, default: cem)")
    parser.add_argument("--mppi_temperature", type=float, default=1.0,
                        help="MPPI temperature — higher = sharper weighting toward low-cost samples (default: 1.0)")
    parser.add_argument("--mppi_std_min", type=float, default=0.01,
                        help="MPPI minimum std — prevents distribution collapse (default: 0.01)")
    parser.add_argument("--mppi_std_max", type=float, default=2.0,
                        help="MPPI maximum std — prevents distribution explosion (default: 2.0)")
    parser.add_argument("--orientation_metric", default="vector",
                        choices=["vector", "yaw"],
                        help="Orientation error metric: vector (default, direction-to-waypoint) or yaw (heading delta)")
    parser.add_argument("--output_plot", default=None,
                        help="Save a visualization of the final example to this path (e.g. eval_results/last.png)")
    parser.add_argument("--recompute_stats", action="store_true",
                        help="Ignore checkpoint stats and recompute from the dataset. "
                             "Required after the CAST state encoding changed (GPS → robot-frame-relative). "
                             "The action stats are unaffected; only proprio/state viz uses fresh stats.")
    parser.add_argument("--debug", action="store_true",
                        help="Insert torch.cuda.synchronize() after encode and planner for accurate "
                             "per-stage timing. Slower; only enable when profiling.")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile on the predictor (useful for debugging / one-off runs).")
    parser.add_argument("--checkpoint_json", default=None,
                        help="Path to periodically save in-progress results (use {split} as a "
                             "placeholder, e.g. eval_results/ckpt_{split}.json)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N examples (default: 100)")
    parser.add_argument("--seed_frames", type=int, default=4,
                        help="GT seed frames used to prime the WM context before planning.")
    parser.add_argument("--eval_window_size", type=int, default=12,
                        help="Data loader window size. Fixed at 12 so the same GT trajectories are "
                             "loaded across runs with different seed_frames / plan_horizon.")
    parser.add_argument("--gt_baseline", action="store_true",
                        help="Skip the planner and use ground-truth actions directly. "
                             "Measures the WM/metric ceiling — if GT baseline is close to the "
                             "planner result, the WM is the bottleneck, not the optimizer.")
    parser.add_argument("--gt_action_cost", action="store_true",
                        help="Also evaluate the planner using L2 distance to GT actions as the "
                             "cost (optimizer lower-bound diagnostic).")
    parser.add_argument("--traj_weight", type=float, default=0.0,
                        help="Weight on mean intermediate-step cost relative to final-step cost "
                             "(0 = final only, 1.0 = equal weight, try 0.5-2.0). "
                             "Adds denser orientation signal from early, low-drift rollout steps.")
    parser.add_argument("--temporal_corr", default="0.0",
                        help="AR(1) rho for noise: scalar (e.g. 0.8) applied to all dims, "
                             "or comma-separated per-dim values (e.g. 0.0,0.0,0.8 for "
                             "x,y,yaw). 0=i.i.d., 0.8=smooth.")
    parser.add_argument("--action_corr_json", default=None,
                        help="Path to JSON with cross-dim action correlation matrix. "
                            "Expected key: 'corr' (A x A).")
    parser.add_argument("--action_cluster_npz", default=None,
                        help="Path to NPZ with action cluster means/stds (from estimate_temporal_corr.py)")
    parser.add_argument("--cluster_index", type=int, default=None,
                        help="If set, use this cluster index for all examples (default: sample by counts)")
    parser.add_argument("--include_gt_sample", action="store_true",
                        help="Include GT action sequence among planner samples (diagnostic).")
    parser.add_argument("--plot_samples", action="store_true",
                        help="Overlay final-iter CEM/MPPI samples on the action plot (requires --output_plot).")

    args = parser.parse_args()
    # Parse --temporal_corr: scalar or per-dim list
    _tc = args.temporal_corr
    if isinstance(_tc, str) and "," in _tc:
        args.temporal_corr = [float(v) for v in _tc.split(",")]
    else:
        args.temporal_corr = float(_tc)

    action_corr = None
    if args.action_corr_json:
        with open(args.action_corr_json, "r") as f:
            _corr_payload = json.load(f)
        action_corr = _corr_payload.get("corr", _corr_payload.get("action_corr"))
        if action_corr is None:
            raise ValueError("action_corr_json must contain a 'corr' key")
        print(f"Loaded action correlation matrix from {args.action_corr_json}")

    action_cluster_means = None
    action_cluster_stds = None
    action_cluster_counts = None
    if args.action_cluster_npz:
        cluster_npz = np.load(args.action_cluster_npz)
        action_cluster_means = cluster_npz["means"]
        action_cluster_stds = cluster_npz["stds"]
        if "counts" in cluster_npz:
            action_cluster_counts = cluster_npz["counts"]
        print(f"Loaded action clusters from {args.action_cluster_npz}")

    import random
    import tensorflow as _tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _tf.random.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    planner_detail = (
        f"temperature={args.mppi_temperature}" if args.planner == "mppi"
        else f"elites={args.cem_elites}"
    )
    print(f"Encoder: {args.encoder} | Split: {args.split} | Plan horizon: {args.plan_horizon} | "
          f"{args.planner.upper()}: {args.cem_samples} samples × {args.cem_iters} iters "
          f"({planner_detail}, momentum={args.cem_momentum}, cost={args.cost_fn})")

    # ── Load config ───────────────────────────────────────────────────────
    CKPT_DIR = os.path.dirname(args.checkpoint)
    config_path = os.path.join(CKPT_DIR, "config.py")  # look for config.py in checkpoint dir
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder)

    # Derive window size from eval args: exactly as many frames as needed.
    eval_window_size = args.eval_window_size if args.eval_window_size is not None else args.seed_frames + args.plan_horizon

    ctxt_window = args.ctxt_window if args.ctxt_window is not None else cfg.ctxt_window
    print(f"Context window for WM unroll: {ctxt_window} frames (set via --ctxt_window or from config)")

    # ── Build model ───────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = build_world_model(cfg).to(device)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    ckpt_step = ckpt.get("step", "?")
    ckpt_loss = ckpt.get("val_loss", "?")
    print(f"  Loaded checkpoint (step={ckpt_step}, val_loss={ckpt_loss})")
    model.eval()

    # Compile the predictor (called once per unroll step inside the planner).
    # reduce-overhead uses CUDA graphs where possible — best for small, repeated
    # forward passes like ours. First few calls trigger compilation; cached afterward.
    if not args.no_compile:
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
        print("  Predictor compiled with mode='reduce-overhead'")

    # ── Determine which splits to evaluate ───────────────────────────────
    if args.split == "test_all":
        splits_to_eval = ["test_seen", "test_unseen"]
    else:
        splits_to_eval = [args.split]

    dataset_name = args.dataset_name
    data_dir = f"{args.data_dir or cfg.get('data_dir', './data')}/{dataset_name}"

    # ── Load normalization stats ──────────────────────────────────────────────
    # ckpt_stats: from the checkpoint (training-split distribution + mask).
    #   Used for: planner initialisation, predicted-action unnorm.
    # gt_stats:   freshly computed from the dataset (no mask).
    #   Used for: data loader (GT action normalisation) + GT-action unnorm.
    ckpt_stats = ckpt.get("dataset_statistics")
    use_ckpt_stats = (
        not args.recompute_stats
        and isinstance(ckpt_stats, dict)
        and dataset_name in ckpt_stats
    )
    if use_ckpt_stats:
        stats = _to_numpy_tree(ckpt_stats)
        print(f"\nUsing checkpoint normalization stats for model/planner ({dataset_name}).")
    else:
        stats = None  # filled below alongside gt_stats

    print(f"\nLoading fresh normalization stats from dataset for GT actions ({dataset_name}) ...")
    standardize_fn = ModuleSpec.create(
        gnm_action_angle_dataset_transform,
        action_horizon=cfg.action_horizon,
    )
    _, _fresh = make_dataset_from_rlds(
        name=dataset_name,
        data_dir=data_dir,
        train=True,
        standardize_fn=standardize_fn,
        image_obs_keys={"primary": "image"},
        proprio_obs_key="state",
        language_key="language_instruction",
        action_proprio_normalization_type=NormalizationType.BOUNDS,
        force_recompute_dataset_statistics=args.recompute_stats,
        skip_norm=True,
    )
    gt_stats = {dataset_name: _fresh}
    if stats is None:
        stats = gt_stats   # no checkpoint stats — fall back to fresh for planner too
    ckpt_stats_saved = stats   # preserve original ckpt stats before any gt_baseline override
    print("Statistics ready.")

    # ── Evaluate each split ───────────────────────────────────────────────
    all_results: Dict[str, dict] = {}

    for split_name in splits_to_eval:
        mwpt = args.max_windows_per_traj
        label = f"{split_name} (max {mwpt} window/traj)" if mwpt else f"{split_name} (full split)"
        print(f"\n=== {label} ===")
        loader = _build_test_loader(cfg, data_dir, split_name, dataset_name, gt_stats, collate_fn,
                                    max_windows_per_traj=mwpt, batch_size=args.batch_size,
                                    window_size=eval_window_size)

        # Derive per-split plot path: foo/last.png → foo/last_test_seen.png
        plot_path = None
        if args.output_plot:
            base, ext = os.path.splitext(args.output_plot)
            plot_path = f"{base}_{split_name}{ext or '.png'}"
        if args.gt_baseline:
            # the stats should be the same for GT baseline since no planner
            stats = gt_stats
        agg, records, agg_gt_l2 = eval_dataset(
            model=model,
            loader=loader,
            ds_stats=stats,
            device=device,
            plan_horizon=args.plan_horizon,
            cem_samples=args.cem_samples,
            cem_iters=args.cem_iters,
            cem_elites=args.cem_elites,
            cem_var_scale=args.cem_var_scale,
            cem_momentum=args.cem_momentum,
            cost_fn=args.cost_fn,
            ctxt_window=ctxt_window,
            num_examples=args.num_examples,
            planner=args.planner,
            mppi_temperature=args.mppi_temperature,
            mppi_std_min=args.mppi_std_min,
            mppi_std_max=args.mppi_std_max,
            orientation_metric=args.orientation_metric,
            output_plot=plot_path,
            debug=args.debug,
            checkpoint_json=args.checkpoint_json.replace("{split}", split_name) if args.checkpoint_json else None,
            save_every=args.save_every,
            seed_frames=args.seed_frames,
            gt_baseline=args.gt_baseline,
            traj_weight=args.traj_weight,
            temporal_corr=args.temporal_corr,
            action_corr=action_corr,
            include_gt_sample=args.include_gt_sample,
            action_cluster_means=action_cluster_means,
            action_cluster_stds=action_cluster_stds,
            action_cluster_counts=action_cluster_counts,
            cluster_index=args.cluster_index,
            gt_ds_stats=gt_stats,
            ckpt_ds_stats=ckpt_stats_saved,
            plot_samples=args.plot_samples,
            gt_action_cost=args.gt_action_cost,
            unicycle=args.unicycle,
        )
        all_results[split_name] = {"aggregate": agg, "per_example": records}
        if agg_gt_l2 is not None:
            all_results[split_name]["aggregate_gt_l2"] = agg_gt_l2

    # ── Summary table ─────────────────────────────────────────────────────
    width = 72
    print(f"\n{'=' * width}")
    print(f"  EVALUATION SUMMARY  (step={ckpt_step}, encoder={args.encoder},"
          f" horizon={args.plan_horizon})")
    print(f"{'=' * width}")
    print(f"  {'Split':<20}  {'AOE':>8}  {'MAOE':>8}  {'ADE':>10}  {'MADE':>10}")
    print(f"  {'-' * (width - 2)}")
    for split_name, res in all_results.items():
        a = res["aggregate"]
        n = len(res["per_example"])
        print(f"  {split_name:<20}  "
              f"{a['AOE']['mean']:>7.2f}°  "
              f"{a['MAOE']['mean']:>7.2f}°  "
              f"{a['ADE']['mean']:>9.4f}m  "
              f"{a['MADE']['mean']:>9.4f}m  n={n}")
    print(f"{'=' * width}")

    has_gt_l2 = any("aggregate_gt_l2" in res for res in all_results.values())
    if has_gt_l2:
        print(f"\n{'=' * width}")
        print("  PLANNER LOWER BOUND  (GT L2 ACTION COST)")
        print(f"{'=' * width}")
        print(f"  {'Split':<20}  {'AOE':>8}  {'MAOE':>8}  {'ADE':>10}  {'MADE':>10}")
        print(f"  {'-' * (width - 2)}")
        for split_name, res in all_results.items():
            if "aggregate_gt_l2" not in res:
                continue
            a = res["aggregate_gt_l2"]
            n = len(res["per_example"])
            print(f"  {split_name:<20}  "
                  f"{a['AOE']['mean']:>7.2f}°  "
                  f"{a['MAOE']['mean']:>7.2f}°  "
                  f"{a['ADE']['mean']:>9.4f}m  "
                  f"{a['MADE']['mean']:>9.4f}m  n={n}")
        print(f"{'=' * width}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        out = {
            "checkpoint": str(args.checkpoint),
            "step": ckpt_step,
            "encoder": args.encoder,
            "split": args.split,
            "plan_horizon": args.plan_horizon,
            "seed_frames": args.seed_frames,
            "window_size": eval_window_size,
            "planner": args.planner,
            "gt_action_cost": args.gt_action_cost,
            "cem": {
                "samples": args.cem_samples,
                "iters": args.cem_iters,
                "elites": args.cem_elites if args.planner == "cem" else None,
                "temperature": args.mppi_temperature if args.planner == "mppi" else None,
                "std_min": args.mppi_std_min if args.planner == "mppi" else None,
                "std_max": args.mppi_std_max if args.planner == "mppi" else None,
                "var_scale": args.cem_var_scale,
                "momentum": args.cem_momentum,
                "cost_fn": args.cost_fn,
                "ctxt_window": ctxt_window,
            },
            "results": {
                split_name: {
                    "aggregate": res["aggregate"],
                    **({"aggregate_gt_l2": res["aggregate_gt_l2"]}
                       if "aggregate_gt_l2" in res else {}),
                    "per_example": res["per_example"],
                }
                for split_name, res in all_results.items()
            },
            "action_corr_json": args.action_corr_json,
            "action_cluster_npz": args.action_cluster_npz,
            "cluster_index": args.cluster_index,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()