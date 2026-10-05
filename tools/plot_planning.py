#!/usr/bin/env python3
"""
CEM planning visualiser — per-example GT / Pred / Plan plots.

For each requested example index the script saves:

  example_NNNN/
    GT/    — frame_00_rgb.png .. frame_09_rgb.png   (seed_frames + plan_horizon)
             frame_00_pca.png .. frame_09_pca.png
    Pred/  — pred_step_01_pca.png .. pred_step_06_pca.png  (model rollout with planned actions)
    Plan/  — trajectories.png   (sample fan coloured by iteration + GT)
             cost_vs_iter.png   (minimum cost curve across CEM iterations)

Usage:
    python tools/plot_planning.py \\
        --checkpoint /path/to/best_model.pt \\
        --encoder dino \\
        --indices 0 1 2 3 \\
        --seed 42 --dataset_name cast_filtered_dataset --split val \\
        --plan_horizon 6 --seed_frames 4 --ctxt_window 4 --eval_window_size 12 \\
        --cem_samples 128 --cem_iters 6 --cem_elites 4 \\
        --cem_var_scale 2.0 --cem_momentum 0.1 \\
        --traj_weight 0.1 --temporal_corr 0.0 \\
        --unicycle --no_compile \\
        --output eval_results/planning_plots
"""

import argparse
import importlib.util
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import logging
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("fontTools").setLevel(logging.WARNING)

from eval_planning import (
    cem_plan_batch,
    cem_unroll,
    compute_xy_trajectory_gt,
    unnormalize_bound_action,
    _yaw_sincos_stats,
    _expand_yaw_to_sincos,
    _unicycle_to_3d,
    _renorm_gt_to_ckpt,
    _normalize_raw_language,
    _build_test_loader,
    PCAVisualizer,
)
from lmwm.world_model import build_world_model
from lmwm.data.torch_rlds_dataset import collate_fn
from lmwm.data.dataset import make_dataset_from_rlds
from lmwm.data.utils.cast_transforms import gnm_action_angle_dataset_transform
from lmwm.data.utils.data_utils import NormalizationType
from lmwm.data.spec import ModuleSpec

from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
import matplotlib.pyplot as plt

class GradientLegendLine:
    def __init__(self, label, cmap="viridis", lw=2.0, alpha=1.0, n_segments=32):
        self.label = label
        self.cmap = plt.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.lw = lw
        self.alpha = alpha
        self.n_segments = n_segments

    def get_label(self):
        return self.label


class HandlerGradientLine(HandlerBase):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height,
        fontsize, trans
    ):
        n = orig_handle.n_segments

        x = np.linspace(xdescent, xdescent + width, n + 1)
        y = np.full_like(x, ydescent + height / 2)

        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        colors = orig_handle.cmap(np.linspace(0.0, 1.0, n))

        line = LineCollection(
            segments,
            colors=colors,
            linewidths=orig_handle.lw,
            alpha=orig_handle.alpha,
            transform=trans,
        )

        return [line]
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy_tree(x):
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _hwc_uint8(img_chw):
    """(C, H, W) float-or-uint8 → (H, W, C) uint8."""
    if isinstance(img_chw, torch.Tensor):
        img_chw = img_chw.numpy()
    img = np.transpose(img_chw, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_examples_at_indices(loader, target_indices):
    """
    Iterate through loader and collect the examples whose global 0-based
    position is in target_indices.  Stops as soon as all targets are found.

    Returns dict: global_idx → {images, actions, language, ds_name}
    """
    targets = set(target_indices)
    max_idx = max(targets)
    collected = {}
    global_idx = 0

    for batch in loader:
        images  = batch["observation"]["image_primary"]   # (B, T, C, H, W) CPU
        actions = batch["action"]
        if actions.dim() == 4:
            actions = actions[:, :, 0, :]                  # (B, T, A)
        raw_lang     = _normalize_raw_language(batch.get("raw_language", ["navigate"]))
        ds_names_raw = batch.get("dataset_name", [None] * images.shape[0])
        B = images.shape[0]

        for b in range(B):
            if global_idx in targets:
                ds_b = ds_names_raw[b]
                if isinstance(ds_b, bytes):
                    ds_b = ds_b.decode("utf-8")
                elif ds_b is None:
                    ds_b = ""
                else:
                    ds_b = str(ds_b)
                collected[global_idx] = dict(
                    images   = images[b].clone(),           # (T, C, H, W) CPU float
                    actions  = actions[b].clone(),          # (T, A)
                    language = raw_lang[b] if b < len(raw_lang) else "navigate",
                    ds_name  = ds_b,
                )
            global_idx += 1
            if global_idx > max_idx and len(collected) >= len(targets):
                return collected

    if len(collected) < len(targets):
        missing = targets - set(collected.keys())
        print(f"  Warning: indices not found in dataset: {sorted(missing)}")
    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Model rollout helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_step_by_step(model, z_seed_1SvHWD, best_acts_T4, seed_acts_1SA,
                          lang_1D, ctxt_window, device):
    """
    Roll out the world model one step at a time using planned 4-D actions.

    Returns a list of plan_horizon (gH, gW, D) numpy arrays — the predicted
    feature map at each step of the horizon.

    z_seed_1SvHWD : (1, S, 1, gH, gW, D) float32 tensor on any device
    best_acts_T4  : (T, 4) numpy array — ckpt-normalised planned actions
    seed_acts_1SA : (1, S-1, A) tensor or None
    lang_1D       : (1, D_lang) tensor or None
    """
    T = best_acts_T4.shape[0]
    z = z_seed_1SvHWD.to(device)               # (1, S, 1, gH, gW, D)
    S = z.shape[1]
    *spatial, D_feat = z.shape[2:]             # spatial = [1, gH, gW]
    A = best_acts_T4.shape[1]

    acts = torch.tensor(best_acts_T4, dtype=torch.bfloat16, device=device)  # (T, 4)

    buf_len  = S + T
    feat_buf = z.to(torch.bfloat16).new_empty(1, buf_len, *spatial, D_feat)
    feat_buf[:, :S] = z.to(torch.bfloat16)
    act_buf  = acts.new_empty(1, buf_len, A)

    if seed_acts_1SA is not None and seed_acts_1SA.shape[1] > 0:
        act_buf[:, :S - 1] = seed_acts_1SA.to(device=device, dtype=acts.dtype)

    lang_1 = lang_1D.to(device=device, dtype=torch.bfloat16) if lang_1D is not None else None

    preds = []
    _ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with _ctx:
        for t in range(T):
            act_buf[:, S - 1 + t] = acts[t]
            ctx_end   = S + t
            ctx_start = max(0, ctx_end - ctxt_window)
            ctx_feats = feat_buf[:, ctx_start:ctx_end]   # (1, <=ctxt, *spatial, D)
            ctx_acts  = act_buf[:, ctx_start:ctx_end]    # (1, <=ctxt, A)

            pred      = model.forward_pred(ctx_feats, ctx_acts, lang_1)
            pred_next = pred[:, -1:]                     # (1, 1, *spatial, D)

            feat_buf[:, S + t] = pred_next[:, 0]
            preds.append(pred_next[0, 0, 0].float().cpu().numpy())  # (gH, gW, D)

    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Saving GT frames
# ─────────────────────────────────────────────────────────────────────────────

def save_gt_frames(images_T, z_feats_Tv, pca_vis, n_frames, out_dir):
    """
    Save individual GT RGB + PCA images for the first n_frames.

    images_T   : (T, C, H, W) float or uint8 tensor/array
    z_feats_Tv : (T, 1, gH, gW, D) tensor — view dim = 1
    pca_vis    : PCAVisualizer fitted on all patches of this example
    """
    from PIL import Image

    gt_dir = os.path.join(out_dir, "GT")
    os.makedirs(gt_dir, exist_ok=True)

    rgb_wh = None   # (W, H) PIL size for resizing PCA images
    for i in range(n_frames):
        rgb = _hwc_uint8(images_T[i])
        if rgb_wh is None:
            rgb_wh = (rgb.shape[1], rgb.shape[0])   # (W, H)
        Image.fromarray(rgb).save(os.path.join(gt_dir, f"frame_{i:02d}_rgb.png"))

        feat = z_feats_Tv[i, 0]
        if isinstance(feat, torch.Tensor):
            feat = feat.numpy()                      # (gH, gW, D)
        pca_rgb = pca_vis(feat)
        if pca_rgb.dtype != np.uint8:
            pca_rgb = (pca_rgb * 255).clip(0, 255).astype(np.uint8)
        pil_pca = Image.fromarray(pca_rgb)
        if rgb_wh is not None:
            pil_pca = pil_pca.resize(rgb_wh, Image.NEAREST)
        pil_pca.save(os.path.join(gt_dir, f"frame_{i:02d}_pca.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Saving predicted features
# ─────────────────────────────────────────────────────────────────────────────

def save_pred_pca(pred_feats, pca_vis, rgb_wh, out_dir):
    """
    Save PCA visualisations of predicted features.

    pred_feats : list of plan_horizon (gH, gW, D) numpy arrays
    rgb_wh     : (W, H) PIL size for resizing
    """
    from PIL import Image

    pred_dir = os.path.join(out_dir, "Pred")
    os.makedirs(pred_dir, exist_ok=True)

    for i, feat in enumerate(pred_feats):
        pca_rgb = pca_vis(feat)
        if pca_rgb.dtype != np.uint8:
            pca_rgb = (pca_rgb * 255).clip(0, 255).astype(np.uint8)
        pil_pca = Image.fromarray(pca_rgb)
        if rgb_wh is not None:
            pil_pca = pil_pca.resize(rgb_wh, Image.NEAREST)
        pil_pca.save(os.path.join(pred_dir, f"pred_step_{i + 1:02d}_pca.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _samps_to_TN4(samps, batch_idx, expected_T):
    """Robustly extract (T, N, 4) for one batch element from (B,T,N,4) or (T,B,N,4)."""
    if isinstance(samps, torch.Tensor):
        samps = samps.detach().cpu().numpy()
    samps = np.asarray(samps)
    if samps.ndim == 4 and samps.shape[-1] == 4:
        if samps.shape[1] == expected_T:        # (B, T, N, 4)
            return samps[batch_idx]
        if samps.shape[0] == expected_T:        # (T, B, N, 4)
            return samps[:, batch_idx]
        return samps[batch_idx]                 # fallback: assume B-first
    raise ValueError(f"Cannot interpret samples shape {samps.shape}")


def _samps_to_xy(samps_4d_TNK, ckpt_unnorm_stats, max_n=None):
    """
    Convert (T, N, 4) ckpt-normalised actions → list of (T+1, 2) XY arrays.
    max_n caps the number of trajectories returned for readability.
    """
    T, N, _ = samps_4d_TNK.shape
    N = min(N, max_n) if max_n is not None else N
    trajs = []
    for n in range(N):
        acts_un = unnormalize_bound_action(samps_4d_TNK[:, n, :], ckpt_unnorm_stats)
        xy, _ = compute_xy_trajectory_gt(acts_un)
        trajs.append(xy)
    return trajs


def _mean_to_xy(mean_T_Aint, ckpt_unnorm_stats, yaw_stats, unicycle_stats):
    """
    Convert internal-space mean (T, A_int) → (T+1, 2) metric XY trajectory.
    """
    mt = torch.tensor(mean_T_Aint[None], dtype=torch.float32)   # (1, T, A_int)
    if unicycle_stats is not None:
        mt = _unicycle_to_3d(mt, unicycle_stats)
    if yaw_stats is not None:
        mt = _expand_yaw_to_sincos(mt, yaw_stats)
    acts_un = unnormalize_bound_action(mt.numpy()[0], ckpt_unnorm_stats)
    xy, _ = compute_xy_trajectory_gt(acts_un)
    return xy


# ─────────────────────────────────────────────────────────────────────────────
# Plan subdir plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_trajectories(
    iter_info,
    batch_idx,
    best_acts_T4,
    gt_xy,
    ckpt_unnorm_stats,
    yaw_stats,
    unicycle_stats,
    out_dir,
    n_iters,
    max_samps_per_iter=64,
    formats=("pdf", "png"),     # pdf = vector (use this in the paper)
    panel_size=(3.2, 2.6),
    cmap_name="viridis",
):
    """
    Paper-ready CEM figures. Saves ONE file per panel, NO titles, into
    <out_dir>/Plan/paper/:

        iter_01.{pdf,png} .. iter_NN.{pdf,png}   per-iteration sample fans
                                                 (best plan + legend only on last)
        means_overlay.{pdf,png}                  means converging to GT (+ colorbar)
        spread.{pdf,png}                          endpoint sigma vs iteration

    Captions / sub-labels ("(a) Iteration 1", etc.) are left to LaTeX.
    All iteration panels share identical axis limits and decoration, so they
    are the same physical size and directly comparable in a subfigure row.
    """
    import os
    import numpy as np
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    # fonttype 42 => TrueType, not type-3 (IEEE/ACM reject type-3 fonts).
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 13, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    pdir = os.path.join(out_dir, "Plan", "paper")
    os.makedirs(pdir, exist_ok=True)

    iter_means     = iter_info[0]
    all_iter_samps = iter_info[3]
    CMAP   = mpl.colormaps[cmap_name]
    COLORS = [CMAP(i / max(n_iters - 1, 1)) for i in range(n_iters)]
    GT_C, BEST_C = "#00d000", "black"

    plan_horizon = best_acts_T4.shape[0]
    samps_xy = None
    if all_iter_samps is not None:
        samps_xy = [_samps_to_xy(_samps_to_TN4(all_iter_samps[k], batch_idx, plan_horizon),
                                 ckpt_unnorm_stats, max_n=max_samps_per_iter)
                    for k in range(n_iters)]
    means_xy = [_mean_to_xy(iter_means[k, batch_idx], ckpt_unnorm_stats,
                            yaw_stats, unicycle_stats) for k in range(n_iters)]
    best_acts_un = unnormalize_bound_action(best_acts_T4, ckpt_unnorm_stats)
    best_xy, _   = compute_xy_trajectory_gt(best_acts_un)

    # Shared limits: GT + best + means + the converged (last) fan only.
    ref = [gt_xy, best_xy] + means_xy + (samps_xy[-1] if samps_xy else [])
    ref = np.concatenate([p for p in ref], axis=0)
    xm = 0.06 * (np.ptp(ref[:, 0]) + 1e-6)
    y_abs_max = max(abs(ref[:, 1].min()), abs(ref[:, 1].max())) + 0.18 * (np.ptp(ref[:, 1]) + 1e-6)
    XLIM = (ref[:, 0].min() - xm, ref[:, 0].max() + xm)
    YLIM = (-y_abs_max, y_abs_max)

    def _frame(ax):
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.22, lw=0.6); ax.set_axisbelow(True)

    def _gt_start_goal(ax):
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color=GT_C, lw=2.6, zorder=11,
                linestyle=(0, (1, 1.8)), dash_capstyle="round")   # dotted: lets best plan show through
        ax.scatter([0], [0], c="black", s=50, marker="o", edgecolors="white", lw=1.0, zorder=12)
        ax.scatter([gt_xy[-1, 0]], [gt_xy[-1, 1]], c=GT_C, s=140, marker="*",
                   edgecolors="black", lw=0.6, zorder=12)

    def _best_plan(ax):
        # ax.plot(best_xy[:, 0], best_xy[:, 1], color="white", lw=4.0, zorder=9, alpha=0.9)
        ax.plot(best_xy[:, 0], best_xy[:, 1], color=BEST_C, lw=2.0, zorder=10, linestyle=(0, (6, 2)))

    def _save(fig, name):
        for ext in formats:
            fig.savefig(os.path.join(pdir, f"{name}.{ext}"), dpi=300)
        plt.close(fig)

    # Shared legend key (gray = generic; per-panel sample/mean color encodes the
    # iteration, which is documented by the colorbar in means_overlay + caption).
    legend_handles = [
        GradientLegendLine(
            label="CEM samples",
            cmap="viridis",
            lw=1.0,
            alpha=0.9,
        ),
            GradientLegendLine(
            label="CEM means",
            cmap="viridis",
            lw=2.4,
            alpha=1.0,
        ),
        Line2D([], [], color=BEST_C, lw=2.0, linestyle=(0, (6, 2)), label="Best plan"),
        Line2D([], [], color=GT_C, lw=2.6, linestyle=(0, (1, 1.8)),
               dash_capstyle="round", label="GT"),
        
        Line2D([], [], color="black", marker="o", lw=0, markersize=6,
               markeredgecolor="white", label="Start"),
        Line2D([], [], color=GT_C, marker="*", lw=0, markersize=11,
               markeredgecolor="black", label="Goal"),
    ]

    # 1. per-iteration panels (no titles). Best plan + legend only on the last.
    for k in range(n_iters):
        is_last = (k == n_iters - 1)
        fig, ax = plt.subplots(figsize=panel_size)
        if samps_xy is not None:
            ax.add_collection(LineCollection(samps_xy[k], colors=[COLORS[k]],
                                             linewidths=0.7, alpha=0.30))
        ax.plot(means_xy[k][:, 0], means_xy[k][:, 1], color=COLORS[k], lw=2.6, zorder=6, path_effects=[
            pe.Stroke(linewidth=3.2, foreground="0.15", alpha=0.8),
            pe.Normal(),
        ],)

        _gt_start_goal(ax)
        if is_last:
            _best_plan(ax)
        _frame(ax)
        if is_last:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
                      framealpha=0.9, handlelength=1.6, borderaxespad=0.4, ncol=2,
                      labelspacing=0.3, handler_map={
        GradientLegendLine: HandlerGradientLine(),
    },)
        _save(fig, f"iter_{k+1:02d}")

    # 2. means overlay + colorbar (no title) -- best plan kept here.
    fig, ax = plt.subplots(figsize=(panel_size[0] + 1.2, panel_size[1] + 0.6))
    for k in range(n_iters):
        f = k / max(n_iters - 1, 1)
        ax.plot(means_xy[k][:, 0], means_xy[k][:, 1], color=COLORS[k],
                lw=1.2 + 2.0 * f, alpha=0.5 + 0.5 * f, zorder=4 + k, solid_capstyle="round")
    _gt_start_goal(ax); _best_plan(ax); _frame(ax)
    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(1, n_iters))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("CEM iteration"); cb.set_ticks(np.arange(1, n_iters + 1))
    _save(fig, "means_overlay")

    # 3. spread vs iteration (no title)
    if samps_xy is not None:
        fig, ax = plt.subplots(figsize=panel_size)
        sig = [float(np.linalg.norm(np.array([t[-1] for t in samps_xy[k]]).std(0)))
               for k in range(n_iters)]
        ax.plot(np.arange(1, n_iters + 1), sig, "-", color="#444", lw=1.8, zorder=2)
        for k in range(n_iters):
            ax.plot(k + 1, sig[k], "o", color=COLORS[k], ms=8, zorder=5)
        ax.set_xlabel("CEM iteration"); ax.set_ylabel(r"Endpoint $\sigma$ (m)")
        ax.set_xticks(np.arange(1, n_iters + 1)); ax.grid(True, alpha=0.3)
        _save(fig, "spread")


def plot_cost_curve(
    iter_costs_b,
    out_dir,
    formats=("pdf", "png"),
    panel_size=(3.2, 2.6),
    cmap_name="viridis",
):
    """
    Paper-ready min-cost vs. CEM-iteration curve (no title), saved as vector
    PDF + PNG into <out_dir>/Plan/paper/cost_vs_iter.{pdf,png}.

    Markers are colored by iteration to match spread.pdf for a consistent look.

    iter_costs_b : (n_iters,) numpy array
    """
    import os
    import numpy as np
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 13, "axes.labelsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 0.9,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    pdir = os.path.join(out_dir, "Plan", "paper")
    os.makedirs(pdir, exist_ok=True)

    n_iters = len(iter_costs_b)
    iters   = np.arange(1, n_iters + 1)
    CMAP    = mpl.colormaps[cmap_name]
    COLORS  = [CMAP(i / max(n_iters - 1, 1)) for i in range(n_iters)]

    fig, ax = plt.subplots(figsize=panel_size)
    ax.plot(iters, iter_costs_b, "-", color="#444", lw=1.8, zorder=2)
    for k in range(n_iters):
        ax.plot(iters[k], iter_costs_b[k], "o", color=COLORS[k], ms=8, zorder=5)
    ax.set_xlabel("CEM iteration"); ax.set_ylabel("Min cost")
    ax.set_xticks(iters); ax.grid(True, alpha=0.3)

    for ext in formats:
        path = os.path.join(pdir, f"cost_vs_iter.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  cost curve    -> {path}")
    plt.close(fig)
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CEM planning visualiser — per-example GT/Pred/Plan plots"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dino",
                        choices=["dino", "dinov3", "vjepa"])
    parser.add_argument("--data_dir", default=None,
                        help="Root directory containing the RLDS dataset (default: cfg.data_dir)")
    parser.add_argument("--dataset_name", default="cast_filtered_dataset")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test_seen", "test_unseen", "test_all"])
    # ── Indices ──────────────────────────────────────────────────────────
    parser.add_argument("--indices", type=int, nargs="+", required=True,
                        help="Global 0-based example indices to visualise")
    # ── Horizon ──────────────────────────────────────────────────────────
    parser.add_argument("--plan_horizon",     type=int,   default=6)
    parser.add_argument("--seed_frames",      type=int,   default=4)
    parser.add_argument("--eval_window_size", type=int,   default=12,
                        help="Data loader window size (>= seed_frames + plan_horizon)")
    parser.add_argument("--ctxt_window",      type=int,   default=4)
    # ── CEM ──────────────────────────────────────────────────────────────
    parser.add_argument("--planner",          default="cem", choices=["cem", "mppi"])
    parser.add_argument("--cem_samples",      type=int,   default=128)
    parser.add_argument("--cem_iters",        type=int,   default=6)
    parser.add_argument("--cem_elites",       type=int,   default=4)
    parser.add_argument("--cem_var_scale",    type=float, default=2.0)
    parser.add_argument("--cem_momentum",     type=float, default=0.1)
    parser.add_argument("--mppi_temperature", type=float, default=2.0)
    parser.add_argument("--mppi_std_min",     type=float, default=0.1)
    parser.add_argument("--mppi_std_max",     type=float, default=2.0)
    parser.add_argument("--traj_weight",      type=float, default=0.1)
    parser.add_argument("--temporal_corr",    type=str,   default="0.0")
    # ── Metrics / cost ────────────────────────────────────────────────────
    parser.add_argument("--orientation_metric", default="vector",
                        choices=["vector", "yaw"])
    parser.add_argument("--gt_action_cost",   action="store_true")
    # ── Action space ─────────────────────────────────────────────────────
    parser.add_argument("--unicycle",         action="store_true")
    # ── Model ────────────────────────────────────────────────────────────
    parser.add_argument("--no_compile",       action="store_true")
    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",       type=int,   default=32,
                        help="Loader batch size for data collection")
    parser.add_argument("--seed",             type=int,   default=42)
    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output",           default="eval_results/planning_plots")
    args = parser.parse_args()

    # ── Seeds ─────────────────────────────────────────────────────────────
    import random
    import tensorflow as _tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _tf.random.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────
    CKPT_DIR    = os.path.dirname(args.checkpoint)
    config_path = os.path.join(CKPT_DIR, "config.py")
    spec        = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder)

    # ── Build model ───────────────────────────────────────────────────────
    print("Building model ...")
    model = build_world_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded checkpoint (step={ckpt.get('step', '?')})")

    if not args.no_compile:
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
        print("  Predictor compiled (reduce-overhead)")

    # ── Dataset stats ─────────────────────────────────────────────────────
    dataset_name = args.dataset_name
    data_dir = f"{args.data_dir or cfg.get('data_dir', './data')}/{dataset_name}"

    ckpt_stats = ckpt.get("dataset_statistics")
    if isinstance(ckpt_stats, dict) and dataset_name in ckpt_stats:
        stats = _to_numpy_tree(ckpt_stats)
        print(f"Using checkpoint stats for {dataset_name}")
    else:
        stats = None

    print("Loading GT normalization stats ...")
    std_fn = ModuleSpec.create(
        gnm_action_angle_dataset_transform,
        action_horizon=cfg.action_horizon,
    )
    _, _fresh = make_dataset_from_rlds(
        name=dataset_name, data_dir=data_dir, train=True,
        standardize_fn=std_fn,
        image_obs_keys={"primary": "image"}, proprio_obs_key="state",
        language_key="language_instruction",
        action_proprio_normalization_type=NormalizationType.BOUNDS,
        skip_norm=True,
    )
    gt_stats = {dataset_name: _fresh}
    if stats is None:
        stats = gt_stats


    split_name = args.split if args.split != "test_all" else "test_seen"
    loader = _build_test_loader(
        cfg, data_dir, split_name, dataset_name, gt_stats, collate_fn,
        max_windows_per_traj=None,
        batch_size=args.batch_size,
        window_size=args.eval_window_size,
    )

    # ── Collect requested examples ────────────────────────────────────────
    print(f"Collecting examples at indices: {sorted(args.indices)} ...")
    collected = collect_examples_at_indices(loader, args.indices)
    print(f"  Found {len(collected)} / {len(args.indices)} examples")
    if not collected:
        print("No examples collected. Check --indices and --split.")
        return

    # ── Derive action stats / priors ──────────────────────────────────────
    fallback_ds  = dataset_name
    _astats      = stats[fallback_ds]["action"]
    _p01_flat    = np.array(_astats["p01"]).flatten()
    _p99_flat    = np.array(_astats["p99"]).flatten()
    _raw_mean_flat = np.array(_astats["mean"]).flatten()
    _std_raw     = _astats.get("std", None)
    _raw_std_flat = (np.array(_std_raw).flatten() if _std_raw is not None
                     else (_p99_flat - _p01_flat) / 5.15)
    _has_per_dim = (len(_raw_mean_flat) >= 4 and len(_raw_std_flat) >= 4
                    and len(_p01_flat) >= 4)

    _mask_arr = _astats.get("mask", None)
    sincos_masked = False
    if _mask_arr is not None:
        _mask_b = np.array(_mask_arr).flatten().astype(bool)
        sincos_masked = len(_mask_b) >= 4 and not _mask_b[2] and not _mask_b[3]
    print(f"sin/cos masked = {sincos_masked}")

    # Normalised action prior mean
    if sincos_masked:
        _prior_norm = np.zeros(4, dtype=np.float32)
        _prior_norm[:2] = np.clip(
            2.0 * (_raw_mean_flat[:2] - _p01_flat[:2])
            / (_p99_flat[:2] - _p01_flat[:2] + 1e-8) - 1.0, -1.0, 1.0)
        _prior_norm[2:4] = _raw_mean_flat[2:4]
    else:
        _prior_norm = np.clip(
            2.0 * (_raw_mean_flat - _p01_flat) / (_p99_flat - _p01_flat + 1e-8) - 1.0,
            -1.0, 1.0,
        ).astype(np.float32)
    action_prior_mean = torch.tensor(_prior_norm.reshape(-1), device=device)

    # xy std in normalised space
    if _has_per_dim:
        _xy_std_norm = np.clip(
            2.0 * _raw_std_flat[:2] / (_p99_flat[:2] - _p01_flat[:2] + 1e-8), 0.05, 2.0)
    else:
        _xy_std_norm = np.array([0.4, 0.4])

    # Yaw std (delta method)
    if _has_per_dim:
        _ms = float(_raw_mean_flat[2])
        _mc = float(_raw_mean_flat[3])
        _Rsq = _ms**2 + _mc**2 + 1e-8
        _yaw_std_rad = float(np.sqrt(
            (_mc**2 * _raw_std_flat[2]**2 + _ms**2 * _raw_std_flat[3]**2) / _Rsq**2))
    elif len(_p01_flat) >= 3:
        _yaw_std_rad = float((_p99_flat[2] - _p01_flat[2]) / 5.15)
    else:
        _yaw_std_rad = 0.3
    _yaw_std_rad = float(np.clip(_yaw_std_rad, 0.05, np.pi))

    _prior_std_3d    = np.array([_xy_std_norm[0], _xy_std_norm[1], _yaw_std_rad], dtype=np.float32)
    action_prior_std = torch.tensor(_prior_std_3d * args.cem_var_scale, device=device)

    yaw_stats = (_yaw_sincos_stats(_p01_flat, _p99_flat, sincos_masked=sincos_masked)
                 if len(_p01_flat) >= 4 else None)
    if yaw_stats is None:
        action_prior_std = None

    # Unicycle stats
    unicycle_stats_dict = None
    if args.unicycle and yaw_stats is not None:
        _u_mean     = float(np.sqrt(_raw_mean_flat[0]**2 + _raw_mean_flat[1]**2))
        _yaw_mean   = float(np.arctan2(float(_raw_mean_flat[2]), float(_raw_mean_flat[3])))
        _u_std_base = float(np.sqrt(_raw_std_flat[0]**2 + _raw_std_flat[1]**2))
        _u_max      = float(np.sqrt(max(_p99_flat[0], 0.0)**2 + _p99_flat[1]**2))
        unicycle_stats_dict = dict(
            p01_x=float(_p01_flat[0]), p99_x=float(_p99_flat[0]),
            p01_y=float(_p01_flat[1]), p99_y=float(_p99_flat[1]),
            u_max=_u_max,
        )
        action_prior_mean = torch.tensor([_u_mean, _yaw_mean],
                                          dtype=torch.float32, device=device)
        action_prior_std  = torch.tensor([_u_std_base, _yaw_std_rad],
                                          dtype=torch.float32, device=device) * args.cem_var_scale
        print(f"  unicycle ON: u_max={_u_max:.3f}  u_mean={_u_mean:.4f}  "
              f"yaw_mean={_yaw_mean:.4f} rad")
    elif args.unicycle:
        print("  Warning: --unicycle requires 4D action stats; unicycle disabled.")

    ckpt_unnorm_stats = stats[fallback_ds]["action"]

    # Temporal correlation
    _tc = str(args.temporal_corr).strip()
    temporal_corr = ([float(v) for v in _tc.split(",")]
                     if "," in _tc else float(_tc))

    # ── Stack examples into a planning batch ──────────────────────────────
    sorted_keys = sorted(collected.keys())
    B = len(sorted_keys)
    print(f"\nProcessing {B} examples ...")

    images_B_cpu = torch.stack([collected[k]["images"] for k in sorted_keys])  # (B, T_win, C, H, W)
    actions_B    = torch.stack([collected[k]["actions"] for k in sorted_keys])  # (B, T_win, A)

    seed_frames  = args.seed_frames
    plan_horizon = args.plan_horizon
    n_frames     = seed_frames + plan_horizon            # frames to visualise (e.g. 10)
    goal_idx_win = seed_frames + plan_horizon - 1        # goal frame index in window

    # ── Encode seed + goal frames ─────────────────────────────────────────
    print("Encoding frames ...")
    seed_idxs    = list(range(seed_frames))
    all_idxs     = list(range(n_frames))                 # seed + horizon frames
    sg_idxs      = seed_idxs + [goal_idx_win]            # for z_seed + goal

    images_all_gpu = images_B_cpu[:, all_idxs].to(device)  # (B, n_frames, C, H, W)
    images_sg_gpu  = images_B_cpu[:, sg_idxs].to(device)   # (B, S+1, C, H, W)

    with torch.no_grad():
        z_all = model.encode_obs(images_all_gpu)  # (B, n_frames, 1, gH, gW, D)
        zg    = model.encode_obs(images_sg_gpu)   # (B, S+1,      1, gH, gW, D)

    z_seed_B = zg[:, :seed_frames].float()                          # (B, S, 1, gH, gW, D)
    goal_B   = zg[:, seed_frames:seed_frames + 1].float()           # (B, 1, 1, gH, gW, D)
    z_all_cpu = z_all[:, :, 0].float().cpu()                        # (B, n_frames, gH, gW, D)

    # ── Language embeddings ───────────────────────────────────────────────
    lang_B = None   # paper models are image-goal only

    # ── Seed and GT actions ───────────────────────────────────────────────
    gt_key_fb = list(gt_stats.keys())[0]

    seed_acts_B = None
    if seed_frames > 1:
        seed_acts_raw = actions_B[:, :seed_frames - 1].to(device)   # (B, S-1, A)
        seed_acts_B   = _renorm_gt_to_ckpt(
            seed_acts_raw,
            gt_stats[gt_key_fb]["action"],
            stats[fallback_ds]["action"],
            device,
        )   # (B, S-1, A)

    act_start      = seed_frames - 1
    gt_acts_raw    = actions_B[:, act_start:act_start + plan_horizon].to(device)   # (B, T, A)
    gt_acts_ckpt_B = _renorm_gt_to_ckpt(
        gt_acts_raw,
        gt_stats[gt_key_fb]["action"],
        stats[fallback_ds]["action"],
        device,
    )   # (B, T, A)

    # ── CEM planning ──────────────────────────────────────────────────────
    print(f"Running CEM ({args.cem_iters} iters × {args.cem_samples} samples × {B} examples) ...")
    best_acts, final_samps, best_costs, iter_info = cem_plan_batch(
        model, z_seed_B, goal_B, lang_B,
        plan_horizon   = plan_horizon,
        n_samples      = args.cem_samples,
        n_iters        = args.cem_iters,
        n_elites       = args.cem_elites,
        var_scale      = args.cem_var_scale,
        momentum       = args.cem_momentum,
        cost_fn        = None,
        ctxt_window    = args.ctxt_window,
        device         = device,
        action_prior_mean = action_prior_mean,
        action_prior_std  = action_prior_std,
        seed_acts_B    = seed_acts_B,
        yaw_stats      = yaw_stats,
        traj_weight    = args.traj_weight,
        temporal_corr  = temporal_corr,
        gt_action_sample_B = gt_acts_ckpt_B if args.gt_action_cost else None,
        gt_action_cost_B   = gt_acts_ckpt_B if args.gt_action_cost else None,
        return_samples = True,
        return_iter_info = True,
        unicycle_stats = unicycle_stats_dict,
    )
    # best_acts : (B, T, 4) numpy
    # iter_info : (iter_means, iter_costs, first_samps, all_iter_samps_list)

    # ── Per-example output ────────────────────────────────────────────────
    print(f"\nSaving outputs → {args.output}/")
    os.makedirs(args.output, exist_ok=True)

    for bi, global_idx in enumerate(sorted_keys):
        ex = collected[global_idx]
        print(f"\n[{bi + 1}/{B}] example_{global_idx:04d} — '{ex['language'][:70]}'")
        out_dir = os.path.join(args.output, f"example_{global_idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        # ── Roll out model with planned actions (for Pred/) ───────────────
        sa_bi = (seed_acts_B[bi:bi + 1] if seed_acts_B is not None else None)
        lang_bi = (lang_B[bi:bi + 1] if lang_B is not None else None)
        pred_feats = rollout_step_by_step(
            model,
            z_seed_B[bi:bi + 1],
            best_acts[bi],         # (T, 4)
            sa_bi,
            lang_bi,
            args.ctxt_window,
            device,
        )   # list of plan_horizon (gH, gW, D) arrays

        # ── Fit joint PCA (GT frames + predicted features) ────────────────
        gt_feats_np = z_all_cpu[bi, :n_frames].numpy()    # (n_frames, gH, gW, D)
        D_feat = gt_feats_np.shape[-1]
        all_patches = np.concatenate(
            [gt_feats_np.reshape(-1, D_feat)]
            + [f.reshape(-1, D_feat) for f in pred_feats],
            axis=0,
        )
        if np.all(np.isfinite(all_patches)):
            pca_vis = PCAVisualizer(all_patches)
        else:
            # Fallback: fit on GT only (pred may have NaNs)
            pca_vis = PCAVisualizer(gt_feats_np.reshape(-1, D_feat))

        # RGB size for resizing PCA images
        rgb0   = _hwc_uint8(images_B_cpu[bi, 0])
        rgb_wh = (rgb0.shape[1], rgb0.shape[0])            # PIL (W, H)

        # ── GT/ ───────────────────────────────────────────────────────────
        save_gt_frames(
            images_B_cpu[bi, :n_frames],                   # (n_frames, C, H, W)
            z_all_cpu[bi, :n_frames].unsqueeze(1),         # (n_frames, 1, gH, gW, D)
            pca_vis, n_frames, out_dir,
        )
        print(f"  GT/ saved ({n_frames} frames)")

        # ── Pred/ ─────────────────────────────────────────────────────────
        save_pred_pca(pred_feats, pca_vis, rgb_wh, out_dir)
        print(f"  Pred/ saved ({plan_horizon} steps)")

        # ── GT XY trajectory ──────────────────────────────────────────────
        gt_acts_unnorm = unnormalize_bound_action(
            gt_acts_ckpt_B[bi].cpu().numpy(), ckpt_unnorm_stats)
        gt_xy, _ = compute_xy_trajectory_gt(gt_acts_unnorm)

        # ── Plan/ ─────────────────────────────────────────────────────────
        if iter_info is not None:
            plot_trajectories(
                iter_info, bi,
                best_acts[bi], gt_xy,
                ckpt_unnorm_stats, yaw_stats, unicycle_stats_dict,
                out_dir, n_iters=args.cem_iters,
            )
            plot_cost_curve(iter_info[1][:, bi], out_dir)
        else:
            print("  Warning: iter_info is None — Plan/ plots skipped.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
