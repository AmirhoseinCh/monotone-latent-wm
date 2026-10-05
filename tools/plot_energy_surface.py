#!/usr/bin/env python3
"""
Plot the energy (cost) surface of the world model around the GT trajectory.

For a single navigation example:
  1. Load model + one batch of data.
  2. Encode the seed frame(s) and the goal frame.
  3. Sweep actions on a 2-D grid of perturbations around GT:
         d_x_norm  ∈ [-range_x,   +range_x]   (forward-speed offset, normalised)
         d_yaw_rad ∈ [-range_yaw, +range_yaw]  (heading offset, radians)
     added uniformly to every time-step of the GT action sequence.
  4. For each candidate, roll out the WM → predicted goal features → compute:
         cosine cost = 1 - cos_sim(pred, goal_features)
         L2 cost     = MSE(pred, goal_features)
         combined    = cosine + L2
  5. Dead-reckon the perturbed actions to get the final XY endpoint.
  6. 3-D plot: x=x (m), y=y (m), z=cost, GT trajectory overlaid.

Usage:
    python tools/plot_energy_surface.py \\
        --checkpoint /path/to/best_model.pt \\
        --encoder dino \\
        [--n_x 40] [--n_yaw 40] \\
        [--range_x 0.8] [--range_yaw 1.2] \\
        [--seed_frames 2] [--plan_horizon 4] \\
        [--example_idx 0] \\
        [--output plots/energy_surface.png]
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from eval_planning import (
    compute_xy_trajectory_gt,
    compute_xy_trajectory_batch,
    unnormalize_bound_action,
    cem_unroll,
    _yaw_sincos_stats,
    _expand_yaw_to_sincos,
    _sincos_to_yaw,
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


def _to_numpy_tree(x):
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@torch.no_grad()
def compute_energy_surface(
    model,
    z_seed,        # (1, S, 1, gH, gW, D) — already on device
    goal,          # (1, 1, 1, gH, gW, D)
    lang_emb,      # (1, D_lang) or None
    gt_acts_ckpt,  # (T, 4) numpy — ckpt-normalised [-1, 1]
    ckpt_unnorm_stats,  # ds_stats[key]["action"] — for perturbed-action unnorm
    yaw_stats,     # dict or None — sin/cos ↔ yaw conversion constants
    ctxt_window,
    n_x,           # grid resolution along d_x axis
    n_yaw,         # grid resolution along d_yaw axis
    range_x,       # max |d_x| in normalised x units
    range_yaw,     # max |d_yaw| in radians
    device,
    seed_acts=None,    # (1, S-1, A) GT seed actions on device, or None
    rollout_batch=512, # candidates per model forward pass
):
    """
    Sweep (d_x_norm, d_yaw_rad) perturbations around the GT action sequence,
    roll out the WM, and return costs and XY endpoints for every grid point.

    Returns
    -------
    d_x_grid          : (n_x, n_yaw) ndarray of d_x values
    d_yaw_grid        : (n_x, n_yaw) ndarray of d_yaw values
    xy_all            : (n_x*n_yaw, T+1, 2) metric XY trajectory for each candidate
    costs             : dict[str -> (n_x, n_yaw)] — "cosine", "l2", "combined"
    gt_xy             : (T+1, 2) metric GT trajectory
    gt_costs          : dict[str -> float] — GT action costs under each metric
    feats             : dict with keys "seed" (gH,gW,D), "goal_gt" (gH,gW,D),
                        "goal_pred_best" (gH,gW,D) — best predicted = lowest combined cost
    """
    T = gt_acts_ckpt.shape[0]
    N_total = n_x * n_yaw

    # 2-D perturbation grid
    d_x_vals   = np.linspace(-range_x,   range_x,   n_x)
    d_yaw_vals = np.linspace(-range_yaw, range_yaw, n_yaw)
    d_x_grid, d_yaw_grid = np.meshgrid(d_x_vals, d_yaw_vals, indexing="ij")
    d_x_flat   = d_x_grid.reshape(-1).astype(np.float32)
    d_yaw_flat = d_yaw_grid.reshape(-1).astype(np.float32)

    # Convert GT ckpt-normalised actions to internal space for perturbation
    gt_t = torch.tensor(gt_acts_ckpt, dtype=torch.float32, device=device)  # (T, 4)
    if yaw_stats is not None:
        gt_internal = _sincos_to_yaw(gt_t, yaw_stats)  # (T, 3): [x_norm, y_norm, yaw_rad]
    else:
        gt_internal = gt_t  # (T, 4)

    # Build (N_total, T, A_internal) candidate array
    gt_rep = gt_internal.unsqueeze(0).expand(N_total, -1, -1).clone()  # (N, T, A)
    d_x_t   = torch.tensor(d_x_flat,   device=device)
    d_yaw_t = torch.tensor(d_yaw_flat, device=device)

    gt_rep[:, :, 0] = (gt_rep[:, :, 0] + d_x_t.unsqueeze(1)).clamp(-1., 1.)  # x_norm
    if yaw_stats is not None:
        gt_rep[:, :, 2] = (gt_rep[:, :, 2] + d_yaw_t.unsqueeze(1)).clamp(-np.pi, np.pi)
    else:
        # No yaw_stats: perturb sin/cos directly via angle arithmetic
        yaw_cur = torch.atan2(gt_rep[:, :, 2], gt_rep[:, :, 3])
        yaw_new = yaw_cur + d_yaw_t.unsqueeze(1)
        gt_rep[:, :, 2] = torch.sin(yaw_new)
        gt_rep[:, :, 3] = torch.cos(yaw_new)

    # Project to 4D ckpt-normalised for the WM
    if yaw_stats is not None:
        acts_4d_all = _expand_yaw_to_sincos(gt_rep, yaw_stats)  # (N_total, T, 4)
    else:
        acts_4d_all = gt_rep  # (N_total, T, 4)

    # XY trajectories in metric space (unnorm the 4D ckpt-normalised actions)
    acts_4d_np = acts_4d_all.cpu().numpy()  # (N_total, T, 4)
    acts_unnorm_all = np.stack(
        [unnormalize_bound_action(acts_4d_np[i], ckpt_unnorm_stats) for i in range(N_total)]
    )  # (N_total, T, 4)
    xy_all_trajs, _ = compute_xy_trajectory_batch(acts_unnorm_all)  # (N_total, T+1, 2)

    # GT trajectory in metric space
    gt_acts_unnorm = unnormalize_bound_action(gt_acts_ckpt, ckpt_unnorm_stats)  # (T, 4)
    gt_xy, _ = compute_xy_trajectory_gt(gt_acts_unnorm)  # (T+1, 2)

    # WM rollout in mini-batches
    cosine_costs = np.zeros(N_total, dtype=np.float32)
    l2_costs     = np.zeros(N_total, dtype=np.float32)

    seed_acts_perm = None
    if seed_acts is not None and seed_acts.shape[1] > 0:
        # (1, S-1, A) → (S-1, 1, A) to match cem_unroll's seed_acts_SNA: (S-1, N, A)
        seed_acts_perm = seed_acts.permute(1, 0, 2)  # (S-1, 1, A)

    n_batches = (N_total + rollout_batch - 1) // rollout_batch
    print(f"  Rolling out {N_total} candidates ({n_batches} mini-batches of {rollout_batch})...")
    for b in range(n_batches):
        start = b * rollout_batch
        end   = min(start + rollout_batch, N_total)
        N_b   = end - start

        acts_b      = acts_4d_all[start:end]              # (N_b, T, 4)
        acts_T_N    = acts_b.permute(1, 0, 2).to(torch.bfloat16)  # (T, N_b, 4)
        z_seed_b    = z_seed.expand(N_b, -1, -1, -1, -1, -1).contiguous().to(torch.bfloat16)
        goal_b      = goal.expand(N_b,   -1, -1, -1, -1, -1).contiguous().to(torch.bfloat16)
        lang_b      = lang_emb.expand(N_b, -1).contiguous().to(torch.bfloat16) \
                      if lang_emb is not None else None
        sa_b        = seed_acts_perm.expand(-1, N_b, -1).contiguous().to(torch.bfloat16) \
                      if seed_acts_perm is not None else None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_final, _ = cem_unroll(
                model, z_seed_b, acts_T_N, lang_b, ctxt_window,
                seed_acts_SNA=sa_b,
            )  # (N_b, 1, 1, gH, gW, D)

        pred_flat = pred_final.flatten(1).float()       # (N_b, *)
        goal_flat = goal_b.flatten(1).float()           # (N_b, *)

        cosine_costs[start:end] = (
            1.0 - F.cosine_similarity(pred_flat, goal_flat, dim=1)
        ).cpu().numpy()
        l2_costs[start:end] = (
            (pred_flat - goal_flat) ** 2
        ).mean(dim=1).cpu().numpy()

        if (b + 1) % max(1, n_batches // 5) == 0 or b == n_batches - 1:
            print(f"    batch {b+1}/{n_batches} done")

    combined_costs = cosine_costs + l2_costs

    # GT cost + GT predicted features: rollout with unperturbed GT actions
    gt_t4 = torch.tensor(gt_acts_ckpt[None], dtype=torch.float32, device=device)  # (1, T, 4)
    sa_gt = seed_acts_perm.expand(-1, 1, -1).contiguous().to(torch.bfloat16) \
            if seed_acts_perm is not None else None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_gt, _ = cem_unroll(
            model,
            z_seed.to(torch.bfloat16),
            gt_t4.permute(1, 0, 2).to(torch.bfloat16),  # (T, 1, 4)
            lang_emb.to(torch.bfloat16) if lang_emb is not None else None,
            ctxt_window,
            seed_acts_SNA=sa_gt,
        )
    gt_pred_flat = pred_gt.flatten(1).float()
    gt_goal_flat = goal.flatten(1).float()
    gt_cosine = float((1.0 - F.cosine_similarity(gt_pred_flat, gt_goal_flat, dim=1)).cpu())
    gt_l2     = float(((gt_pred_flat - gt_goal_flat) ** 2).mean(dim=1).cpu())

    # Best candidate (lowest combined cost): roll out one more time to get features
    best_idx  = int(np.argmin(combined_costs))
    best_act4 = acts_4d_all[best_idx:best_idx+1]  # (1, T, 4)
    sa_best   = seed_acts_perm.expand(-1, 1, -1).contiguous().to(torch.bfloat16) \
                if seed_acts_perm is not None else None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_best, _ = cem_unroll(
            model,
            z_seed.to(torch.bfloat16),
            best_act4.permute(1, 0, 2).to(torch.bfloat16),
            lang_emb.to(torch.bfloat16) if lang_emb is not None else None,
            ctxt_window,
            seed_acts_SNA=sa_best,
        )
    feat_goal_pred_best = pred_best[0, 0, 0].cpu().float().numpy()  # (gH, gW, D)

    costs = {
        "cosine":   cosine_costs.reshape(n_x, n_yaw),
        "l2":       l2_costs.reshape(n_x, n_yaw),
        "combined": combined_costs.reshape(n_x, n_yaw),
    }
    gt_costs = {"cosine": gt_cosine, "l2": gt_l2, "combined": gt_cosine + gt_l2}

    feats = {
        "seed":            z_seed[0, -1, 0].cpu().float().numpy(),  # (gH, gW, D) last seed frame
        "goal_gt":         goal[0, 0, 0].cpu().float().numpy(),     # (gH, gW, D)
        "goal_pred_best":  feat_goal_pred_best,                     # (gH, gW, D)
    }

    return d_x_grid, d_yaw_grid, xy_all_trajs, costs, gt_xy, gt_acts_unnorm, gt_costs, feats


# ─────────────────────────────────────────────────────────────────────────────
# Paper-figure styling (kept identical to tools/plot_planning.py)
# ─────────────────────────────────────────────────────────────────────────────
# fonttype 42 => TrueType, not type-3 (IEEE/ACM reject type-3 fonts).
_PAPER_RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 13, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 0.9,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


def _apply_paper_style():
    """Apply the same matplotlib rcParams the planning plots use, so the
    energy-surface figures share fonts / line-weights / vector-embedding."""
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update(_PAPER_RC)


def plot_energy_surface(
    d_x_grid, d_yaw_grid,
    xy_all_trajs,   # (N_total, T+1, 2)
    costs,          # dict[str -> (n_x, n_yaw)]
    gt_xy,          # (T+1, 2)
    gt_acts_unnorm, # (T, 4) metric — for GT trajectory heading arrows
    gt_costs,       # dict[str -> float]
    language,
    output_dir=None,
    formats=("pdf", "png"),   # pdf = vector (use this in the paper)
    figsize=(5.6, 4.1),
):
    """3-D energy surface — one paper-ready figure per cost type (vector PDF +
    PNG) saved to output_dir/3D/.  No titles: captions are left to LaTeX."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import griddata

    _apply_paper_style()
    # Font sizes mirror plot_planning.py (axes.labelsize 15, ticks 12).
    _FS, _FS_T, _FS_L, _FS_C = 15, 12, 8.5, 13  # label, tick, legend, colorbar

    n_x, n_yaw = d_x_grid.shape
    xy_endpoints = xy_all_trajs[:, -1, :]
    x_pts = xy_endpoints[:, 0]
    y_pts = xy_endpoints[:, 1]

    cost_keys = ["cosine", "l2", "combined"]

    margin = 0.08
    x_min, x_max = x_pts.min(), x_pts.max()
    y_min, y_max = y_pts.min(), y_pts.max()
    xr, yr = max(x_max - x_min, 1e-3), max(y_max - y_min, 1e-3)
    xi = np.linspace(x_min - margin * xr, x_max + margin * xr, 80)
    yi = np.linspace(y_min - margin * yr, y_max + margin * yr, 80)
    Xi, Yi = np.meshgrid(xi, yi)

    for key in cost_keys:
        cost_flat = costs[key].reshape(-1)
        gt_cost_val = gt_costs[key]

        p98 = np.percentile(cost_flat, 98)
        cost_clip = cost_flat.clip(max=p98)

        Zi = griddata((x_pts, y_pts), cost_clip, (Xi, Yi), method="cubic")
        Zi_nn = griddata((x_pts, y_pts), cost_clip, (Xi, Yi), method="nearest")
        Zi = np.where(np.isnan(Zi), Zi_nn, Zi)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.01, 0.05, 0.79, 0.88], projection="3d")

        surf = ax.plot_surface(
            Xi, Yi, Zi, cmap="plasma_r", alpha=0.70,
            linewidth=0, antialiased=True, vmin=cost_clip.min(), vmax=p98,
        )
        ax.scatter(
            x_pts, y_pts, cost_clip,
            c=cost_clip, cmap="plasma_r",
            vmin=cost_clip.min(), vmax=p98,
            s=6, alpha=0.35, zorder=3,
        )

        gt_z = min(gt_cost_val, p98)
        ax.plot(
            gt_xy[:, 0], gt_xy[:, 1],
            np.full(len(gt_xy), gt_z),
            color="lime", linewidth=2.5, zorder=6, label="GT trajectory",
        )
        ax.plot(
            gt_xy[:, 0], gt_xy[:, 1],
            np.linspace(Zi.min(), gt_z, len(gt_xy)),
            color="lime", linewidth=0.8, linestyle="--", alpha=0.7, zorder=5,
        )
        ax.scatter(
            gt_xy[1:, 0], gt_xy[1:, 1],
            np.full(len(gt_xy) - 1, gt_z),
            c="lime", s=40, marker="o", zorder=7,
        )
        ax.scatter(
            [gt_xy[-1, 0]], [gt_xy[-1, 1]], [gt_z],
            c="lime", s=150, marker="*", zorder=8, label=f"GT endpoint\ncost={gt_cost_val:.2f}",
        )
        ax.scatter([0], [0], [gt_z], c="black", s=80, marker="o", edgecolors="white",
                   linewidths=1.0, zorder=9, label="Start")

        ax.set_xlabel("x (m)", fontsize=_FS, labelpad=4, loc="left")
        ax.set_ylabel("y (m)", fontsize=_FS, labelpad=4)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("cost", fontsize=_FS, labelpad=4, rotation=90)
        ax.tick_params(labelsize=_FS_T, pad=1)
        ax.zaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax.legend(fontsize=_FS_L, loc="upper left", framealpha=0.6)
        ax.view_init(elev=30, azim=-60)

        cax = fig.add_axes([0.84, 0.33, 0.018, 0.36])
        cb = fig.colorbar(surf, cax=cax)
        cb.set_label("cost", fontsize=_FS_C)
        cb.ax.tick_params(labelsize=_FS_T)
        cb.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.1f'))

        if output_dir:
            out = os.path.join(output_dir, "3D")
            os.makedirs(out, exist_ok=True)
            for ext in formats:
                path = os.path.join(out, f"{key}_3d.{ext}")
                fig.savefig(path, dpi=300)
                print(f"  3-D surface saved → {path}")
        else:
            plt.show()
        plt.close(fig)


def plot_energy_surface_2d(
    d_x_grid, d_yaw_grid,
    xy_all_trajs,
    costs,
    gt_xy,
    gt_acts_unnorm,
    gt_costs,
    language,
    output_dir=None,
    formats=("pdf", "png"),   # pdf = vector (use this in the paper)
    figsize=(4.4, 3.2),
):
    """2-D top-down views — paper-ready fan and contour (vector PDF + PNG)
    saved separately to output_dir/2D/.  No titles: captions left to LaTeX."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    _apply_paper_style()
    # Font sizes mirror plot_planning.py (axes.labelsize 15, ticks 12).
    _FS, _FS_T, _FS_L, _FS_C = 15, 12, 8.5, 13  # label, tick, legend, colorbar

    n_x, n_yaw = d_x_grid.shape
    N_total = n_x * n_yaw
    cost_keys = ["cosine", "l2", "combined"]

    if output_dir:
        out2d = os.path.join(output_dir, "2D")
        os.makedirs(out2d, exist_ok=True)

    for key in cost_keys:
        cost_flat = costs[key].reshape(-1)
        gt_cost_val = gt_costs[key]
        p98 = np.percentile(cost_flat, 98)
        cost_clip = cost_flat.clip(max=p98)
        cmap = mpl.colormaps["plasma_r"]
        norm = plt.Normalize(vmin=cost_clip.min(), vmax=p98)
        xy_ends = xy_all_trajs[:, -1, :]

        # ── Trajectory fan ────────────────────────────────────────────────
        fig0, ax0 = plt.subplots(figsize=figsize)
        for i in range(N_total):
            traj  = xy_all_trajs[i]
            c_val = float(cost_clip[i])
            ax0.plot(traj[:, 0], traj[:, 1],
                     color=cmap(norm(c_val)), alpha=0.3, linewidth=0.6)
        ax0.plot(gt_xy[:, 0], gt_xy[:, 1], "g-o", linewidth=2.5, markersize=5,
                 zorder=5, label="GT")
        ax0.scatter([0], [0], c="black", s=80, marker="o", edgecolors="white",
                    linewidths=1.0, zorder=6, label="Start")
        ax0.scatter([gt_xy[-1, 0]], [gt_xy[-1, 1]], c="lime", s=120, marker="*",
                    zorder=6, label="Goal")
        sm = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
        sm.set_array([])
        cb0 = fig0.colorbar(sm, ax=ax0, fraction=0.046, pad=0.08, shrink=0.75)
        cb0.set_label("cost", fontsize=_FS_C)
        cb0.ax.tick_params(labelsize=_FS_T)
        cb0.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax0.set_aspect("equal")
        ax0.grid(True, alpha=0.3)
        ax0.set_xlabel("x (m)", fontsize=_FS)
        ax0.set_ylabel("y (m)", fontsize=_FS)
        ax0.tick_params(labelsize=_FS_T)
        ax0.legend(fontsize=_FS_L, loc="upper left")
        if output_dir:
            for ext in formats:
                path = os.path.join(out2d, f"{key}_fan.{ext}")
                fig0.savefig(path, dpi=300)
                print(f"  2-D fan saved → {path}")
        else:
            plt.show()
        plt.close(fig0)

        # ── Contour map of endpoint costs ─────────────────────────────────
        x_pts = xy_ends[:, 0]
        y_pts = xy_ends[:, 1]
        xr    = max(x_pts.max() - x_pts.min(), 1e-3)
        yr    = max(y_pts.max() - y_pts.min(), 1e-3)
        mg    = 0.06
        xi    = np.linspace(x_pts.min() - mg*xr, x_pts.max() + mg*xr, 80)
        yi    = np.linspace(y_pts.min() - mg*yr, y_pts.max() + mg*yr, 80)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi     = griddata((x_pts, y_pts), cost_clip, (Xi, Yi), method="cubic")
        Zi_nn  = griddata((x_pts, y_pts), cost_clip, (Xi, Yi), method="nearest")
        Zi     = np.where(np.isnan(Zi), Zi_nn, Zi)

        fig1, ax1 = plt.subplots(figsize=figsize)
        cf = ax1.contourf(Xi, Yi, Zi, levels=30, cmap="plasma_r",
                          vmin=cost_clip.min(), vmax=p98)
        ax1.contour(Xi, Yi, Zi, levels=10, colors="white", alpha=0.25, linewidths=0.5)
        cb1 = fig1.colorbar(cf, ax=ax1, fraction=0.046, pad=0.08, shrink=0.75)
        cb1.set_label("cost", fontsize=_FS_C)
        cb1.ax.tick_params(labelsize=_FS_T)
        cb1.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax1.scatter(x_pts, y_pts, c=cost_clip, cmap="plasma_r",
                    vmin=cost_clip.min(), vmax=p98, s=8, alpha=0.5, zorder=3)
        ax1.plot(gt_xy[:, 0], gt_xy[:, 1], "g-o", linewidth=2.5, markersize=5,
                 zorder=5, label="GT")
        ax1.scatter([0], [0], c="black", s=80, marker="o", edgecolors="white",
                    linewidths=1.0, zorder=6, label="Start")
        ax1.scatter([gt_xy[-1, 0]], [gt_xy[-1, 1]], c="lime", s=120, marker="*",
                    zorder=6, label=f"Goal cost={gt_cost_val:.2f}")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.2)
        ax1.set_xlabel("x (m)", fontsize=_FS)
        ax1.set_ylabel("y (m)", fontsize=_FS)
        ax1.tick_params(labelsize=_FS_T)
        ax1.legend(fontsize=_FS_L, loc="upper left")
        if output_dir:
            for ext in formats:
                path = os.path.join(out2d, f"{key}_contour.{ext}")
                fig1.savefig(path, dpi=300)
                print(f"  2-D contour saved → {path}")
        else:
            plt.show()
        plt.close(fig1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot WM energy surface around GT trajectory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", default="dino",
                        choices=["dino", "dinov3", "vjepa"])
    parser.add_argument("--data_dir", default=None,
                        help="Root directory containing the RLDS dataset (default: cfg.data_dir)")
    parser.add_argument("--dataset_name", default="cast_filtered_dataset")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test_seen", "test_unseen", "test_all"])
    parser.add_argument("--plan_horizon", type=int, default=4)
    parser.add_argument("--seed_frames", type=int, default=2)
    parser.add_argument("--ctxt_window", type=int, default=None)
    parser.add_argument("--example_idx", type=int, default=0,
                        help="Which example in the first batch to visualise (0-indexed)")
    # Sweep grid
    parser.add_argument("--n_x",   type=int,   default=20, help="Grid resolution: forward direction")
    parser.add_argument("--n_yaw", type=int,   default=20, help="Grid resolution: heading direction")
    parser.add_argument("--range_x",   type=float, default=0.4,
                        help="Max |d_x| perturbation in normalised action space (default 0.4 ≈ 40%% of range)")
    parser.add_argument("--range_yaw", type=float, default=0.6,
                        help="Max |d_yaw| perturbation in radians (default 0.6 ≈ 34°)")
    parser.add_argument("--rollout_batch", type=int, default=512,
                        help="Candidates per WM forward pass — reduce if OOM")
    # Output
    parser.add_argument("--output", default="eval_results/energy_surface",
                        help="Output directory; plots are saved under RGB/, PCA/, 3D/, 2D/ subdirs")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    print(f"Loading config from {config_path} ...")
    spec        = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder)

    eval_window_size = args.seed_frames + args.plan_horizon
    ctxt_window = args.ctxt_window if args.ctxt_window is not None else cfg.ctxt_window
    print(f"Context window: {ctxt_window} | Window size: {eval_window_size}")

    # ── Build model ───────────────────────────────────────────────────────
    print("Building model ...")
    model = build_world_model(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  Loaded checkpoint (step={ckpt.get('step','?')})")

    if not args.no_compile:
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
        print("  Predictor compiled (reduce-overhead)")

    # ── Dataset / stats ───────────────────────────────────────────────────
    dataset_name = args.dataset_name
    data_dir = f"{args.data_dir or cfg.get('data_dir', './data')}/{dataset_name}"

    ckpt_stats = ckpt.get("dataset_statistics")
    if isinstance(ckpt_stats, dict) and dataset_name in ckpt_stats:
        stats = _to_numpy_tree(ckpt_stats)
        print(f"Using checkpoint stats for {dataset_name}")
    else:
        stats = None

    print("Loading fresh GT normalization stats ...")
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
        max_windows_per_traj=None, batch_size=max(args.example_idx + 1, 4),
        window_size=eval_window_size,
    )

    # Grab one batch
    print(f"Fetching one batch ...")
    batch = next(iter(loader))

    # ── Extract example ───────────────────────────────────────────────────
    b = args.example_idx
    images  = batch["observation"]["image_primary"].to(device)   # (B, T_win, C, H, W)
    actions = batch["action"]
    if actions.dim() == 4:
        actions = actions[:, :, 0, :]  # (B, T_win, A)

    if b >= images.shape[0]:
        raise ValueError(f"example_idx={b} >= batch size {images.shape[0]}")
    if images.shape[1] < eval_window_size:
        raise ValueError(f"Window too short: {images.shape[1]} < {eval_window_size}")

    raw_language  = _normalize_raw_language(batch.get("raw_language", ["navigate"]))
    language      = raw_language[b] if b < len(raw_language) else "navigate"
    fallback_ds   = list(stats.keys())[0]
    ds_names_raw  = batch.get("dataset_name", [None])
    ds_key = (ds_names_raw[b].decode("utf-8") if isinstance(ds_names_raw[b], bytes)
              else str(ds_names_raw[b])) if b < len(ds_names_raw) else fallback_ds
    ds_key = ds_key if ds_key in stats else fallback_ds
    gt_key = ds_key if ds_key in gt_stats else list(gt_stats.keys())[0]

    # Detect sin/cos masking
    _astats    = stats[fallback_ds]["action"]
    _p01_flat  = np.array(_astats["p01"]).flatten()
    _p99_flat  = np.array(_astats["p99"]).flatten()
    _mask_arr  = _astats.get("mask", None)
    sincos_masked = False
    if _mask_arr is not None:
        _mask_arr = np.array(_mask_arr).flatten().astype(bool)
        sincos_masked = len(_mask_arr) >= 4 and not _mask_arr[2] and not _mask_arr[3]
    yaw_stats = (_yaw_sincos_stats(_p01_flat, _p99_flat, sincos_masked=sincos_masked)
                 if len(_p01_flat) >= 4 else None)
    print(f"sin/cos masked = {sincos_masked}")

    # Encode seed frames and goal
    seed_idxs = list(range(args.seed_frames))
    goal_idx  = args.seed_frames + args.plan_horizon - 1
    images_sg = images[b:b+1, seed_idxs + [goal_idx]]   # (1, S+1, C, H, W)
    with torch.no_grad():
        zg_feats = model.encode_obs(images_sg)           # (1, S+1, 1, gH, gW, D)
    z_seed = zg_feats[:, :args.seed_frames]              # (1, S, 1, gH, gW, D)
    goal   = zg_feats[:, args.seed_frames:args.seed_frames+1]  # (1, 1, 1, gH, gW, D)

    # Language embedding
    lang_emb = None   # paper models are image-goal only

    # Seed actions (S-1 GT actions for multi-frame seed context)
    act_start = args.seed_frames - 1
    seed_acts = None
    if args.seed_frames > 1:
        seed_acts_raw = actions[b:b+1, :args.seed_frames-1].to(device)  # (1, S-1, A)
        # Re-normalise from GT stats to ckpt stats
        _gt_as   = gt_stats[gt_key]["action"]
        _ckpt_as = stats[ds_key]["action"]
        seed_acts = _renorm_gt_to_ckpt(seed_acts_raw, _gt_as, _ckpt_as, device)  # (1, S-1, A)

    # GT actions in ckpt-normalised space
    gt_acts_gt_norm = actions[b, act_start:act_start+args.plan_horizon].cpu().numpy()  # (T, 4) — GT stats
    # Re-norm to ckpt stats
    gt_acts_ckpt_t = torch.tensor(gt_acts_gt_norm[None], dtype=torch.float32, device=device)
    gt_acts_ckpt_np = _renorm_gt_to_ckpt(
        gt_acts_ckpt_t,
        gt_stats[gt_key]["action"],
        stats[ds_key]["action"],
        device,
    ).squeeze(0).cpu().numpy()  # (T, 4)

    # Unnorm stats for ckpt-normalised perturbed actions → metric
    ckpt_unnorm_stats = stats[ds_key]["action"]

    print(f"\nExample {b}: '{language}'")
    print(f"  GT actions (ckpt-norm): {gt_acts_ckpt_np.round(3)}")

    # ── Sweep and compute ─────────────────────────────────────────────────
    print(f"\nSweeping {args.n_x}×{args.n_yaw} grid  "
          f"(d_x ∈ ±{args.range_x:.2f}, d_yaw ∈ ±{args.range_yaw:.2f} rad) ...")
    (
        d_x_grid, d_yaw_grid,
        xy_all_trajs, costs, gt_xy, gt_acts_unnorm, gt_costs, feats,
    ) = compute_energy_surface(
        model, z_seed, goal, lang_emb,
        gt_acts_ckpt=gt_acts_ckpt_np,
        ckpt_unnorm_stats=ckpt_unnorm_stats,
        yaw_stats=yaw_stats,
        ctxt_window=ctxt_window,
        n_x=args.n_x, n_yaw=args.n_yaw,
        range_x=args.range_x, range_yaw=args.range_yaw,
        device=device,
        seed_acts=seed_acts,
        rollout_batch=args.rollout_batch,
    )

    print("\nGT costs:")
    for k, v in gt_costs.items():
        print(f"  {k}: {v:.5f}")
    for k, arr in costs.items():
        print(f"  {k} grid — min={arr.min():.5f}  max={arr.max():.5f}  "
              f"mean={arr.mean():.5f}")

    # Raw images
    img_seed_np = images[b, args.seed_frames - 1].cpu().numpy()  # (C, H, W)
    img_goal_np = images[b, goal_idx].cpu().numpy()              # (C, H, W)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # ── Save RGB images (PIL — no border) ────────────────────────────────
    import matplotlib.pyplot as plt
    from PIL import Image

    def _to_hwc(img):
        img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    rgb_dir = os.path.join(out_dir, "RGB")
    os.makedirs(rgb_dir, exist_ok=True)
    _rgb_hw = None  # (W, H) for PIL — used to match PCA size
    for fname, arr in [("start.png", img_seed_np), ("goal.png", img_goal_np)]:
        path = os.path.join(rgb_dir, fname)
        pil_rgb = Image.fromarray(_to_hwc(arr))
        if _rgb_hw is None:
            _rgb_hw = pil_rgb.size  # (W, H)
        pil_rgb.save(path)
        print(f"  RGB saved → {path}")

    # ── Save PCA feature images ───────────────────────────────────────────
    pca_dir = os.path.join(out_dir, "PCA")
    os.makedirs(pca_dir, exist_ok=True)
    has_feats = feats is not None and all(
        feats.get(k) is not None for k in ("seed", "goal_gt", "goal_pred_best")
    )
    if has_feats:
        _all_f = [feats[k] for k in ("seed", "goal_gt", "goal_pred_best")]
        D = _all_f[0].shape[-1]
        _all_pts = np.concatenate([f.reshape(-1, D) for f in _all_f], axis=0)
        if np.all(np.isfinite(_all_pts)):
            pca_vis = PCAVisualizer(_all_pts)
            pca_items = [
                ("start_pca.png",    feats["seed"]),
                ("goal_gt_pca.png",  feats["goal_gt"]),
                ("goal_pred_pca.png", feats["goal_pred_best"]),
            ]
            for fname, feat in pca_items:
                rgb_arr = pca_vis(feat)
                if rgb_arr.dtype != np.uint8:
                    rgb_arr = (rgb_arr * 255).clip(0, 255).astype(np.uint8)
                path = os.path.join(pca_dir, fname)
                pil_img = Image.fromarray(rgb_arr)
                if _rgb_hw is not None:
                    pil_img = pil_img.resize(_rgb_hw, Image.NEAREST)
                pil_img.save(path)
                print(f"  PCA saved → {path}")

    # ── Dump raw sweep data (for figure re-rendering) ────────────────────
    _npz_path = os.path.join(out_dir, "energy_surface_data.npz")
    np.savez_compressed(
        _npz_path,
        d_x_grid=d_x_grid, d_yaw_grid=d_yaw_grid,
        xy_all=xy_all_trajs, gt_xy=gt_xy,
        cost_cosine=costs["cosine"], cost_l2=costs["l2"],
        cost_combined=costs["combined"],
        gt_cost_cosine=gt_costs["cosine"], gt_cost_l2=gt_costs["l2"],
        gt_cost_combined=gt_costs["combined"],
        img_start=_to_hwc(img_seed_np), img_goal=_to_hwc(img_goal_np),
        language=np.array(language),
    )
    print(f"  raw sweep data saved → {_npz_path}")

    # ── Plot 3-D and 2-D energy surfaces ─────────────────────────────────
    print(f"\nPlotting energy surfaces → {out_dir}")
    plot_energy_surface(
        d_x_grid, d_yaw_grid,
        xy_all_trajs, costs, gt_xy, gt_acts_unnorm, gt_costs,
        language=language, output_dir=out_dir,
    )
    plot_energy_surface_2d(
        d_x_grid, d_yaw_grid,
        xy_all_trajs, costs, gt_xy, gt_acts_unnorm, gt_costs,
        language=language,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()