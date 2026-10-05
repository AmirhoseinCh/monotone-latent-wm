#!/usr/bin/env python3
"""
Robot inference script: CEM planning with the latent world model.

Given a current observation image and a goal image, runs CEM planning
using the trained world model as dynamics function (with the unicycle
kinematic constraint) and outputs a diagnostic plot with:
  - Current and goal images
  - Planned trajectory with initial CEM sample cloud
  - CEM cost convergence over iterations

Usage:
    python deploy/infer_robot.py \
        --checkpoint /path/to/final_model.pt \
        --current_image current.png \
        --goal_image goal.png \
        --encoder dino \
        --plan_horizon 6 \
        --output_plot robot_plan.png
"""

import argparse
import importlib.util
import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from PIL import Image

from lmwm.world_model import build_world_model
from cem_planning import (
    cem_plan_batch,
    unnormalize_bound_action,
    compute_xy_trajectory_gt,
    _yaw_sincos_stats,
)


def _to_numpy_tree(x):
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def load_image(path: str, img_size: int) -> torch.Tensor:
    """Load image → (1, 1, C, H, W) float32 in [0, 255]; encoder normalises internally."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img)  # (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0)


def _build_cem_stats(astats: dict, cem_var_scale: float, device: torch.device):
    """Derive unicycle stats, yaw stats, and prior mean/std from checkpoint action stats."""
    p01 = np.array(astats["p01"]).flatten()
    p99 = np.array(astats["p99"]).flatten()
    raw_mean = np.array(astats["mean"]).flatten()
    _std_raw = astats.get("std", None)
    raw_std = np.array(_std_raw).flatten() if _std_raw is not None else (p99 - p01) / 5.15

    mask_arr = astats.get("mask", None)
    if mask_arr is not None:
        mask_arr = np.array(mask_arr).flatten().astype(bool)
        sincos_masked = len(mask_arr) >= 4 and not mask_arr[2] and not mask_arr[3]
    else:
        sincos_masked = False

    yaw_stats = _yaw_sincos_stats(p01, p99, sincos_masked=sincos_masked)

    unicycle_stats = {
        "p01_x": float(p01[0]),
        "p99_x": float(p99[0]),
        "p01_y": float(p01[1]),
        "p99_y": float(p99[1]),
        "u_max": float(np.sqrt(max(p99[0], 0.0) ** 2 + p99[1] ** 2)),
    }

    u_mean = float(np.sqrt(raw_mean[0] ** 2 + raw_mean[1] ** 2))
    yaw_mean = float(np.arctan2(float(raw_mean[2]), float(raw_mean[3])))
    u_std = float(np.sqrt(raw_std[0] ** 2 + raw_std[1] ** 2))

    R_sq = raw_mean[2] ** 2 + raw_mean[3] ** 2 + 1e-8
    yaw_std = float(
        np.sqrt(
            (raw_mean[3] ** 2 * raw_std[2] ** 2 + raw_mean[2] ** 2 * raw_std[3] ** 2)
            / R_sq ** 2
        )
    )
    yaw_std = float(np.clip(yaw_std, 0.05, np.pi))

    action_prior_mean = torch.tensor([u_mean, yaw_mean], dtype=torch.float32, device=device)
    action_prior_std = (
        torch.tensor([u_std, yaw_std], dtype=torch.float32, device=device) * cem_var_scale
    )
    print(
        f"  unicycle prior: u_mean={u_mean:.4f}  yaw_mean={yaw_mean:.4f} rad\n"
        f"  unicycle std  : u_std={u_std * cem_var_scale:.4f}  "
        f"yaw_std={yaw_std * cem_var_scale:.4f} rad  (var_scale={cem_var_scale})"
    )
    return unicycle_stats, yaw_stats, action_prior_mean, action_prior_std


def _plot_result(
    curr_img: np.ndarray,
    goal_img: np.ndarray,
    xy_plan: np.ndarray,
    sample_xys: List[np.ndarray],
    iter_costs: np.ndarray,
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def to_hwc(img: np.ndarray) -> np.ndarray:
        img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.4, wspace=0.35, height_ratios=[2, 1]
    )

    ax_curr = fig.add_subplot(gs[0, 0])
    ax_curr.imshow(to_hwc(curr_img))
    ax_curr.set_title("Current (t = 0)", fontsize=12)
    ax_curr.axis("off")

    ax_goal_im = fig.add_subplot(gs[0, 1])
    ax_goal_im.imshow(to_hwc(goal_img))
    ax_goal_im.set_title("Goal", fontsize=12)
    ax_goal_im.axis("off")

    ax_traj = fig.add_subplot(gs[0, 2])
    for xy_s in sample_xys:
        ax_traj.plot(
            xy_s[:, 0], xy_s[:, 1],
            color="orange", alpha=0.15, linewidth=0.8, zorder=1,
        )
    ax_traj.plot(
        xy_plan[:, 0], xy_plan[:, 1], "r.-",
        linewidth=2, markersize=5, label="Planned", zorder=3,
    )
    ax_traj.plot(0, 0, "ko", markersize=7, zorder=5, label="Start")
    ax_traj.plot(xy_plan[-1, 0], xy_plan[-1, 1], "r*", markersize=10, zorder=5)
    ax_traj.set_aspect("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel("x_fwd (m)")
    ax_traj.set_ylabel("y_left (m)")
    ax_traj.set_title("Planned trajectory (top-down)", fontsize=11)
    ax_traj.legend(fontsize=9)

    ax_cost = fig.add_subplot(gs[1, :])
    iters = np.arange(1, len(iter_costs) + 1)
    ax_cost.plot(iters, iter_costs, "o-", color="steelblue", linewidth=2, markersize=5)
    ax_cost.set_xlabel("CEM iteration")
    ax_cost.set_ylabel("min cost")
    ax_cost.set_title("CEM cost vs. iteration", fontsize=11)
    ax_cost.grid(True, alpha=0.3)
    ax_cost.set_xticks(iters)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Robot inference: CEM planning with CAST world model"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint (e.g. best_model.pt)")
    parser.add_argument("--current_image", required=True,
                        help="Path to current observation image")
    parser.add_argument("--goal_image", required=True,
                        help="Path to goal image")
    parser.add_argument("--encoder", default="dino",
                        choices=["dino", "dinov3", "vjepa"],
                        help="Encoder type used during training (default: dino = DINOv2)")
    parser.add_argument("--config", default=None,
                        help="Config file path (default: config.py in checkpoint directory)")
    parser.add_argument("--plan_horizon", type=int, default=6,
                        help="Number of planning steps (default: 6)")
    parser.add_argument("--cem_samples", type=int, default=128,
                        help="CEM population size per iteration (default: 128)")
    parser.add_argument("--cem_iters", type=int, default=6,
                        help="CEM refinement iterations (default: 6)")
    parser.add_argument("--cem_elites", type=int, default=4,
                        help="Elite candidates per iteration (default: 4)")
    parser.add_argument("--cem_var_scale", type=float, default=2.0,
                        help="Initial std scale (default: 2.0)")
    parser.add_argument("--cem_momentum", type=float, default=0.1,
                        help="Momentum for mean/std update (default: 0.1)")
    parser.add_argument("--no_unicycle", action="store_true",
                        help="Disable unicycle kinematic constraint (unicycle ON by default)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile on the predictor")
    parser.add_argument("--output_plot", default="robot_plan.png",
                        help="Output plot path (default: robot_plan.png)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── Load config ───────────────────────────────────────────────────────
    ckpt_dir = os.path.dirname(args.checkpoint)
    config_path = args.config or os.path.join(ckpt_dir, "config.py")
    spec = importlib.util.spec_from_file_location("cfg_module", config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = cfg_module.get_config(encoder_type=args.encoder)
    print(f"Config: {config_path}")

    # ── Build model ───────────────────────────────────────────────────────
    print("\nBuilding model ...")
    model = build_world_model(cfg).to(device)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys   : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()
    print(f"  step={ckpt.get('step', '?')}  val_loss={ckpt.get('val_loss', '?')}")

    if not args.no_compile:
        model.predictor = torch.compile(model.predictor, mode="reduce-overhead")
        print("  Predictor compiled (reduce-overhead)")

    # ── Load normalization stats ──────────────────────────────────────────
    ckpt_stats_raw = ckpt.get("dataset_statistics")
    if ckpt_stats_raw is None:
        raise RuntimeError(
            "Checkpoint has no 'dataset_statistics'. Cannot unnormalize actions."
        )
    ckpt_stats = _to_numpy_tree(ckpt_stats_raw)
    ds_name = list(ckpt_stats.keys())[0]
    astats = ckpt_stats[ds_name]["action"]
    print(f"\nStats from dataset '{ds_name}':")
    print(f"  p01 : {np.array(astats['p01']).flatten().tolist()}")
    print(f"  p99 : {np.array(astats['p99']).flatten().tolist()}")
    print(f"  mean: {np.array(astats['mean']).flatten().tolist()}")

    # ── Load images ───────────────────────────────────────────────────────
    img_size = cfg.img_size
    print(f"\nLoading images at {img_size}×{img_size} ...")
    curr_img_t = load_image(args.current_image, img_size).to(device)
    goal_img_t = load_image(args.goal_image, img_size).to(device)

    curr_img_np = np.array(
        Image.open(args.current_image).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR
        )
    ).transpose(2, 0, 1)
    goal_img_np = np.array(
        Image.open(args.goal_image).convert("RGB").resize(
            (img_size, img_size), Image.BILINEAR
        )
    ).transpose(2, 0, 1)

    # ── Encode observations ───────────────────────────────────────────────
    print("Encoding observations ...")
    imgs_both = torch.cat([curr_img_t, goal_img_t], dim=1)  # (1,2,C,H,W)
    with torch.no_grad():
        feats = model.encode_obs(imgs_both)   # (1,2,1,gH,gW,D)
    z_seed = feats[:, :1]
    goal_feat = feats[:, 1:2]

    # ── Build CEM stats ───────────────────────────────────────────────────
    print("\nBuilding CEM stats ...")
    unicycle_stats, yaw_stats, action_prior_mean, action_prior_std = _build_cem_stats(
        astats, args.cem_var_scale, device
    )
    ctxt_window = getattr(cfg, "ctxt_window", args.plan_horizon + 1)

    # ── CEM planning ──────────────────────────────────────────────────────
    use_unicycle = not args.no_unicycle
    print(
        f"\nRunning CEM: T={args.plan_horizon}  N={args.cem_samples}"
        f"  iters={args.cem_iters}  elites={args.cem_elites}"
        f"  unicycle={'ON' if use_unicycle else 'OFF'}"
        f"  ctxt_window={ctxt_window}"
    )

    best_actions, _, _, iter_info = cem_plan_batch(
        model=model,
        z_seed_B=z_seed,
        goal_B=goal_feat,
        lang_emb_B=None,
        plan_horizon=args.plan_horizon,
        n_samples=args.cem_samples,
        n_iters=args.cem_iters,
        n_elites=args.cem_elites,
        var_scale=args.cem_var_scale,
        momentum=args.cem_momentum,
        cost_fn=None,
        ctxt_window=ctxt_window,
        device=device,
        action_prior_mean=action_prior_mean,
        action_prior_std=action_prior_std,
        yaw_stats=yaw_stats,
        unicycle_stats=unicycle_stats if use_unicycle else None,
        return_iter_info=True,
    )

    # ── Unnormalize and compute planned trajectory ────────────────────────
    best_unnorm = unnormalize_bound_action(best_actions[0], astats)
    xy_plan, _ = compute_xy_trajectory_gt(best_unnorm)

    print("\nPlanned trajectory (metric):")
    for t, (x, y) in enumerate(xy_plan):
        print(f"  t={t}: ({x:+.3f} m, {y:+.3f} m)")

    # ── First-iter samples and cost curve ────────────────────────────────
    sample_xys: List[np.ndarray] = []
    iter_costs = np.array([0.0])
    if iter_info is not None:
        _, iter_costs_all, first_samples_4d = iter_info
        iter_costs = iter_costs_all[:, 0]

        if first_samples_4d is not None:
            _fs_b = first_samples_4d[0]
            for n in range(_fs_b.shape[1]):
                unnorm_n = unnormalize_bound_action(_fs_b[:, n, :], astats)
                xy_n, _ = compute_xy_trajectory_gt(unnorm_n)
                sample_xys.append(xy_n)

        print(
            f"\nCEM cost:  initial={iter_costs[0]:.4f}"
            f"  final={iter_costs[-1]:.4f}"
            f"  ({len(sample_xys)} first-iter samples)"
        )

    # ── Plot ──────────────────────────────────────────────────────────────
    _plot_result(
        curr_img=curr_img_np,
        goal_img=goal_img_np,
        xy_plan=xy_plan,
        sample_xys=sample_xys,
        iter_costs=iter_costs,
        save_path=args.output_plot,
    )


if __name__ == "__main__":
    main()
