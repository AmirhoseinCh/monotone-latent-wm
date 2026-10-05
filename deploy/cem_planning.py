"""
CEM (Cross-Entropy Method) planning helpers for the latent world model.

Lightweight copy of the CEM planner in eval_planning.py (no TensorFlow / dataset deps).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─── Trajectory helpers ───────────────────────────────────────────────────────

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


# ─── Yaw / unicycle helpers ───────────────────────────────────────────────────

def _yaw_sincos_stats(p01: np.ndarray, p99: np.ndarray, sincos_masked: bool = False) -> dict:
    """BOUNDS normalisation constants for sin_yaw (dim 2) and cos_yaw (dim 3)."""
    return {
        "p01_sin": float(p01[2]), "p99_sin": float(p99[2]),
        "p01_cos": float(p01[3]), "p99_cos": float(p99[3]),
        "sincos_masked": sincos_masked,
    }


def _expand_yaw_to_sincos(acts_3d: torch.Tensor, ys: dict) -> torch.Tensor:
    """(…, 3) [x_norm, y_norm, yaw_rad] → (…, 4) [x_norm, y_norm, sin_out, cos_out]."""
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
        sin_r, cos_r = sin_n, cos_n
    else:
        sin_r = (sin_n + 1.0) * (ys["p99_sin"] - ys["p01_sin"]) / 2.0 + ys["p01_sin"]
        cos_r = (cos_n + 1.0) * (ys["p99_cos"] - ys["p01_cos"]) / 2.0 + ys["p01_cos"]
    yaw = torch.atan2(sin_r, cos_r)
    return torch.cat([acts_4d[..., :2], yaw.unsqueeze(-1)], dim=-1)


def _unicycle_to_3d(samples: torch.Tensor, us: dict) -> torch.Tensor:
    """(…, 2) [u_metric, delta_yaw] → (…, 3) [x_norm, y_norm, delta_yaw]."""
    u, yaw = samples[..., 0], samples[..., 1]
    x_m = u * torch.cos(yaw)
    y_m = u * torch.sin(yaw)
    p01x, p99x = us["p01_x"], us["p99_x"]
    p01y, p99y = us["p01_y"], us["p99_y"]
    x_n = (2.0 * (x_m - p01x) / (p99x - p01x + 1e-8) - 1.0).clamp(-1., 1.)
    y_n = (2.0 * (y_m - p01y) / (p99y - p01y + 1e-8) - 1.0).clamp(-1., 1.)
    return torch.stack([x_n, y_n, yaw], dim=-1)


def _3d_to_unicycle(samples_3d: torch.Tensor, us: dict) -> torch.Tensor:
    """(…, 3) [x_norm, y_norm, yaw_rad] → (…, 2) [u_metric, yaw_rad]."""
    x_n, y_n, yaw = samples_3d[..., 0], samples_3d[..., 1], samples_3d[..., 2]
    p01x, p99x = us["p01_x"], us["p99_x"]
    p01y, p99y = us["p01_y"], us["p99_y"]
    x_m = (x_n + 1.0) * (p99x - p01x) / 2.0 + p01x
    y_m = (y_n + 1.0) * (p99y - p01y) / 2.0 + p01y
    u = (x_m.pow(2) + y_m.pow(2)).sqrt()
    return torch.stack([u, yaw], dim=-1)


# ─── Noise helpers ────────────────────────────────────────────────────────────

def _prepare_action_corr(
    action_corr: Optional[np.ndarray],
    A: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Prepare a Cholesky factor for cross-dimension correlation."""
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


def _sample_correlated_noise(
    B: int, T: int, N: int, A: int,
    rho: float,
    device: torch.device,
    corr_chol: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AR(1) temporally-correlated noise with unit stationary variance."""
    if isinstance(rho, (int, float)):
        rho_vec = torch.full((A,), float(rho), device=device)
    else:
        rho_vec = torch.as_tensor(rho, device=device, dtype=torch.float32)
    if (rho_vec == 0).all():
        noise = torch.randn(B, T, N, A, device=device)
        if corr_chol is not None:
            noise = noise @ corr_chol.T
        return noise
    innov_scale = torch.sqrt(1.0 - rho_vec * rho_vec)
    noise = torch.zeros(B, T, N, A, device=device)
    prev = torch.zeros(B, N, A, device=device)
    for t in range(T):
        eps = torch.randn(B, N, A, device=device)
        if corr_chol is not None:
            eps = eps @ corr_chol.T
        prev = rho_vec * prev + innov_scale * eps
        noise[:, t] = prev
    return noise


# ─── CEM core ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def cem_unroll(
    model,
    z_seed_N: torch.Tensor,
    actions_TNA: torch.Tensor,
    lang_N: Optional[torch.Tensor],
    ctxt_window: int,
    seed_acts_SNA: Optional[torch.Tensor] = None,
    goal_N: Optional[torch.Tensor] = None,
    traj_cost_fn: Optional[str] = None,
    traj_weight: float = 0.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Autoregressive WM rollout for N candidate action sequences.

    Returns (pred_final, intermed_cost):
        pred_final:    (N, 1, 1, gH, gW, D) — last predicted frame.
        intermed_cost: (N,) intermediate cost scaled by traj_weight, or None.
    """
    T, N = actions_TNA.shape[:2]
    S = z_seed_N.shape[1]
    *spatial, D = z_seed_N.shape[2:]
    buf_len = S + T

    feat_buf = z_seed_N.new_empty(N, buf_len, *spatial, D)
    feat_buf[:, :S] = z_seed_N

    A = actions_TNA.shape[2]
    act_buf = actions_TNA.new_empty(N, buf_len, A)
    if seed_acts_SNA is not None and S > 1:
        act_buf[:, :S - 1] = seed_acts_SNA.permute(1, 0, 2)

    pred_next: Optional[torch.Tensor] = None
    intermed_cost: Optional[torch.Tensor] = None
    use_traj = (goal_N is not None and traj_weight > 0.0 and T > 1)

    for t in range(T):
        act_buf[:, S - 1 + t] = actions_TNA[t]

        ctx_end = S + t
        ctx_start = max(0, ctx_end - ctxt_window)
        ctx_feats = feat_buf[:, ctx_start:ctx_end]
        ctx_acts = act_buf[:, ctx_start:ctx_end]

        pred = model.forward_pred(ctx_feats, ctx_acts, lang_N)
        pred_next = pred[:, -1:]

        if t < T - 1:
            feat_buf[:, S + t] = pred_next[:, 0]
            if use_traj:
                c = cem_cost(pred_next, goal_N, traj_cost_fn)
                intermed_cost = c if intermed_cost is None else intermed_cost + c

    if intermed_cost is not None:
        intermed_cost = intermed_cost * (traj_weight / (T - 1))

    return pred_next, intermed_cost


def cem_cost(
    predicted_NTvHWD: torch.Tensor,
    goal_N: torch.Tensor,
    cost_fn: Optional[str] = None,
) -> torch.Tensor:
    """Cost between the last predicted frame and the goal frame.

    cost_fn: "cosine" | "mse" | "l1" | None (combined cosine+mse, default).
    Returns: (N,) — lower is better.
    """
    final = predicted_NTvHWD[:, -1].flatten(1)
    g = goal_N.flatten(1)

    if cost_fn == "cosine":
        return 1.0 - F.cosine_similarity(final, g, dim=1)
    if cost_fn == "mse":
        return ((final - g) ** 2).mean(dim=1)
    if cost_fn == "l1":
        return torch.abs(final - g).mean(dim=1)
    return (1.0 - F.cosine_similarity(final, g, dim=1)
            + ((final - g) ** 2).mean(dim=1))


@torch.no_grad()
def cem_plan_batch(
    model,
    z_seed_B: torch.Tensor,
    goal_B: torch.Tensor,
    lang_emb_B: Optional[torch.Tensor],
    plan_horizon: int,
    n_samples: int,
    n_iters: int,
    n_elites: int,
    var_scale: float,
    momentum: float,
    cost_fn: Optional[str],
    ctxt_window: int,
    device: torch.device,
    action_prior_mean: Optional[torch.Tensor] = None,
    seed_acts_B: Optional[torch.Tensor] = None,
    yaw_stats: Optional[dict] = None,
    action_prior_std: Optional[torch.Tensor] = None,
    traj_weight: float = 0.0,
    temporal_corr=0.0,
    action_corr: Optional[np.ndarray] = None,
    gt_action_sample_B: Optional[torch.Tensor] = None,
    action_cluster_mean_B: Optional[torch.Tensor] = None,
    action_cluster_std_B: Optional[torch.Tensor] = None,
    return_samples: bool = False,
    gt_action_cost_B: Optional[torch.Tensor] = None,
    return_iter_info: bool = False,
    unicycle_stats: Optional[dict] = None,
):
    """Batched CEM planning for B examples simultaneously.

    Returns: (best_actions, sample_actions_or_None, best_cost_or_None, iter_info_or_None)
        best_actions: (B, plan_horizon, 4) normalized [-1, 1]
        iter_info: (iter_means, iter_costs, first_samples_4d) when return_iter_info=True
    """
    B = z_seed_B.shape[0]
    S = z_seed_B.shape[1]
    A_model = model.predictor.action_dim if hasattr(model.predictor, "action_dim") else 4
    A = 2 if unicycle_stats is not None else (3 if yaw_stats is not None else A_model)
    T, N = plan_horizon, n_samples
    n_elites = min(n_elites, N)

    z_seed_BN = z_seed_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
    goal_BN = goal_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
    lang_BN = (lang_emb_B.repeat_interleave(N, dim=0).to(torch.bfloat16)
               if lang_emb_B is not None else None)
    seed_acts_BN = None
    if seed_acts_B is not None and S > 1:
        seed_acts_BN = (seed_acts_B.repeat_interleave(N, dim=0)
                        .to(torch.bfloat16)
                        .permute(1, 0, 2))

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
    first_samples_4d = None
    last_cost = None
    _iter_means = [] if return_iter_info else None
    _iter_costs = [] if return_iter_info else None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for iter_j in range(n_iters):
            noise = _sample_correlated_noise(B, T, N, A, temporal_corr, device, corr_chol=corr_chol)
            action_samples = mean.unsqueeze(2) + std.unsqueeze(2) * noise
            action_samples = _clamp(action_samples.reshape(B * T * N, A)).reshape(B, T, N, A)
            action_samples[:, :, 0] = _clamp(mean)
            if gt_internal is not None and gt_sample_idx is not None:
                action_samples[:, :, gt_sample_idx] = _clamp(gt_internal)
            if return_samples:
                last_samples = action_samples
            if return_iter_info and iter_j == 0:
                _fs = action_samples
                if unicycle_stats is not None:
                    _fs = _unicycle_to_3d(_fs, unicycle_stats)
                if yaw_stats is not None:
                    _fs = _expand_yaw_to_sincos(_fs, yaw_stats)
                first_samples_4d = _fs.detach().float().cpu()

            acts_BTN = action_samples.permute(0, 2, 1, 3).reshape(B * N, T, A)
            if unicycle_stats is not None:
                acts_BTN = _unicycle_to_3d(acts_BTN, unicycle_stats)
            if yaw_stats is not None:
                acts_BTN = _expand_yaw_to_sincos(acts_BTN, yaw_stats)
            actions_T_BN = acts_BTN.permute(1, 0, 2).to(torch.bfloat16)

            predicted, i_cost = cem_unroll(
                model, z_seed_BN, actions_T_BN, lang_BN, ctxt_window,
                seed_acts_SNA=seed_acts_BN,
                goal_N=goal_BN, traj_cost_fn=cost_fn, traj_weight=traj_weight,
            )
            cost = cem_cost(predicted, goal_BN, cost_fn).float().reshape(B, N)
            if i_cost is not None:
                cost = cost + i_cost.float().reshape(B, N)
            last_cost = cost

            elite_idxs = torch.topk(-cost, n_elites, dim=1).indices
            idx_exp = elite_idxs[:, None, :, None].expand(B, T, n_elites, A)
            elite_acts = action_samples.gather(2, idx_exp)

            new_mean = elite_acts.mean(dim=2)
            new_std = elite_acts.std(dim=2).clamp(min=0.01)
            mean = _clamp(new_mean * (1 - momentum) + mean * momentum)
            std = new_std * (1 - momentum) + std * momentum

            if _iter_costs is not None:
                _iter_costs.append(cost.min(dim=1).values.float().cpu())
                _iter_means.append(mean.detach().float().cpu())

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
            torch.stack(_iter_means, dim=0).numpy(),
            torch.stack(_iter_costs, dim=0).numpy(),
            first_samples_4d.numpy() if first_samples_4d is not None else None,
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
    return best, None, best_cost, iter_info
