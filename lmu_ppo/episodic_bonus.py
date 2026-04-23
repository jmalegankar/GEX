"""
Elliptical Episodic Bonus (E3B, Henaff et al. NeurIPS 2022)
────────────────────────────────────────────────────────────

Per-episode Mahalanobis coverage signal computed in the LMU's W_query readout
space y ∈ R^C. The bonus at step t is

    b_t = y_t^T Λ_{t-1}^{-1} y_t

where Λ_t = λ I + Σ_{i=1..t-1} y_i y_i^T accumulates over the current episode
and is reset to λI at episode boundary.

This class maintains M_t := Λ_t^{-1} directly (never inverts) via Sherman-
Morrison rank-1 updates:

    M_t = M_{t-1} - (M_{t-1} y y^T M_{t-1}) / (1 + y^T M_{t-1} y)

Cost per step per env: O(C^2), dominated by one bmm of shape (C,C) × (C,1).
For C=64, n_envs=16: ~65k flops per env step. Negligible.

Design choices and why:

1. M stored as (n_envs, C, C) on the SAME device as y. Sherman-Morrison
   update is a single batched bmm + outer product + elementwise subtract.
   No CPU roundtrip, no env-loop in Python.

2. Bonus is computed with M_{t-1} (before update), matching E3B paper. This
   matters because if we updated first, y at step 1 of a new episode would
   give b_1 = 0 (it would have already contributed to M_1, so M_1 y is
   colinear with y and the Sherman-Morrison numerator cancels exactly).

3. reset() takes an integer (env idx) or bool mask over n_envs. Called at
   episode boundaries. The caller is responsible for deciding when an
   episode ended — we match SB3's convention where `dones[i] == True` means
   the episode just ended and the NEXT observation is the start of a new
   episode.

4. Running std normalization (RunningStd below) is a separate class. The
   bonus is heavy-tailed; raw values are not scale-comparable to extrinsic
   reward. Normalize by a running std before scaling with beta_ep. This
   matches RND and E3B practice.

Testing notes (add to your test suite, not this file):
  - With λ=1 and phi=e_0 (unit vector, first coord), b_1 should equal 1.0
    exactly. Then M_1 = I - (e_0 e_0^T)/2, and b_2 with the same phi equals
    0.5 (since the direction is now 'half used up').
  - Two orthogonal phi should give b_1 = b_2 = 1.0 (no interference).
  - Reset then observe same phi again → bonus returns to 1.0.
"""

from typing import List, Optional, Union

import numpy as np
import torch


class EllipticalEpisodicBonus:
    """
    Multi-env E3B bonus buffer.

    Args:
        n_envs:     number of parallel envs (matches PPO's n_envs)
        dim:        feature dimension C (matches encoder_dim in our LMU setup)
        lambda_reg: regularization λ; M is initialized to (1/λ) I
        device:     torch device (same as the policy)
    """
    def __init__(
        self,
        n_envs: int,
        dim: int,
        lambda_reg: float = 1.0,
        device: Union[torch.device, str] = 'cpu',
    ):
        self.n_envs = n_envs
        self.dim = dim
        self.lam = lambda_reg
        self.device = torch.device(device)
        # M: (n_envs, C, C) — each slice is Λ_t^{-1} for one env
        self.M = self._fresh_M()

    def _fresh_M(self) -> torch.Tensor:
        """Fresh M = (1/λ) I, broadcast across n_envs."""
        eye = torch.eye(self.dim, device=self.device) / self.lam
        return eye.unsqueeze(0).expand(self.n_envs, -1, -1).contiguous()

    @torch.no_grad()
    def bonus_and_update(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute b_t using M_{t-1}, then update M to M_t in place.

        Args:
            phi: (n_envs, C) float tensor. Detach before passing in.
        Returns:
            bonus: (n_envs,) float tensor on same device as phi.
        """
        assert phi.shape == (self.n_envs, self.dim), \
            f"phi shape {phi.shape} != ({self.n_envs}, {self.dim})"

        # Mphi: (n_envs, C, 1) then squeeze → (n_envs, C)
        Mphi = torch.bmm(self.M, phi.unsqueeze(-1)).squeeze(-1)

        # b_t = phi^T M phi — per-env dot product
        bonus = (phi * Mphi).sum(dim=-1)  # (n_envs,)

        # Sherman-Morrison:
        # M_new = M - (M phi phi^T M) / (1 + phi^T M phi)
        # Numerator = outer(Mphi, Mphi); denominator = 1 + bonus
        denom = 1.0 + bonus  # (n_envs,), always >= 1
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)  # (n_envs, C, C)
        self.M = self.M - outer / denom.view(-1, 1, 1)

        return bonus

    @torch.no_grad()
    def reset(self, env_ids: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """
        Reset M to (1/λ)I for specified envs. Call at episode boundaries.

        Args:
            env_ids: list/array of env indices, or a bool mask of shape (n_envs,).
                     Empty list is a no-op.
        """
        # Normalize to a list of int indices
        if isinstance(env_ids, (np.ndarray, torch.Tensor)):
            if env_ids.dtype == torch.bool or env_ids.dtype == np.bool_:
                idx = np.where(env_ids)[0].tolist()
            else:
                idx = np.asarray(env_ids).tolist()
        else:
            idx = list(env_ids)

        if not idx:
            return

        eye = torch.eye(self.dim, device=self.device) / self.lam
        for i in idx:
            self.M[i] = eye

    @torch.no_grad()
    def reset_all(self) -> None:
        """Reset all envs' M. Call at the start of training or after a full rollout."""
        self.M = self._fresh_M()


class RunningStd:
    """
    Online running std (Welford's algorithm, batched).

    Used to normalize the E3B bonus stream. Raw b values have large scale
    variation across training; dividing by a running std makes beta_ep
    scale-invariant.

    This is a single-stream tracker (not per-env). The bonus values from all
    envs are pooled into the same running estimate.
    """
    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Args:
            x: 1-D array/tensor of values to add to the running stat.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return

        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        # Chan's parallel variance algorithm for numerical stability
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = float(new_mean)
        self.var = float(new_var)
        self.count = float(tot_count)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))


class HintCorridorDiscriminationTracker:
    """
    Live monitoring of ratio_b = E[b | hint] / E[b | corridor] during training.

    The plan.md risk register flags "b_ep loses discrimination as the policy
    learns" as Medium. This tracker catches that live by classifying steps
    using the same ground-truth ball/key visibility signal the Phase 1
    diagnostic used, and reports rolling-mean ratios per rollout.

    Call .record(obs, bonus) at every rollout step. Call .flush() at rollout
    end; it returns a dict of scalars to log and resets the accumulators.

    If ratio_b drops below 1.5 for multiple consecutive rollouts, the signal
    is collapsing — that's the trigger to add the inverse dynamics auxiliary
    loss (plan.md Phase 2.5).
    """
    BALL_IDX = 6
    KEY_IDX = 5

    def __init__(self):
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self.b_hint_sum = 0.0
        self.b_hint_count = 0
        self.b_corridor_sum = 0.0
        self.b_corridor_count = 0

    def record(self, obs_images: np.ndarray, bonus: np.ndarray) -> None:
        """
        Args:
            obs_images: (n_envs, C, H, W) uint8 numpy — after VecTransposeImage,
                        channel 0 is object IDs.
            bonus:      (n_envs,) float numpy — the bonus at this step.
        """
        # Per-env: how many hint objects visible?
        obj = obs_images[:, 0]  # (n_envs, H, W)
        n_ball = (obj == self.BALL_IDX).sum(axis=(1, 2))
        n_key = (obj == self.KEY_IDX).sum(axis=(1, 2))
        n_hint_objs = n_ball + n_key  # (n_envs,)

        # Categorize (matching the diagnostic script):
        #   hint     : exactly 1 hint object visible
        #   corridor : 0 hint objects visible
        #   excluded : >= 2 (decision junction)
        hint_mask = n_hint_objs == 1
        corridor_mask = n_hint_objs == 0

        if hint_mask.any():
            self.b_hint_sum += float(bonus[hint_mask].sum())
            self.b_hint_count += int(hint_mask.sum())
        if corridor_mask.any():
            self.b_corridor_sum += float(bonus[corridor_mask].sum())
            self.b_corridor_count += int(corridor_mask.sum())

    def flush(self) -> dict:
        """
        Returns a dict of scalars for logging and resets accumulators.
        Keys: b_hint_mean, b_corridor_mean, ratio_b, n_hint, n_corridor.
        If a bucket is empty, that stat is omitted.
        """
        stats = {
            'n_hint': self.b_hint_count,
            'n_corridor': self.b_corridor_count,
        }
        if self.b_hint_count > 0:
            h_mean = self.b_hint_sum / self.b_hint_count
            stats['b_hint_mean'] = h_mean
        if self.b_corridor_count > 0:
            c_mean = self.b_corridor_sum / self.b_corridor_count
            stats['b_corridor_mean'] = c_mean
        if self.b_hint_count > 0 and self.b_corridor_count > 0:
            stats['ratio_b'] = stats['b_hint_mean'] / (stats['b_corridor_mean'] + 1e-8)

        self._reset_accumulators()
        return stats