"""
lmu_ppo/episodic_bonus.py

E3B-style episodic elliptical bonus for LMU-PPO.

Two classes:
  RunningMeanStd        — Welford online variance tracker (numpy, CPU)
  EllipticalEpisodicBonus — batched Sherman-Morrison inverse (torch, GPU-compatible)

Design notes:
  - M is reset to (1/λ)·I at every episode boundary. Within an episode,
    rank-1 SM updates accumulate observations: b_t = φ^T M_{t-1} φ, which
    is high for novel φ and decays as the SM matrix "fills in" that direction.
  - φ = L2-normalize(y_t) throughout. Raw y_t norms vary (y_norm_mean ≈ 2.0
    from Phase 1), and we want the bonus to reflect directional novelty in φ-
    space, not magnitude. With ‖φ‖=1: b_t ∈ (0, 1/λ] and b_0 = 1/λ exactly.
  - Running std normalization (RunningMeanStd) converts raw b_t to unit-scale
    before weighting by beta_ep. This is critical because b_t is heavy-tailed:
    early-episode bonuses can be 10–100× late-episode bonuses.
  - No masking of b_t at episode boundaries. The SM matrix is reset BEFORE
    the first step of each new episode, so b_0 is the correct high-novelty
    value. Masking it out would throw away the intended exploration signal.
    (Contrast with r_intr, which IS masked at boundaries because the LMU
    state is stale at episode transitions — SM state is not stale because we
    explicitly reset it.)
"""

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# RunningMeanStd
# ─────────────────────────────────────────────────────────────────────────────

class RunningMeanStd:
    """
    Welford online mean/variance, batch-safe.

    Accepts batches of scalar values (b_t per step per env) and tracks
    a global running std used to normalize the episodic bonus.

    Initialized with var=1, count=1e-4 so std=1 at the first step,
    avoiding a divide-by-zero before enough data accumulates.

    Usage:
        rms = RunningMeanStd()
        rms.update(b_t_array)       # b_t_array: (n_envs,) numpy array
        b_normalized = b_t / (rms.std + 1e-6)
    """

    def __init__(self):
        self.mean  = 0.0
        self.var   = 1.0    # start at 1 → std=1 before any data
        self.count = 1e-4   # small nonzero to avoid edge cases at step 0

    def update(self, x: np.ndarray) -> None:
        """Welford parallel batch update."""
        x = np.asarray(x, dtype=np.float64).ravel()
        n = len(x)
        if n == 0:
            return
        batch_mean = float(x.mean())
        batch_var  = float(x.var()) if n > 1 else 0.0

        total = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * n / total
        # Parallel Welford for M2 (sum of squared deviations from mean):
        #   M2_new = M2_old + batch_M2 + delta² * n_old * n_batch / n_total
        self.var = (
            self.var   * self.count
            + batch_var * n
            + delta**2 * self.count * n / total
        ) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))

    def reset(self) -> None:
        self.mean  = 0.0
        self.var   = 1.0
        self.count = 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# EllipticalEpisodicBonus
# ─────────────────────────────────────────────────────────────────────────────

class EllipticalEpisodicBonus:
    """
    Batched E3B episodic bonus: one (C×C) inverse covariance matrix per env.

    φ is assumed to be L2-normalised before being passed in (caller's
    responsibility). With ‖φ‖=1, the initial bonus b_0 = 1/λ for every
    episode, making the scale independent of y_t magnitude.

    Memory: (n_envs × C × C) tensors on device.
    Cost per call: two batched matmuls — O(n_envs × C²) — negligible vs policy.

    Reset behaviour:
        reset(env_ids) reinitialises M[i] = (1/λ)·I for each i in env_ids.
        Call this AFTER env.step() returns done=True, inside the done-env loop.
        The SM matrix will be fresh for the very first step of the next episode.
    """

    def __init__(
        self,
        n_envs:     int,
        dim:        int,
        lambda_reg: float = 1.0,
        device:     str   = 'cpu',
    ):
        self.n_envs     = n_envs
        self.dim        = dim
        self.lambda_reg = lambda_reg
        self.device     = torch.device(device)

        # M[i] = Λ_i^{-1}, shape (n_envs, C, C)
        I_over_lam = torch.eye(dim, device=self.device) / lambda_reg
        self.M = I_over_lam.unsqueeze(0).expand(n_envs, -1, -1).clone()

    @torch.no_grad()
    def bonus_and_update(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute E3B bonus for each env, then update M via Sherman-Morrison.

        Args:
            phi: (n_envs, C) — L2-normalised feature vectors.

        Returns:
            bonus: (n_envs,) — b_t = φ^T M_{t-1} φ ∈ (0, 1/λ].
                   High = novel (unseen direction in episodic history).
                   Low  = familiar (this φ direction is well-covered by past φ's).

        SM update:
            Mphi  = M_{t-1} φ                    (n_envs, C)
            bonus = φ^T Mphi                      (n_envs,)
            M_t   = M_{t-1} - Mphi Mphi^T / (1 + bonus)

        Numerical note: denom = 1 + bonus ≥ 1 always (M is PSD, φ^T Mφ ≥ 0).
        """
        phi  = phi.to(self.device)
        Mphi = torch.bmm(self.M, phi.unsqueeze(-1)).squeeze(-1)    # (n_envs, C)
        bonus = (phi * Mphi).sum(dim=-1)                           # (n_envs,)
        denom = (1.0 + bonus).view(-1, 1, 1)                       # (n_envs, 1, 1)
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)            # (n_envs, C, C)
        self.M = self.M - outer / denom
        return bonus   # (n_envs,) — in [0, 1/λ] after L2-normalised φ

    def reset(self, env_ids) -> None:
        """
        Reset M[i] = (1/λ)·I for all i in env_ids.

        env_ids: list[int] or array-like of env indices that just finished.
        """
        I_over_lam = torch.eye(self.dim, device=self.device) / self.lambda_reg
        for i in env_ids:
            self.M[i] = I_over_lam

    def reset_all(self) -> None:
        """Reset all envs — call at the start of a new learn() call."""
        I_over_lam = torch.eye(self.dim, device=self.device) / self.lambda_reg
        self.M = I_over_lam.unsqueeze(0).expand(self.n_envs, -1, -1).clone()