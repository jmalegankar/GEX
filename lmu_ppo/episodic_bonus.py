"""
Elliptical Episodic Bonus (E3B, Henaff et al. NeurIPS 2022)
────────────────────────────────────────────────────────────

Per-episode Mahalanobis coverage signal computed in feature space φ ∈ R^C:

    b_t = φ_t^T Λ_{t-1}^{-1} φ_t,   Λ_t = λ I + Σ_{i≤t} φ_i φ_i^T

We maintain M_t := Λ_t^{-1} directly via Sherman-Morrison rank-1 updates:

    M_t = M_{t-1} - (M_{t-1} φ φ^T M_{t-1}) / (1 + φ^T M_{t-1} φ)

═══ MATHEMATICAL CORRECTNESS — READ BEFORE EDITING ═══

The Sherman-Morrison identity is exact ONLY when the denominator uses the
TRUE bonus b_true = φ^T M φ. The previous version clamped the denominator
at b ≤ 2:

    bonus = clamp(b_true, max=2.0)
    denom = 1.0 + bonus            # ← WRONG when b_true > 2
    M -= outer / denom              # over-subtracts along φ direction

When true b is e.g. 50, the correct denom is 51 but the clamped denom is 3.
The update over-subtracts by ~17×. After a few such hits, M's smallest
eigenvalue goes negative and b can become negative for new φ — at which
point M is no longer Λ^{-1} for ANY positive-definite Λ and the bonus has
no Mahalanobis interpretation.

This was visible as massive negative spikes in episodic/disc_ratio_b
(-1.4e5 range) that the log-scale plot of episodic/ratio_b hid entirely
because log axes can't render negative values.

═══ NUMERICAL SAFETY POLICY (this version) ═══

The math correctness and the numerical safety are now SEPARATED:

  • The Sherman-Morrison denominator uses the TRUE, unclamped b_true.
  • The RETURNED bonus (which feeds RunningStd and the PPO reward) is
    sanitized: NaN/Inf → 0, negative → 0, clipped to max_bonus.
  • Per-env update outcomes:
      - SAFE  : b_true is finite, ≥ 0, and < reject_threshold
                → standard SM update applied.
      - SKIP  : b_true is NaN/Inf or > reject_threshold
                → M update is omitted (M preserves last-good state).
      - RESET : b_true is finite but negative (M has lost PSD-ness)
                → M reset to (1/λ) I for that env. The within-episode
                  coverage history is lost, but going forward the math
                  is correct again.

This means a fresh run starts with a clean M and stays clean. A run that
inherits a corrupted M (e.g., loaded from an old checkpoint) will heal
itself the first time it encounters a negative bonus.

═══ OPTIONAL: φ NORMALIZATION ═══

The bound on b_true is ||φ||² / λ_min(Λ). With λ=1 and ||φ|| unbounded,
b can be arbitrarily large in transients. Setting normalize_phi=True
L2-normalizes φ inside this class, which bounds b ∈ [0, 1/λ] absolutely.
This is OFF by default (matches E3B paper semantics — φ has its own
learned scale) but can be enabled as an additional safety net if you
observe persistent reject_threshold hits in the diagnostics.

═══ DIAGNOSTICS ═══

Per call, the following are tracked and accumulated until reset_diagnostics():
  - n_skipped : count of (env, step) pairs where M update was skipped
  - n_reset   : count of (env, step) pairs where M was reset (PSD recovery)
  - b_max     : maximum un-sanitized b_true seen this rollout

Call get_diagnostics() at the end of each rollout for logging, then
reset_diagnostics() to start the next rollout fresh.

═══ UNIT TESTS (manually verifiable) ═══

With λ=1, no normalization:

  e0 = unit basis vector
  bonus_and_update(e0) → 1.0    (b = e0^T (I) e0 = 1)
  bonus_and_update(e0) → 0.5    (M is now I - e0 e0^T / 2; e0^T M e0 = 0.5)
  bonus_and_update(e0) → ~0.33  (b → 1/(t+1) for repeated φ)

With phi = 10 * e0 (large magnitude — used to trigger the old bug):

  bonus_and_update(10*e0) → b_true = 100, returned = clamp(100, max=100) = 100
  M update: outer/(1+100) = 100·outer/101 ≈ correct
  After update: M[0,0] = 1 - 100/101 ≈ 0.0099  (NOT negative, NOT zero)
  bonus_and_update(10*e0) → 100 * 0.0099 ≈ 0.99  (small, as expected)

Compare to OLD code:
  bonus_and_update(10*e0) → clamp(100, max=2) = 2, denom=3, M[0,0] = 1 - 100/3 = -32.3
  bonus_and_update(10*e0) on negative M → 100 * (-32.3) = -3230  ← bug
"""

from typing import List, Optional, Union

import numpy as np
import torch


class EllipticalEpisodicBonus:
    """
    Multi-env E3B bonus buffer with numerical safety.

    Args:
        n_envs:           number of parallel envs (matches PPO's n_envs)
        dim:              feature dimension C (matches encoder_dim)
        lambda_reg:       regularization λ. M_0 = (1/λ) I.
        device:           torch device.
        normalize_phi:    L2-normalize φ before SM. Default False.
                          Enables hard bound b ≤ 1/λ; sacrifices magnitude info.
        reject_threshold: skip SM update if b_true exceeds this. Default 1e6.
                          Should rarely fire under normalize_phi=True.
        max_bonus:        clip RETURNED bonus to this. Default 100.0.
                          The downstream RunningStd normalizes further.
    """
    def __init__(
        self,
        n_envs: int,
        dim: int,
        lambda_reg: float = 1.0,
        device: Union[torch.device, str] = 'cpu',
        normalize_phi: bool = False,
        reject_threshold: float = 1e6,
        max_bonus: float = 100.0,
    ):
        self.n_envs = n_envs
        self.dim = dim
        self.lam = lambda_reg
        self.device = torch.device(device)
        self.normalize_phi = normalize_phi
        self.reject_threshold = reject_threshold
        self.max_bonus = max_bonus

        # M: (n_envs, C, C) — each slice is Λ_t^{-1} for one env
        self.M = self._fresh_M()

        # Diagnostics — accumulate across calls until reset_diagnostics()
        self._cum_skipped = 0
        self._cum_reset   = 0
        self._cum_b_max   = -float('inf')
        self._cum_calls   = 0

    def _fresh_M(self) -> torch.Tensor:
        """Fresh M = (1/λ) I, broadcast across n_envs."""
        eye = torch.eye(self.dim, device=self.device) / self.lam
        return eye.unsqueeze(0).expand(self.n_envs, -1, -1).contiguous()

    @torch.no_grad()
    def bonus_and_update(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute b_t = φ^T M_{t-1} φ using TRUE math, then update M to M_t
        via correct Sherman-Morrison. Sanitize the returned bonus.

        Args:
            phi: (n_envs, C) float tensor. Detach before passing in.
                 Must be on self.device.
        Returns:
            bonus: (n_envs,) float tensor on self.device.
                   Always finite, always in [0, max_bonus].
        """
        assert phi.shape == (self.n_envs, self.dim), \
            f"phi shape {phi.shape} != ({self.n_envs}, {self.dim})"

        # ── Optional φ normalization ────────────────────────────────────
        if self.normalize_phi:
            phi = torch.nn.functional.normalize(phi, dim=-1, eps=1e-8)

        # ── 1. TRUE bonus: b = φ^T M φ ──────────────────────────────────
        # Mphi: (n_envs, C, 1) → (n_envs, C)
        Mphi = torch.bmm(self.M, phi.unsqueeze(-1)).squeeze(-1)
        b_true = (phi * Mphi).sum(dim=-1)                    # (n_envs,)

        # ── 2. Classify each env's update safety ────────────────────────
        finite = torch.isfinite(b_true)
        nonneg = b_true >= 0
        small  = b_true < self.reject_threshold

        # SAFE  : standard SM update applied
        # SKIP  : NaN/Inf or huge — preserve last-good M
        # RESET : negative (M lost PSD-ness) — recover by resetting M
        safe_mask  = finite & nonneg & small
        reset_mask = finite & (~nonneg)
        skip_mask  = (~finite) | (finite & ~small)

        # ── 3. Compute SM update with TRUE denom ────────────────────────
        # IMPORTANT: this is the fix. Use b_true (clamped only at min=0
        # to guard 1/0 in pathological cases), NOT a clamped bonus.
        denom = 1.0 + b_true.clamp(min=0.0)                  # (n_envs,) ≥ 1
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)      # (n_envs, C, C)
        delta = outer / denom.view(-1, 1, 1)                 # (n_envs, C, C)

        # Apply update only to safe envs (zero delta elsewhere)
        safe_f = safe_mask.float().view(-1, 1, 1)
        self.M = self.M - delta * safe_f

        # ── 4. Reset M for negative-bonus envs ──────────────────────────
        # M has lost PSD; reset to fresh (1/λ)I to recover correctness.
        # The Python loop is fine: in healthy training reset_mask is empty.
        if reset_mask.any():
            eye = torch.eye(self.dim, device=self.device) / self.lam
            for i in torch.where(reset_mask)[0].tolist():
                self.M[i] = eye

        # ── 5. Update diagnostics ───────────────────────────────────────
        self._cum_skipped += int(skip_mask.sum().item())
        self._cum_reset   += int(reset_mask.sum().item())
        if finite.any():
            batch_max = float(b_true[finite].max().item())
            if batch_max > self._cum_b_max:
                self._cum_b_max = batch_max
        self._cum_calls += 1

        # ── 6. Sanitize returned bonus ──────────────────────────────────
        # NaN/Inf → 0; negative → 0; clip to max_bonus.
        # This is what feeds RunningStd and (via b_norm) the PPO reward.
        zeros = torch.zeros_like(b_true)
        b_clean = torch.where(finite, b_true, zeros)
        b_clean = b_clean.clamp(min=0.0, max=self.max_bonus)

        return b_clean

    @torch.no_grad()
    def reset(self, env_ids: Union[List[int], np.ndarray, torch.Tensor]) -> None:
        """Reset M to (1/λ)I for specified envs. Call at episode boundaries."""
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
        """Reset all envs' M. Call at training start or after a full rollout."""
        self.M = self._fresh_M()

    @torch.no_grad()
    def min_eigenvalue(self) -> float:
        """
        Smallest eigenvalue of M across ALL envs.
        Should be > 0 for correctly-maintained M (since M = Λ^{-1} of PD Λ).
        If this dips negative, the math is broken — investigate immediately.
        Computed via eigvalsh which assumes symmetry (M is symmetric by
        construction since outer products are symmetric).
        Cost: O(n_envs · C^3). Call once per rollout, not per step.
        """
        try:
            eigvals = torch.linalg.eigvalsh(self.M)           # (n_envs, C)
            return float(eigvals.min().item())
        except Exception:
            return float('nan')

    def reset_diagnostics(self) -> None:
        """Zero the cumulative counters. Call at the start of each rollout."""
        self._cum_skipped = 0
        self._cum_reset   = 0
        self._cum_b_max   = -float('inf')
        self._cum_calls   = 0

    def get_diagnostics(self) -> dict:
        """
        Returns the cumulative numerical-safety counters since last reset.

        Keys:
          n_skipped : total (env, step) pairs where M update was skipped
          n_reset   : total (env, step) pairs where M was reset (PSD recovery)
          b_max     : max un-sanitized b_true seen (-inf if no finite values)
          n_calls   : total bonus_and_update calls (n_envs * n_steps)
          skip_frac : n_skipped / (n_calls * n_envs)
          reset_frac: n_reset / (n_calls * n_envs)
        """
        n_steps = max(self._cum_calls, 1)
        denom = float(n_steps * self.n_envs)
        return {
            'n_skipped':  self._cum_skipped,
            'n_reset':    self._cum_reset,
            'b_max':      self._cum_b_max,
            'n_calls':    self._cum_calls,
            'skip_frac':  self._cum_skipped / denom,
            'reset_frac': self._cum_reset / denom,
        }


class RunningStd:
    """
    Online running std (Welford's algorithm, batched).

    Used to normalize the E3B bonus stream so beta_ep is scale-invariant.
    This is a single-stream tracker (not per-env): all envs' bonuses
    contribute to one running estimate.

    Numerical safety: non-finite values are dropped before update. This is
    defensive — under the new EllipticalEpisodicBonus the input is always
    finite, but a stray NaN here would corrupt mean/var permanently.
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
        # Defensive: drop any non-finite entries so they can't poison the
        # running statistics (Welford propagates NaN permanently).
        x = x[np.isfinite(x)]
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

    Classifies steps by ground-truth ball/key visibility (matching the
    Phase 1 diagnostic). Reports rolling-mean ratios per rollout.

    Under the corrected EllipticalEpisodicBonus, ratio_b should stay
    strictly positive throughout training. Negative ratio_b indicates
    M corruption — a regression bug.
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
            obs_images: (n_envs, C, H, W) uint8 numpy. Channel 0 is object IDs.
            bonus:      (n_envs,) float numpy — the bonus at this step.
        """
        obj = obs_images[:, 0]                                # (n_envs, H, W)
        n_ball = (obj == self.BALL_IDX).sum(axis=(1, 2))
        n_key = (obj == self.KEY_IDX).sum(axis=(1, 2))
        n_hint_objs = n_ball + n_key                          # (n_envs,)

        # hint     : exactly 1 hint object visible
        # corridor : 0 hint objects visible
        # excluded : >= 2 (decision junction)
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
        Returns dict of scalars for logging and resets accumulators.
        Keys: b_hint_mean, b_corridor_mean, ratio_b, n_hint, n_corridor.
        Empty buckets are omitted.
        """
        stats = {
            'n_hint': self.b_hint_count,
            'n_corridor': self.b_corridor_count,
        }
        if self.b_hint_count > 0:
            stats['b_hint_mean'] = self.b_hint_sum / self.b_hint_count
        if self.b_corridor_count > 0:
            stats['b_corridor_mean'] = self.b_corridor_sum / self.b_corridor_count
        if self.b_hint_count > 0 and self.b_corridor_count > 0:
            stats['ratio_b'] = stats['b_hint_mean'] / (stats['b_corridor_mean'] + 1e-8)

        self._reset_accumulators()
        return stats