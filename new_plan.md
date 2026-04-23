# LMU-PPO: Episodic-Aware Extension Plan

## Context

Current validated baseline: Gated-write LMU with r_intr (lifelong) on MemoryS11 (~1.5M steps) and MemoryS13 (~2M steps), both using MemoryStartWrapper to place the agent near the goal. The gated write is mathematically validated (two null conditions, W_pre isometry, Cayley updates stable). r_intr at β=0.001 is established as a helpful but not essential lifelong signal.

## Goal

Enable the agent to solve MiniGrid Memory *without* MemoryStartWrapper, then progress to ObstructedMaze-Full, POPGym, and eventually Craftax. The missing ingredient is episodic novelty. We add an E3B-style elliptical episodic bonus computed in the LMU's readout space.

## Core Thesis

r_intr is a 1-step world-model prediction error (lifelong signal). The elliptical episodic bonus b_t = y^T Λ_t^{-1} y is a per-episode coverage signal in the policy's readout space. Together they match the two-signal structure every successful MiniGrid exploration method uses (RIDE, NovelD, AGAC, E3B, DEIR), while reusing the LMU's existing representations rather than bolting on a separate encoder or VAE.

---

## Phase 0 — Low-risk telemetry (1–2 days)

These are zero-risk changes. Ship them first; they unblock every diagnostic we'll need later.

### 0.1 Fix mem_start.py comment
- dir=2 is West in MiniGrid, not "right". Fix comment; keep code behavior.

### 0.2 LayerNorm on critic path only
In policies.py `_critic_input`:
```python
m_pooled = F.layer_norm(m.mean(dim=1), [self.encoder_dim])
```
DO NOT apply LayerNorm inside LMUCell — the Legendre dynamics depend on raw magnitude. Only normalize at the read-out.

### 0.3 Per-component gradient clipping
In lmu_ppo.py `train()`, replace the global clip:
```python
for component in [self.policy.encoder, self.policy.lmu_cell,
                  self.policy.actor, self.policy.critic]:
    clip_grad_norm_(component.parameters(), self.max_grad_norm)
```
W_pre is excluded because ortho_update already zeros its grad.

### 0.4 Log W_pre gradient norm before zeroing
In lmu_ppo.py train(), immediately before ortho_update:
```python
if self.policy.lmu_cell.W_pre.weights.grad is not None:
    wpre_gn = self.policy.lmu_cell.W_pre.weights.grad.norm().item()
    self.logger.record("debug/W_pre_grad_norm", wpre_gn)
self.policy.lmu_cell.W_pre.ortho_update(lr=1e-3)
```
If wpre_gn stays >> 1 consistently, the Cayley lr=1e-3 is too large.

### 0.5 Expanded r_intr diagnostics in collect_rollouts
Add:
```python
# prod_positive_frac: Bug 1 diagnostic (cold-start sign bias)
#   Near 1.0 early + stays near 1.0  → bias is real
#   Near 0.5 throughout              → bias is not a problem
prod = gate * innov  # need to expose this from lmu cell
logger.record("debug/prod_positive_frac",
              (prod > 0).float().mean().item())

# u_x_zero_frac: encoder ReLU suppressing write inputs
logger.record("debug/u_x_zero_frac",
              (u_x.abs() < 1e-6).float().mean().item())

# Per-channel novelty contribution: which encoder channels drive r_intr
logger.record("debug/gate_innov_per_channel",
              prod.abs().mean(dim=0).tolist())  # list of C floats
```
This requires exposing gate, innov, prod from `LMUCell.forward` alongside r_intr. Minimal API change: return them in a dict.

### 0.6 Validate A0 baseline with new telemetry
Re-run MemoryS11 with the current gated write + new logging for 3 seeds. Confirm:
- Steps-to-solve ≈ 1.5M (consistent with prior)
- `prod_positive_frac` behavior → informs 0.7 decision
- `u_x_zero_frac` < 0.3 (if ≥ 0.3, encoder ReLU is saturating)
- `W_pre_ortho_error` < 1e-3 throughout
- `m_norm` stable (not monotone growing or shrinking)

### 0.7 Conditional: apply e_m ~ N(0, 0.01) init
Only if `prod_positive_frac` stays near 1.0 for > 10 steps into each episode. Apply the minimal fix:
```python
# In LMUCell._reset_parameters:
nn.init.normal_(self.e_m, mean=0.0, std=0.01)
```
Rationale: the gate=0 null condition preserves silent-write behavior even with nonzero e_m. The Voelker 2019 e_m=0 argument was for the ungated LMU. Do NOT change to gate - innov (signed innovation); that breaks both null conditions and risks a feedback-loop instability.

---

## Phase 1 — Post-hoc episodic bonus diagnostic (2–3 days)

**This is the go/no-go gate for the whole plan. Run it before writing any training code.**

### 1.1 Checkpoint a converged S11 baseline
Load a converged S11 checkpoint (e.g., from 0.6).

### 1.2 Offline episodic bonus computation script
Create `scripts/diagnose_e3b.py`:
- Roll out the policy for N=50 episodes (no training, no bonus in rewards)
- At each step, extract y_t from the LMU's forward pass
- Maintain per-episode buffer M_t = Λ_t^{-1}, reset at episode start
- Compute b_t = y^T M_{t-1} y via Sherman-Morrison
- Also compute r_intr_t (already available)

### 1.3 Discrimination metrics
Classify each timestep as:
- hint_room: first ~10 steps where r_intr > mean + 1.5·std (proxy)
- corridor: all other non-hint steps

Then report:
```
b_hint_mean, b_hint_std
b_corridor_mean, b_corridor_std
ratio_b = b_hint_mean / b_corridor_mean          # target: > 3
ratio_r = r_intr_hint_mean / r_intr_corridor_mean # for comparison
```

Also:
```
b_start_of_episode (first 3 steps): should be highest
b_late_episode (last 20%):         should be near zero if exploration complete
```

### 1.4 Decision table

| ratio_b | Decision |
|---|---|
| > 3 | Proceed to Phase 2 with y as φ. |
| 1.5–3 | Proceed to Phase 2 but with inverse-dynamics auxiliary loss on y. |
| < 1.5 | Stop. Investigate: is y collapsing to a low-dimensional manifold? |

### 1.5 Secondary diagnostic: aliasing check
Does y at a corridor-step-50 (with hint seen at step 1) look like y at a different-episode corridor-step-50 (no hint)? Compute cosine and also Mahalanobis distance between them. If cosine > 0.95, the hint information has decayed from the memory too much for the readout to distinguish — this is a theta problem (LegT window too short) or an e_m problem (memory write-out too weak).

---

## Phase 2 — E3B implementation (1 week)

Conditional on Phase 1 success.

### 2.1 EpisodicBonus class
New file `lmu_ppo/episodic_bonus.py`:

```python
class EllipticalEpisodicBonus:
    """
    E3B-style episodic bonus maintained per-environment.
    Uses Sherman-Morrison update; O(C^2) per step per env.
    Reset Λ → λI at episode boundary.
    """
    def __init__(self, n_envs: int, dim: int,
                 lambda_reg: float = 1.0, device='cuda'):
        self.n_envs = n_envs
        self.dim = dim
        self.lam = lambda_reg
        # M[i] = Λ_i^{-1}, shape (C, C) per env
        self.M = torch.stack([
            torch.eye(dim, device=device) / lambda_reg
            for _ in range(n_envs)
        ])  # (n_envs, C, C)

    def bonus_and_update(self, phi: torch.Tensor) -> torch.Tensor:
        """
        phi: (n_envs, C)
        Returns: bonus (n_envs,). Uses M before update.
        Side effect: updates self.M via Sherman-Morrison.
        """
        # Compute bonus using M_{t-1}
        Mphi = torch.bmm(self.M, phi.unsqueeze(-1)).squeeze(-1)  # (n_envs, C)
        bonus = (phi * Mphi).sum(dim=-1)                          # (n_envs,)
        # Sherman-Morrison update: M_t = M - (M φφ^T M) / (1 + φ^T M φ)
        denom = 1.0 + bonus                                        # (n_envs,)
        outer = Mphi.unsqueeze(-1) * Mphi.unsqueeze(-2)            # (n_envs, C, C)
        self.M = self.M - outer / denom.view(-1, 1, 1)
        return bonus

    def reset(self, env_ids):
        device = self.M.device
        I_over_lam = torch.eye(self.dim, device=device) / self.lam
        for i in env_ids:
            self.M[i] = I_over_lam
```

### 2.2 Wire into collect_rollouts
In `lmu_ppo.py collect_rollouts`, after computing (h_new, m_new):
```python
# Recompute y for episodic bonus (already computed inside forward, but
# forward returns h, m; need to extract or recompute y explicitly)
with torch.no_grad():
    C_t = F.normalize(self.policy.lmu_cell.W_query(self._lmu_h), dim=-1)
    y_t = torch.einsum('bd,bdc->bc', C_t, m_new)
    b_t = self.ep_bonus.bonus_and_update(y_t).cpu().numpy()

# Combine rewards
r_intr_masked = r_intr.cpu().numpy() * (1.0 - self._last_episode_starts)
b_t_masked    = b_t                    * (1.0 - self._last_episode_starts)
rewards_combined = (rewards
                    + self.beta_life * r_intr_masked
                    + self.beta_ep   * b_t_masked)
```
After stepping:
```python
for i, done in enumerate(dones):
    if done:
        self.ep_bonus.reset([i])
```

### 2.3 Reward normalization (critical)
Raw E3B bonuses are heavy-tailed. Normalize using a running standard deviation, same pattern RND uses:
```python
self.b_running_std.update(b_t)
b_t_normalized = b_t / (self.b_running_std.std + 1e-6)
```
Then use b_t_normalized * self.beta_ep in reward combination.

### 2.4 Hyperparameter sweep on MemoryS11 *without* MemoryStartWrapper
Target: eval reward > 0.9 within 5M steps.

Sweep:
- β_ep ∈ {0.01, 0.03, 0.1, 0.3}
- λ ∈ {0.1, 1.0, 10.0}
- β_life fixed at 0.001

Metrics:
- steps_to_solve
- b_mean, b_max trajectory
- ratio b_hint/b_corridor during training (decreases as exploration completes; this is correct)

### 2.5 Contingency: inverse dynamics auxiliary on y
If 2.4 fails with any hyperparameters:
- Add a small MLP head `f_inv: (y_t, y_{t+1}) -> a_t`
- Cross-entropy loss, weight 0.1 on PPO loss
- Retrain S11 without wrapper

---

## Phase 3 — Benchmark against literature (2–3 weeks)

### 3.1 ObstructedMaze-Full
Standard hardest MiniGrid benchmark. Published numbers for NovelD, DEIR, E3B available for direct comparison.
- Run 5 seeds with best config from Phase 2
- Target: match or beat E3B's published steps-to-solve

### 3.2 POPGym subset
Focus on memory-intensive tasks:
- Autoencode-Medium
- RepeatPrevious-Hard
- Concentration-Medium

These test pure memory capacity, which is where LMU vs. GRU/LSTM matters. Expected: LMU significantly beats GRU baselines (Morad et al. 2023 data); comparable or better than R2I's S4 (ICLR 2024).

Caveat: POPGym's reference implementations are JAX. Comparison may require re-running GRU/S5 baselines in our PyTorch pipeline.

---

## Phase 4 — HiPPO-LegS extension (1 week)

Only after Phase 3 results are published/noted. LegT is fine for the above.

### 4.1 Add LegS variant of get_AB
```python
def get_AB_legs(d: int):
    # A, B derived from scaled Legendre measure. No theta.
    i, j = np.meshgrid(np.arange(d), np.arange(d))
    A = np.where(i > j, np.sqrt((2*i+1)*(2*j+1)),
                 np.where(i == j, i+1, 0.0))
    B = np.sqrt(2 * np.arange(d) + 1)[:, None]
    return A, B  # time-varying dynamics, no ZOH at this level
```

### 4.2 Add `measure` flag to LMUCell
Keep LegT as default; LegS available via `measure='LegS'`. Forward becomes:
```python
if self.measure == 'LegT':
    Am = einsum('ij,bjc->bic', self.A, m_prev)
    Bu = self.B * u_actual.unsqueeze(1)
    m_new = Am + Bu
else:  # LegS
    k = self.step_counter.clamp(min=1)
    Am = einsum('ij,bjc->bic', self.A / k, m_prev)
    Bu = (self.B / k) * u_actual.unsqueeze(1)
    m_new = m_prev - Am + Bu
    self.step_counter += 1
```
Step counter tracked per-env, reset on episode boundary (wire through from lmu_ppo.py collect_rollouts).

### 4.3 Revalidate on Phase 3 tasks with LegS
Confirm no regression; then use LegS as default for Craftax.

---

## Phase 5 — Craftax (4+ weeks; exploratory)

With LegS + gated write + r_intr + elliptical episodic bonus.

### 5.1 Initial viability test
Craftax-Classic (2D grid version), single seed, 10M steps. Look for any non-zero extrinsic reward (achievements unlocked). If zero after 10M, stop and diagnose.

### 5.2 DEIR-style noisy-TV scaling (conditional)
If Craftax's resource spawning stochasticity creates spurious bonuses:
- Add DEIR's conditional MI scaling on top of E3B bonus
- b_t → b_t * CMI(a_t; y_{t+1} | y_t)

### 5.3 Selective decay (conditional; only if theta-scheduling fails)
If fixed LegS is still insufficient for Craftax's variable event timescales (walking vs crafting vs exploring), add Mamba-style input-dependent Δ_t. This changes LMUCell significantly; defer until other options exhausted.

---

## What we are NOT doing (and why)

- **VAE on top of LMU (HSWVIME plan).** Architectural redundancy; LMU already gives a 1-step forward model.
- **k-WMPE with constant-pred approximation.** The approximation breaks exactly when novelty is high (rapid pred dynamics at hint room). The signal would conflate novelty with approximation error. Also incompatible with later selective-decay.
- **Two-cell LMU design.** Temporal-scale separation was solving the wrong problem. The missing signal was episodic, not multi-scale.
- **Signed innovation (gate - innov).** Breaks both null conditions, creates a feedback-loop instability risk (derived mathematically).
- **Mean-pooled m for episodic novelty (LEM proposal).** Destroys Legendre order structure. Use y (readout) instead.
- **Inverse dynamics encoder from scratch (E3B's original φ).** Only if y-based bonus fails in Phase 1/2.
- **Learnable θ via gradients.** LegT is brittle; LegS removes the hyperparameter entirely.

---

## Critical metrics to track throughout

Lifetime (add to wandb/tensorboard):
- eval/mean_reward
- steps_to_solve (first eval > 0.9)
- debug/r_intr_mean, debug/r_intr_max
- debug/b_ep_mean, debug/b_ep_max
- debug/b_hint_over_corridor_ratio (during training, should start high and decay as exploration completes)
- debug/W_pre_ortho_error, debug/W_pre_grad_norm
- debug/m_norm
- debug/prod_positive_frac
- debug/u_x_zero_frac

Per-experiment:
- β_ep, λ, β_life, seed, env, chunk_len

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| y too entangled to work as E3B φ | Medium | Phase 1 diagnostic; fallback to inverse dynamics aux |
| LegT theta brittleness fails on POPGym hardest settings | Medium | Move to LegS (Phase 4) |
| Cayley updates drift W_pre off O(C) on very long runs | Low | Hard SVD reset already implemented |
| E3B bonus dominates extrinsic reward, agent only explores | Medium | Bonus normalization; β_ep sweep |
| Noisy-TV in Craftax | Medium | DEIR-style CMI scaling (5.2) |
| PyTorch slow vs JAX on long sequences | Low | Only an issue if we need parallel scan; optional port |
| r_intr vanishes too fast, episodic can't compensate | Low | r_intr + b_ep together have both local and global coverage |

---

## Definition of done per phase

- **Phase 0**: All diagnostics logging; A0 reproduces prior 1.5M-step S11 result.
- **Phase 1**: Hint/corridor b_t ratio > 3 (or > 1.5 with aux plan).
- **Phase 2**: S11 without MemoryStartWrapper solved in < 5M steps, 3 seeds.
- **Phase 3**: ObstructedMaze-Full solved within 20M steps, competitive with E3B. POPGym hard memory tasks solved.
- **Phase 4**: LegS parity with LegT on MiniGrid, advantage on POPGym long-horizon.
- **Phase 5**: Non-zero achievement count on Craftax at 10M steps.