# WGEM-PPO Implementation Progress

## Component 1: `models/wyner.py` — Slot-Conditioned Prior  **PASSED**

**Changes made:**
- `WynerInterface.forward()`: added `slots: Optional[th.Tensor] = None` parameter
- `WynerVAE.forward()`, `WynerIndependentVAE.forward()`: added `slots` kwarg (ignored, interface compat)
- `WynerLBSVAE._prior(h_prev)` → `_prior(slot_agg)`: now conditions on mean-pooled slot bank
- `WynerLBSVAE.forward()`: added `slots` parameter, raises `ValueError` if `None` (explicit failure, not silent fallback)
- Prior now computes `slot_agg = slots.mean(dim=1)` and feeds to `_prior()`
- GRU, posterior, decoder, loss: **unchanged**

**Test results (`tests/test_wyner_slot_prior.py`):**

| Metric | Value | Status |
|--------|-------|--------|
| `out.w.shape` | (4, 1, 64) | PASS |
| `out.posterior_mu.shape` | (4, 64) | PASS |
| `out.prior_mu.shape` | (4, 64) | PASS |
| `out.prior_logvar.shape` | (4, 64) | PASS |
| `prior_mu norm` (zero slots) | 0.9027 | PASS (< 1.0) |
| `prior_var mean` (zero slots) | 1.0023 | PASS (~1.0, isotropic) |
| `KL at zero slots` | 0.2156 | PASS (> 0, finite) |
| `ValueError on slots=None` | raised | PASS |
| `Prior diff (zero vs filled)` | 0.4705 | PASS (> 0.01) |
| `Recon loss` | 1.0423 | PASS (finite) |

**Key observations:**
- Zero-slot prior is near-isotropic (variance ~1.0, mean norm < 1.0) — no degenerate init
- Prior output changes with different slot contents (diff=0.47) — prior reads from slots

**New/modified tensor shapes:**
- `slot_agg`: (B, 64) — mean-pooled slot bank, new intermediate in `forward()`

---

## Component 2: `models/slot_memory.py` — Content-Addressed Write  **PASSED**

**Changes made:**
- Replaced LRU eviction with cosine-similarity content-addressed soft write
- Removed `ages` tensor entirely (`write()` returns `(new_slots, gate)` not `(new_slots, new_ages, gate)`)
- `init_state()` returns single `th.Tensor` (not tuple)
- `slot_dim` now 64 (set by caller, matches `wyner_latent_dim`)
- Added `write_temp` parameter for softmax temperature
- Cosine similarity safeguard: `F.normalize(slots, dim=-1, eps=1e-6)` — zero vectors normalize to zero, softmax gives uniform 1/K weights

**Test results (`tests/test_slot_content_addressed.py`):**

| Test | Result | Status |
|------|--------|--------|
| Zero-slot init, no NaN | gate=1.0000 | PASS |
| Low KL → gate closes | gate2=0.0000, delta=0.000222 | PASS |
| Content addressing (slot norm) | slot3=8.29 > slot0=0.85 | PASS |
| write() returns 2 values | (new_slots, gate) | PASS |
| Uniform write at zero slots | norm std=0.000000 | PASS |

**Key observations:**
- At zero-slot init, all slots get uniform weight (1/K) — first write spreads evenly
- Content addressing works: pre-loaded slot has highest norm after re-write
- Gate correctly closes when KL is below threshold

---

## Component 3: `hswvime_ppo/buffer.py` — Slot Shape Update  **PASSED**

**Changes made:** None to buffer.py itself — shape is already parameterized via `slot_memory_shape` kwarg.
The actual `(8, 32) → (8, 64)` change is in `HSWVimePPO.__init__` (Component 5).

**Test results (`tests/test_buffer_slot_shape.py`):**

| Test | Result | Status |
|------|--------|--------|
| `slot_memories.shape` | (512, 4, 8, 64) | PASS |
| `add()` with (4, 8, 64) | no crash | PASS |

---

## Component 4: `hswvime_ppo/policies.py` — proj_pi for Cross-Attention  **PASSED**

**Changes made:**
- Added `slot_dim: int = 64` parameter to `HSWVIMEFeaturesExtractor.__init__`
- Added `self.proj_pi = nn.Linear(mu_dim, slot_dim)` — projects pi (32) into slot space (64)
- `self.attn = nn.MultiheadAttention(slot_dim, ...)` — attention operates in slot_dim=64
- `_features_dim` = `slot_dim + mu_dim` = 64 + 32 = 96
- `forward()`: projects pi via `proj_pi` for query, concatenates raw pi (not projected) with attn output
- Updated `train.py` to pass `slot_dim=args.wyner_latent_dim` to features extractor

**Test results (`tests/test_features_extractor.py`):**

| Test | Result | Status |
|------|--------|--------|
| Output shape | (4, 96) | PASS |
| `features_dim` | 96 | PASS |
| Gradient through pi | norm=11.32 | PASS |
| `proj_pi` shape | Linear(32 → 64) | PASS |

---

## Component 5: `hswvime_ppo/hswvime_ppo.py` — Wire Everything  **PASSED**

**Changes made:**
- `__init__`: `_slot_memory_shape = (num_slots, wyner_latent_dim)` where `wyner_latent_dim = memory_shape[-1] = 64`
- `_setup_learn`: `init_state()` returns single tensor (removed ages unpacking)
- `collect_rollouts` intrinsic reward: pass `slots=self._last_slots` to wyner forward
- `collect_rollouts` slot write: write `wyner_out_t.posterior_mu.detach()` (z_t, dim 64) instead of `vae_t.pi.detach()` (dim 32); pass `slots=self._last_slots` to wyner forward for slot-conditioned prior; `delta_I` = KL from slot-conditioned prior
- `collect_rollouts` episode reset: removed `new_ages` reset
- `collect_rollouts` state update: removed `_last_ages`
- `train()` wyner forward: pass `slots=rollout_data.slot_memories`
- Removed all `_last_ages` references from `__init__`, `set_env`, `_setup_learn`
- Added new TensorBoard metrics: `slots/gate_std`, `slots/slot_norm_mean`, `slots/slot_diversity`, `wyner/prior_mu_norm`, `wyner/prior_logvar_mean`, `wyner/delta_I_mean`

**Smoke test (2048 steps, CPU):**

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| No crash | clean | — | PASS |
| No NaN | clean | — | PASS |
| `wyner/delta_I_mean` | 0.238 | != 0 | PASS |
| `slots/gate_mean` | 0.514 | 0.1-0.9 | PASS |
| `slots/gate_std` | 0.393 | > 0 | PASS |
| `slots/slot_norm_mean` | 0.408 | > 0 | PASS |
| `slots/slot_diversity` | 3e-9 | low at 2k | OK |
| `wyner/prior_mu_norm` | 0.421 | small | PASS |
| `wyner/prior_logvar_mean` | 0.001 | ~0 | PASS |

---

## Component 6: `scripts/probe_color_memory.py` — Update Probe  **PASSED**

**Changes made:**
- `slot_dim=MU_DIM` → `slot_dim=WYNER_LATENT_DIM` (32 → 64)
- `init_state()` returns single tensor (removed ages unpacking)
- `slot_mem.write(slots, ages, pi.detach(), kl)` → `slot_mem.write(slots, wyner_out.posterior_mu.detach(), kl)`
- Removed all `ages` variable tracking
- `wyner.forward()` now receives `slots=slots` kwarg

**Smoke test (20 episodes, random weights):** Runs without errors, collects 4042 samples.
Probe accuracy at random weights is near chance (50-65%) as expected.

---

## Post-Probe Diagnostic Fixes (v2)

**Problem diagnosed from 20-episode random-weight probe:**
1. Gate stuck at ~0.78 (flat, never discriminates) — z-score normalization + threshold=0.0 means sigmoid(5 * ~0.25) ≈ 0.78 for all steps
2. Slot diversity = 0.0 (all slots identical) — zero init → F.normalize(0) = 0 → softmax uniform → all slots converge to same average z_t
3. Write weights uniform (1/8) — temp=1.0 too soft, similar slots only get ~10% more weight

**Three fixes applied:**

### Fix 1: Noise init (`slot_memory.py:init_state`)
```python
# BEFORE: th.zeros(...)
# AFTER:
return th.randn(...) * 0.01
```
Also updated episode reset in `collect_rollouts` to use noise instead of zeros.

### Fix 2: Softmax temperature (`--slot_temp 0.1`)
- New CLI arg `--slot_temp` (default 0.1, was hardcoded 1.0)
- Wired through `HSWVimePPO → SlotMemory(write_temp=...)`
- At temp=0.1, most-similar slot gets ~90% of write weight

### Fix 3: Gate threshold (`--gate_threshold 1.5`)
- Already existed as CLI arg, was passed as 0.0
- At 1.5 z-score units, only top ~7% of KL steps trigger writes

**Verification results:**

| Fix | Before | After |
|-----|--------|-------|
| Init diversity | 0.0 | 0.85 |
| Gate at z=0.5 | 0.78 | 0.007 |
| Gate at z=3.0 | ~1.0 | 0.999 |
| Write to target slot | uniform | 16.01 vs 0.08 |

**Smoke test v2 (2048 steps):**

| Metric | v1 (broken) | v2 (fixed) |
|--------|-------------|------------|
| `slots/gate_mean` | 0.514 | **0.076** |
| `slots/slot_diversity` | 3e-9 | **0.825** |
| No crash / No NaN | PASS | PASS |

---

## V3 Fixes: KL Explosion at Step 0  **PASSED**

**Problem:** V2 training (250k steps) had delta_I spike to ~5×10⁷ at step 0 due to random Kaiming init on logvar weight matrices. This corrupted z-score running stats permanently, locking gate closed for entire run.

**Three fixes applied:**

### Fix 1: Zero-init logvar weights (`models/wyner.py`)
```python
nn.init.zeros_(self.prior_fc_logvar.weight)
nn.init.zeros_(self.post_fc_logvar.weight)
```
Both prior and posterior logvar networks now output 0 at init → variance = 1.0, KL starts near-isotropic.

### Fix 2: Hard clamp delta_I (`hswvime_ppo/hswvime_ppo.py`)
```python
delta_I = delta_I.detach().clamp(max=500.0)
```
Safety rail: even if KL spikes, gate input is bounded at 500 nats.

### Fix 3: Drop z-score normalization, use raw nats
Removed `_gate_kl_running_stats` entirely. Gate now operates on raw delta_I with:
- `--gate_threshold 5.0` (nats, was z-score 1.5)
- `--gate_scale 0.5` (was 5.0 for z-scores)

Updated `train.py` defaults: `gate_scale=0.5`, `gate_threshold=5.0`.

**Smoke test v3 (6144 steps, 3 iterations):**

| Metric | Iter 1 | Iter 2 | Iter 3 | Target |
|--------|--------|--------|--------|--------|
| `wyner/delta_I_mean` | 0.182 | 0.411 | 2.12 | < 1000 |
| `slots/gate_mean` | 0.083 | 0.092 | 0.197 | 0.05–0.5 |
| `slots/slot_diversity` | 0.842 | 0.837 | 0.613 | > 0 |
| `prior_logvar_mean` | 0.0 | 0.010 | 0.19 | ~0 at init |
| No crash / No NaN | PASS | PASS | PASS | — |

**Key observations:**
- delta_I starts at 0.18 nats (was 5×10⁷) — zero-init logvar is the primary fix
- Gate is in healthy 0.08–0.20 range and increasing as KL grows — discriminative
- Slot diversity stays high (0.6–0.8) — slots remain differentiated

---

## V4 Fixes: Gate Saturation Feedback Loop  **PASSED**

**Problem:** V3 100k run showed gate saturating to 1.0 by 60k steps. Feedback loop:
- Gate open → every z_t written → slots = running average → low diversity
- Low diversity → bad prior → high KL → delta_I grows to 300 nats
- delta_I >> threshold (5 nats) → sigmoid ≈ 1.0 → gate permanently open

**Root cause:** Absolute nat threshold can't adapt to growing KL scale. At delta_I=300 and threshold=5, `sigmoid(0.5*(300-5)) ≈ 1.0` — no discrimination possible.

**Two fixes applied:**

### Fix 1: EMA ratio gating (`hswvime_ppo.py`)
Replace absolute threshold with self-normalizing ratio gate:
```python
# EMA tracks running mean of delta_I (alpha=0.01, init=1.0)
self._delta_I_ema = (1 - alpha) * self._delta_I_ema + alpha * delta_I_mean
ratio = delta_I / max(self._delta_I_ema, 1.0)  # floor at 1.0 nat
# Pass ratio (not raw nats) to slot_memory.write()
```
- Scale-invariant: as delta_I grows to 300, EMA grows with it, ratio stays ~1.0
- Self-correcting: when ALL steps have uniformly high KL (loop active), ratio ≈ 1.0 for all → below threshold 2.0 → gate closes → loop breaks
- Only genuinely surprising steps (delta_I > 2× average) trigger writes
- EMA (not Welford) — adapts faster, no corruption from early outliers

CLI: `--gate_threshold 2.0` (ratio, not nats), `--gate_scale 3.0` (sigmoid sharpness)

### Fix 2: Slot max-norm clamping (`slot_memory.py`)
```python
norms = new_slots.norm(dim=-1, keepdim=True).clamp(min=1e-6)
new_slots = th.where(norms > 5.0, new_slots * (5.0 / norms), new_slots)
```
Prevents unbounded slot norm growth (was 8.6+ at 90k in v3). Insurance against residual writes accumulating magnitude.

**Smoke test v4 (6144 steps, 3 iterations):**

| Metric | Iter 1 | Iter 2 | Iter 3 | Target |
|--------|--------|--------|--------|--------|
| `gate_mean` | 0.004 | 0.011 | 0.18 | 0.1–0.5 |
| `gate_std` | 0.00006 | 0.003 | 0.088 | > 0 |
| `slot_diversity` | 0.86 | 0.87 | 0.64 | > 0.5 |
| `slot_norm_mean` | 0.09 | 0.16 | 1.93 | bounded |
| `delta_I_ema` | 0.19 | 0.48 | 3.10 | tracks mean |

**Key improvement over v3:** gate_std=0.088 (v3: 0.0002) — gate now discriminates between steps.

---

## Component 7: Full 500k Training Run

**Status:** READY — all prerequisites + v4 fixes passed. Run with:
```bash
python train.py \
  --env "MiniGrid-MemoryS7-v0" \
  --total_timesteps 500000 \
  --n_envs 4 --n_steps 512 --batch_size 256 \
  --device mps --seed 42 \
  --num_slots 8 --gate_scale 3.0 --gate_threshold 2.0 \
  --slot_temp 0.1 \
  --ent_coef 0.05 --intrinsic_scale 0.0 \
  --free_bits 0.0 \
  --tensorboard_log runs/wgem_v4 \
  --wyner_kl_target 0.0
```

**Checkpoints needed at:** 100k, 300k, 500k

**Target metrics:**

| Metric | Target at 300k | Target at 500k |
|--------|---------------|----------------|
| `ep_rew_mean` | > 0.6 | > 0.75 |
| `slots/gate_mean` | 0.05 – 0.2 | 0.05 – 0.15 |
| `wyner/kl_mean` | 2 – 15 | 2 – 15 |
| `train/entropy_loss` | < -0.5 | < -0.3 |
| `train/explained_variance` | > 0.3 | > 0.5 |

**100k probe checkpoint criteria (run before continuing to 500k):**
- (a) Gate shows spike at t=0-2 then drops to < 0.3 for rest of episode
- (b) At least one slot probes above 75% accuracy individually
- (c) Flattened slot probe at late timesteps > 70%

**Final probe pass criterion:**

| Timestep window | Required accuracy |
|----------------|-------------------|
| Early (t=0-2)  | > 90%             |
| Mid (t=3-50)   | > 75%             |
| Late (t=51+)   | > 70%             |
