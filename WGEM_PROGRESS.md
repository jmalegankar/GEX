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

## Component 7: Full 500k Training Run

**Status:** READY — all prerequisites passed. Run with:
```bash
python train.py \
  --env "MiniGrid-MemoryS7-v0" \
  --total_timesteps 500000 \
  --n_envs 4 --n_steps 512 --batch_size 256 \
  --device mps --seed 42 \
  --num_slots 8 --gate_scale 5.0 --gate_threshold 0.0 \
  --ent_coef 0.05 --intrinsic_scale 0.0 \
  --free_bits 1.0 \
  --tensorboard_log runs/wgem_v1 \
  --wyner_kl_target 0.0
```

**Checkpoints needed at:** 100k, 300k, 500k

**Target metrics:**

| Metric | Target at 300k | Target at 500k |
|--------|---------------|----------------|
| `ep_rew_mean` | > 0.6 | > 0.75 |
| `slots/gate_mean` | 0.1 – 0.4 | 0.1 – 0.3 |
| `wyner/kl_mean` | 2 – 15 | 2 – 15 |
| `train/entropy_loss` | < -0.5 | < -0.3 |
| `train/explained_variance` | > 0.3 | > 0.5 |

**Probe pass criterion:**

| Timestep window | Required accuracy |
|----------------|-------------------|
| Early (t=0-2)  | > 90%             |
| Mid (t=3-50)   | > 75%             |
| Late (t=51+)   | > 70%             |
