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

## V4: Full 500k Training Run

**Run:** `runs/wgem_v4` — 500k steps, seed=42, ent_coef=0.05, gate_scale=3.0, gate_threshold=2.0

**Results:**

| Metric | Value |
|--------|-------|
| Final reward (500k) | 0.549 |
| Peak reward | **0.620 @ 291k** |
| `slots/gate_mean` | 0.051 |
| `slots/slot_norm_mean` | 1.28 |
| `slots/slot_diversity` | 0.34 |

**Observations:**
- Best result to date at the time (peak 0.62)
- Gate discriminates (0.05 mean, not saturated)
- Slot diversity decays over training (0.85 → 0.34) — slots converging
- Reward oscillates 0.45–0.62 after 200k, never sustains above 0.6

---

## V5: Signal Visibility Fix

**Problem diagnosed:** `MemorySignalVisibleWrapper` was consuming turn-around observations without exposing them to the RL loop. The agent never "saw" the key/ball signal during rollout.

**Fix:** Added turn-around observation processing in `_process_turn_history_all_envs()` and mid-rollout episode resets. Turn history from the wrapper is replayed through VAE/Wyner/slots at episode start.

**Run:** `runs/wgem_v5` — 500k steps

**Results:**

| Metric | Value |
|--------|-------|
| Final reward (500k) | 0.443 |
| Peak reward | 0.607 @ 444k |
| `slots/gate_mean` | 0.050 |
| `slots/slot_norm_mean` | 0.83 |
| `slots/slot_diversity` | 0.54 |

**Observations:**
- Worse than v4 despite fixing signal visibility
- Probe investigation revealed: the signal IS in the pipeline, but mu (after L2 norm) kills it

---

## V6: Signal Bottleneck Diagnosis

**Diagnostic script:** `scripts/diagnose_signal_bottleneck.py` — tests linear separability at 7 pipeline stages.

**Probe results (key/ball discrimination accuracy):**

| Stage | Random Weights | Trained Weights |
|-------|---------------|-----------------|
| Raw obs | ~50% | ~50% |
| Embedding | 76% | 76% |
| Conv encoder | **89%** | **89%** |
| Trunk (fc) | 72% | 68% |
| fc_mu (before L2 norm) | 61% | 61% |
| mu (after L2 norm) | **50%** | **50%** |
| pi (unnormalized) | 61% | 61% |

**Key finding:** L2 normalization (required by spCauchy prior) destroys the key/ball signal. Conv encoder features retain 89% but the VAE bottleneck (256 → 32 → L2 norm) compresses it to chance. Training actively destroys signal — random weights preserve it better than trained weights because VAE reconstruction allocates capacity to high-variance features (corridor layout), not the low-variance signal (1 cell key/ball).

**Run:** `runs/wgem_v6` — 100k steps (short diagnostic run)

| Metric | Value |
|--------|-------|
| Final reward (100k) | 0.446 |
| Peak reward | 0.538 @ 78k |

---

## V7: Conv Features → Slots (Bypass VAE Bottleneck)

**Approach:** Write conv encoder features (128-dim, 89% signal) directly to slots instead of posterior_mu (64-dim, inherits dead mu). Use frozen random projection (JL lemma) from 128 → 64 to match slot_dim.

**Changes:**
- Added `encode_obs_conv()` method to `TransitionSCVAE` — exposes conv encoder output
- Added `_slot_proj = nn.Linear(128, 64, bias=False)` with `requires_grad_(False)` — frozen random projection
- Slot writes during turn-around use `_slot_proj(encode_obs_conv(s_curr).detach())` instead of `posterior_mu`

**Run:** `runs/wgem_v7_convslots` — 100k steps

| Metric | Value |
|--------|-------|
| Final reward (100k) | 0.400 |
| Peak reward | 0.435 @ 74k |
| `slots/gate_mean` | 0.089 |
| `slots/slot_norm_mean` | 2.03 |

**Probe results (v7 conv-slot probe):**
- Slots at turn-around: **73.5%** (up from ~55% with posterior_mu)
- Slots at late timesteps: **58.4%** (signal decaying during forward movement)
- Gate mean 0.029 — small but nonzero writes at every forward step accumulate and overwrite turn-around signal

---

## V7b: Tight Gate (gate_threshold=8.0)

**Hypothesis:** Higher gate threshold would prevent forward-step writes from eroding signal.

**Run:** `runs/wgem_v7b_tightgate` — 8k steps (aborted)

**Result:** Gate completely shut. `sigmoid(5.0*(1.0 - 8.0)) ≈ 0`. Ratio gating normalizes to ~1.0, so threshold=8 means "8× average surprise" which never happens. Slots stayed at noise initialization.

---

## V7c: Frozen Slots After Turn-Around

**Approach:** Instead of tuning the gate, freeze slots entirely during forward movement. Only write during turn-around replay. This guarantees the signal is preserved.

**Changes:**
- Forward movement: `new_slots = self._last_slots` (no write, no gate)
- Turn-around: normal conv-feature write via `_slot_proj`
- Added `--freeze_slots` flag to probe script for consistency

**Probe results (v7c):**

| Window | Accuracy |
|--------|----------|
| Turn-around (early) | **87–88%** |
| Late timesteps | **87–88%** (constant — no erosion) |

Signal fully preserved! But reward didn't improve:

**Run:** `runs/wgem_v7c_frozenslots` — 100k steps

| Metric | Value |
|--------|-------|
| Final reward (100k) | 0.485 |
| Peak reward | 0.485 @ 100k |
| `slots/gate_mean` | 0.000 (frozen) |
| `slots/slot_norm_mean` | 0.37 |

**Run:** `runs/wgem_v7c_500k` — 500k intended, died at 76k

| Metric | Value |
|--------|-------|
| Final reward (76k) | 0.454 |
| Peak reward | 0.471 @ 72k |

**Diagnosis:** Slots contain signal (87% probe) but policy can't use it. Cross-attention with uniform slot content (all 8 slots identical) is degenerate — attention weights are irrelevant. `proj_pi` has too sparse a learning signal (1–2 T-junction steps per episode).

---

## V8: Mean Pooling (Replace Cross-Attention)

**Approach:** Replace `nn.MultiheadAttention` + `proj_pi` with simple `mean(slots)` concatenated with pi. Rationale: all 8 slots contain identical content (same turn-around observation), so attention weights don't matter. Mean pooling removes unnecessary learned indirection.

**Changes in `HSWVIMEFeaturesExtractor`:**
- Removed: `self.proj_pi`, `self.attn` (nn.MultiheadAttention)
- `forward()`: `slot_mean = slots.mean(dim=1); features = cat(pi, slot_mean)`
- `_features_dim` = slot_dim (64) + mu_dim (32) = 96

**Run:** `runs/wgem_v8_meanpool` — 200k steps

| Step | Reward |
|------|--------|
| 20k | 0.13 |
| 50k | 0.37 |
| 80k | **0.57** (first time above 0.5) |
| 100k | 0.47 |
| 149k | **0.586** (peak) |
| 200k | 0.44 |

| Metric | Value |
|--------|-------|
| `slots/gate_mean` | 0.000 (frozen) |
| `slots/slot_norm_mean` | **0.14** |

**Key finding:** Slot norms collapsed from ~2.5 at 43k to ~0.12 by 63k. The frozen random projection doesn't adapt as the conv encoder's feature distribution shifts during VAE training. Policy was effectively blind to slots after 60k.

---

## V9: L2 Normalization on Projected Slot Content

**Fix:** Added `F.normalize(self._slot_proj(conv_features), dim=-1)` so slot content is unit-norm regardless of conv encoder scale drift.

**Run:** `runs/wgem_v9_normed_slots` — 200k steps

| Step | Reward |
|------|--------|
| 20k | 0.17 |
| 50k | 0.54 |
| 82k | **0.605** (peak) |
| 200k | 0.57 |

| Metric | Value |
|--------|-------|
| `slots/slot_norm_mean` | **0.08** |

**Problem:** Despite L2 normalizing the *content*, the *slot states* had norm 0.08. The gate was still closed (gate_mean=0.0) — turn-around writes also went through SlotMemory.write() which applies `sigmoid(3.0 * (ratio - 2.0))` as gate. Turn ratio ≈ 1.0 → gate ≈ sigmoid(-3.0) ≈ 0.05. Almost no content actually written.

---

## V9b: Broadcast Write (Bypass SlotMemory.write)

**Fix:** Replace content-addressed gated write during turn-around with direct broadcast:
```python
turn_slots = slot_content.unsqueeze(1).expand_as(turn_slots).clone()
```
This writes unit-norm content to ALL 8 slots, so `mean(slots) = slot_content` (full signal).

**Run:** `runs/wgem_v9b_broadcast_slots` — 200k steps

| Step | Reward |
|------|--------|
| 20k | 0.44 |
| 40k | 0.52 |
| 60k | 0.52 |
| 80k | 0.46 |
| 115k | **0.577** (peak) |
| 200k | 0.54 |

| Metric | Value |
|--------|-------|
| `slots/slot_norm_mean` | **1.00** (stable) |
| `slots/gate_mean` | 0.000 (frozen during forward) |
| `slots/slot_diversity` | 0.87 |

**Observations:**
- Slot norms finally stable at 1.0 throughout training
- Fastest ramp (0.44 at 20k vs 0.13 for v8)
- Still plateaus at 0.45–0.58 — same ceiling as all prior versions

---

## Summary: All Versions

| Version | Key Change | Steps | Peak Reward | Final Reward | Slot Norm |
|---------|-----------|-------|-------------|--------------|-----------|
| v4 | EMA ratio gate, norm clamp | 500k | **0.620** | 0.549 | 1.28 |
| v5 | Signal visibility fix | 500k | 0.607 | 0.443 | 0.83 |
| v6 | Diagnostic (bottleneck probe) | 100k | 0.538 | 0.446 | — |
| v7 | Conv features → slots | 100k | 0.435 | 0.400 | 2.03 |
| v7b | Tight gate (threshold=8) | 8k | 0.127 | 0.092 | — |
| v7c | Frozen slots after turn-around | 100k | 0.485 | 0.485 | 0.37 |
| v8 | Mean-pool (replace attention) | 200k | 0.586 | 0.442 | 0.14 |
| v9 | L2 norm on slot content | 200k | **0.605** | 0.567 | 0.08 |
| v9b | Broadcast write to all slots | 200k | 0.577 | 0.538 | **1.00** |

**Conclusion:** All versions plateau at 0.45–0.62 regardless of representation fixes. The signal is verified at every stage (probes show 85–89% accuracy), slot norms are stable, yet the policy cannot learn to use the information. The 0.5 ceiling suggests the bottleneck is NOT the representation — it may be PPO credit assignment on this 50+ step horizon.

**Next step:** Run a GRU-PPO baseline to determine if PPO can solve MemoryS7 at all (see pivot plan).
