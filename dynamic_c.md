# LMU-PPO Memory Research — Full Record & Forward Plan

---

## Part 1: What We Built

### Architecture

```
obs_t → MinigridEncoder (CNN + embeddings)
      → LMUCell (multichannel, dynamic W_query)
      → (h_t, m_t)
      → cat([h_t, m_pooled]) → actor + critic
```

**MinigridEncoder:** Separate embeddings for object/color/state/direction, 3-layer CNN, linear projection to `encoder_dim=64`. Orthogonal init throughout.

**LMUCell (multichannel):** S4-style parallel LMUs sharing frozen (Ā, B̄). One scalar u per channel, C independent Legendre trajectories. `h: (B, n)`, `m: (B, d, C)`. Dynamic W_query read head (Step 1).

**Actor + Critic:** Both receive `cat([h, m.mean(dim=1)])` directly — bypasses the C_proj bottleneck that caused early collapse. Critic is a 2-layer MLP with Tanh. Actor is linear with gain=0.01 init for near-uniform initial policy.

**Training:** Full BPTT (chunk_len = episode length), TBPTT buffer storing (h, m) at chunk boundaries, episode boundary masking via `h * (1 - reset)`. `n_envs=16`, `lr=3e-4`, `gamma=0.999`, `gae_lambda=0.98`.

---

## Part 2: What We Tried, What Worked, What Failed

### Fix 1 — Canonical LMU (Worked)
**Problem:** Original `lmu.py` replaced Legendre ODEs with MultiheadAttention. Broke mathematical guarantees entirely.  
**Fix:** Reverted to canonical Voelker 2019 equations. u_t must be scalar per channel — vectorizing it destroys the Legendre basis decomposition.

### Fix 2 — Multichannel LMU (Worked)
**Problem:** Scalar bottleneck — all C=64 encoder features compressed to one number before writing to memory.  
**Fix:** Run C independent scalar LMUs in parallel sharing (Ā, B̄). `m: (B, d, C)`, `y = einsum('d,bdc->bc', C_proj, m)`.  
**Why it worked:** Object identity, color, and position can now be tracked independently in memory.

### Fix 3 — Full TBPTT (Worked)
**Problem:** Single-step gradient. Encoders only learned "was this step's write useful right now" — never learned that the ball encoding from step 1 must be retrievable at step 40.  
**Fix:** Buffer stores (h, m) at chunk boundaries. `evaluate_actions` unrolls K steps with full gradient. Episode boundaries zero state via `h * (1 - reset)` — no gradient bleeds across episodes.

### Fix 4 — Direct Memory Access for Actor + Critic (Worked)
**Problem:** `explained_variance ≈ 0`. Critic received only h. Early in training C_proj is random so h carries no memory signal. C_proj then collapsed to ~0.04 (from ±0.177 init) because Adam shrunk it when gradient through `C_proj → W_m → h` was noise.  
**Fix:** Both actor and critic receive `cat([h, m.mean(dim=1)])` directly. C_proj + W_m remain as auxiliary pathway but primary signal bypasses the bottleneck.

### Fix 5 — Entropy Coefficient Scaling (Worked)
**Problem:** `ent_coef=0.05` worked on S7 but destroyed S11. Entropy bonus scales with episode length: `0.05 × 1.65 × 50 = 4.1` vs sparse reward ~0.95.  
**Rule:** `ent_coef ≈ reward / (entropy × T) × 0.25`. S11: ~0.01. S13: ~0.008.

### Fix 6 — Spawn Wrapper (Worked, Partially)
**Problem:** MiniGrid Memory spawns agent at random corridor position. On many spawns the hint ball is behind the agent's FOV and never observed.  
**Fix:** `MemoryStartWrapper` forces spawn at `x=1, dir=2` (facing west into hint room). Note: `dir=0` (east) was initially implemented — this is **wrong** for large grids. For S13, the hint object is outside the eastward FOV from x=1, causing the flat value_loss=0.165 failure observed in early S13 runs.

### Step 1 — Dynamic W_query Read Head (Worked, Key Contribution)
**Problem:** Fixed `C_proj ∈ ℝ^d` reads the same Legendre frequency combination at every timestep regardless of behavioral context. At the junction the agent needs a specific lag readout; mid-corridor it needs nothing. The fixed projection is a compromise that serves neither phase well.  
**Fix:** Replace `C_proj` with `W_query: Linear(hidden_size, memory_size, bias=False)`.
```python
C_t = self.W_query(h_prev)                        # (B, d)
C_t = F.normalize(C_t, dim=-1)                    # unit norm
y   = torch.einsum('bd,bdc->bc', C_t, m_new)      # (B, C)
```
**Critical implementation details:**
- `gain=0.01` on orthogonal init — standard gain caused grad_norm=2.4×10⁹ on first update, permanent weight corruption
- `F.normalize` — bounds gradient through einsum regardless of W_query magnitude, makes query interpretable as unit direction in Legendre space
- `bias=False` — ensures `W_query(h=0) = 0`, satisfying the e_m=0 analog from Voelker §3

**Results:**

| Env | Fixed C_proj | Dynamic W_query | Speedup |
|---|---|---|---|
| MemoryS11 | ~2.0M steps | ~1.5M steps | ~1.3x |
| MemoryS13 | ~5.5M steps | ~2.0M steps | **~2.75x** |

The speedup scales with corridor length — the core finding. Fixed C_proj sample complexity grows superlinearly with environment size. Dynamic W_query stays near-linear.

### Step 2 — Separate Read/Write Queries (Failed, Informative)
**What we tried:** Replaced both `C_proj` (read) and `e_m` (write) with dynamic networks `W_query_read` and `W_query_write`.  
**Result:** Regression — phase transition delayed to >2.4M steps on S11. approx_kl spike to 0.16, oscillating value_loss.  
**Why it failed:** The original architecture already had effective read/write separation: `e_m` initialized to 0 (Voelker §3) gives a static, silent write path that learns gradually. Making both paths dynamic simultaneously couples them to h before either has learned meaningful structure. The optimization instability is from two competing dynamic signals early in training.  
**Lesson:** Step 1 is the correct implementation. Dynamic read, static write.

### Interpretation B — Explicit Lag Reconstruction (Failed)
**What we tried:** Colleague's implementation — h generates explicit scalar lag values t ∈ (0,1), signal reconstructed at those lags via Voelker Eq. 3. Read path generates `hidden_size` different lags simultaneously.  
**Result:** Never converged on S11. Reward flatlined at ~0.4-0.5 for full 2M step run. value_loss permanently flat — same signature as broken S13 run.  
**Why it failed:** Gradient path from junction decision → h → Conv1d extractor → lag → reconstruction → memory is much longer and more nonlinear than Step 1's single linear layer. Gradient vanishes through the Conv1d → Sigmoid chain before the extractor learns meaningful lags. Additionally, the `multi_timestep_extractor` had a P_0 initialization bug (P_0=1 only at index 0 instead of all positions) that corrupted the Legendre recurrence for num_timesteps > 1.  
**Note:** Even with bugs fixed, the theoretical motivation is questionable — the Legendre basis is already an optimal polynomial approximation of history (HiPPO theorem). Querying it in time space is a change of basis on top of an already-optimal representation, adding computation without additional expressiveness on this task.

---

## Part 3: Full Results Table

| Env | Architecture | Steps to Solve | Notes |
|---|---|---|---|
| MemoryS7 | Fixed C_proj | ~0.3M | Baseline |
| MemoryS9 | Fixed C_proj | ~0.5M | Baseline |
| MemoryS11 | Fixed C_proj | ~2.0M | Baseline |
| MemoryS13 | Fixed C_proj | ~5.5M | approx_kl spike 4×10⁹ at transition |
| MemoryS11 | Dynamic W_query | ~1.5M | Step 1 ✓ |
| MemoryS13 | Dynamic W_query | ~2.0M | **4x improvement over fixed** |
| MemoryS11 | Separate read+write | >2.4M | Regression — Step 2 ✗ |
| MemoryS11 | Interp B (lag recon) | Never | Architecture failure ✗ |

---

## Part 3b: Final Validated Configuration

**This is the configuration to carry forward into Phase 2.**

```python
# lmu.py — LMUCell
from torch.nn.utils import spectral_norm

self.W_h = spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False))
self.E_h = spectral_norm(nn.Linear(hidden_size, input_size,  bias=False))
self.W_query = nn.Linear(hidden_size, memory_size, bias=False)
nn.init.orthogonal_(self.W_query.weight, gain=0.01)

# forward Step 3 (read path):
C_t = self.W_query(h_prev)
C_t = F.normalize(C_t, dim=-1)
y   = torch.einsum('bd,bdc->bc', C_t, m_new)
```

```python
# lmu_ppo.py — LMUPPO constructor
target_kl = 0.05       # mandatory — prevents early policy instability
max_grad_norm = 0.5    # unchanged
```

```bash
# train.py — S13 command
python train.py \
    --env MemoryS13 \
    --seed 0 \
    --n_envs 16 \
    --total_steps 6_000_000 \
    --n_steps 512 \
    --n_chunks_per_batch 32 \
    --n_epochs 4 \
    --lr 3e-4
# NOTE: no --full_bptt flag — truncated K=32 is correct for dynamic W_query
```

**Result:** Converges ~800k steps, stable at 0.985 through 6M steps. value_loss → 0, entropy → 0, explained_variance → 1.0. No resets.

**What not to change before Phase 2:**
- Do not re-add `--full_bptt` — truncated BPTT is faster and more stable with dynamic W_query
- Do not remove `target_kl=0.05` — it prevents early transition instability
- Do not remove spectral norm — W_h drift causes catastrophic reset at ~4M without it
- Do not remove `F.normalize(C_t)` — bounds gradient through the einsum regardless of W_query magnitude

---

## Part 4: Forward Plan — VIME Integration

### Long-Term Architecture

```
obs_t → Encoder
      → LMUCell (dynamic W_query)     ← current ✓
      → s_t = cat([h_t, m_pooled])
      → VAE → z_t                     ← Phase 2
      → r_int = KL(posterior||prior)
      → PPO (r_ext + β·r_int)
```

---

### Phase 1.5: Truncated BPTT Ablation on S13 (Before Phase 2)

**Goal:** Determine how much of the 2.75x speedup comes from dynamic W_query vs gradient length. Critical for Craftax where full BPTT over long episodes is computationally expensive.

 matches B, truncated BPTT is safe to use in Craftax without sacrificing the architecture benefit.

```bash
# Run C — fixed C_proj, truncated (no --full_bptt flag)
python train.py --env MemoryS13 --seed 0 --n_envs 16 \
    --total_steps 4_000_000 --n_steps 512 \
    --n_chunks_per_batch 32 --n_epochs 4 --lr 3e-4

# Run D — dynamic W_query, truncated
python train.py --env MemoryS13 --seed 0 --n_envs 16 \
    --total_steps 4_000_000 --n_steps 512 \
    --n_chunks_per_batch 32 --n_epochs 4 --lr 3e-4
```

**Interpretation guide:**
- C fails, D solves → full BPTT was load-bearing for fixed C_proj; W_query is robust
- C matches A (~5.5M), D matches B (~2M) → speedup is purely architectural; gradient length irrelevant
- C matches A, D fails → W_query requires long gradient chains to learn lag queries; Craftax will need full BPTT or very long chunks
- D matches B → truncated BPTT is safe for Phase 4 Craftax

---

### Phase 2: Gaussian VIME on top of working backbone

**Goal:** Verify the VAE trains and intrinsic reward doesn't destabilize PPO. Keep mem_start ON.

**Why Gaussian, not spCauchy yet:** spCauchy adds β annealing, posterior collapse risk on new state space, and curvature monitoring. Debugging any of these on top of a new backbone simultaneously is too many unknowns. Gaussian VAE fails loudly (KL→0, reconstruction explodes) — clean diagnostic signal.

**Implementation:**
```python
# VAE sits on top of _critic_input output
s_t   = cat([lmu_h[t], lmu_m[t].mean(dim=1)], dim=-1)  # (B, 192)
s_t1  = cat([lmu_h[t+1], lmu_m[t+1].mean(dim=1)], dim=-1)

# During collect_rollouts:
r_int = vae.information_gain(s_t, action, s_t1)  # KL posterior||prior
rewards = rewards + beta * r_int.cpu().numpy()
```

**Critical implementation notes:**
- Add `target_kl=0.05` before Phase 2 — the S13 run showed approx_kl=4×10⁹ without intrinsic reward. Changing reward scale mid-training amplifies this risk.
- VAE input is 192-dimensional (hidden_size=128 + encoder_dim=64). With n_envs=16 and n_steps=512, the effective dataset per update is 8192 transitions — many of them near-identical corridor steps. Use a **separate replay buffer for VAE training** to ensure balanced representation of hint-room observations.
- β should start small (0.001) and be verified to not change rollout/ep_rew_mean before increasing.

**Success metric:** Reward unchanged from Phase 1 + VAE trains (reconstruction loss decreasing, KL > 0).

---

### Phase 3a: Remove mem_start, keep Gaussian VIME

**Goal:** Verify intrinsic reward drives exploration to hint room without the spawn wrapper.

**Key risk:** Corridor tiles are visually near-identical. VAE novelty on MiniGrid state representations will saturate quickly — most tiles will have near-zero novelty after ~100k steps. The agent has no directional signal toward the hint room.

**Mitigation — episodic position novelty as fallback:**
```python
# Per-episode visited position hash
visited = set()
r_pos = 1.0 if (x, y) not in visited else 0.0
visited.add((x, y))
rewards += alpha * r_pos  # alpha annealed to 0 after convergence
```
This is explicitly spatial rather than representation-based — reliable for driving corridor traversal regardless of VAE quality. Run both signals and verify which one is actually driving exploration before removing either.

**Success metric:** Agent reaches hint room and makes correct junction choice without mem_start.

---

### Phase 3b: spCauchy VAE + LBS Intrinsic Reward

**Goal:** Replace Gaussian VAE with collapse-resistant hyperspherical geometry. Use Latent Bayesian Surprise as novelty signal.

**Architecture:**
```
s_t = cat([h_t, m_pooled]) → single spCauchy VAE → z_t ∈ S^{d-1}
r_int = LBS(z_t) = KL(spCauchy posterior || spCauchy prior)
```

**Why a single VAE, not split h/m streams:** h_t is an explicit deterministic function of m_t through the LMU equations (`y = W_query(h) @ m`, `h = tanh(W_x + W_h + W_m(y))`). The Wyner Common Information formulation requires `h ⊥ m | Z` — conditional independence that is architecturally impossible here. A single VAE on `cat([h, m_pooled])` captures the joint representation honestly. The split-stream framing is theoretically incorrect for this architecture.

**The closed-form collapse curvature result (4(d−1)²/d)** from HSWVIME is directly applicable here — monitor C_proj norm equivalent in the VAE encoder to detect collapse before it propagates.

**β schedule:** Single β annealed to 0 after convergence. Verify on S11/S13 before Craftax.

---

### Phase 4: Craftax

Three scaling changes required:

**theta:** MiniGrid episodes are 50-65 steps (theta=100-200). Craftax episodes span thousands of steps across multiple lives. Fixed-window LegT (current LMU) with any finite theta will lose early-episode information. Consider switching to **HiPPO-LegS** (scaled Legendre) which covers full history without a fixed window — eliminates theta as a hyperparameter entirely.

**memory_size (d):** d=64 was sufficient for S13 (theta=200). For Craftax memory horizons, d=128 minimum. Monitor reconstruction quality via the Voelker Eq. 3 MSE diagnostic.

**Encoder:** MinigridEncoder is hardcoded for MiniGrid obs space. Craftax has a different symbolic observation space — this is a clean swap, everything downstream is architecture-agnostic.

---

## Part 5: Open Questions

| Question | Priority | Notes |
|---|---|---|
| Does spCauchy VAE train stably on 192-dim s_t? | High | Test in Phase 2 before committing |
| Is VAE novelty directional enough to drive hint-room exploration? | High | Episodic position novelty as fallback |
| Does LBS scale to Craftax state space diversity? | Medium | Much larger state space than MiniGrid |
| HiPPO-LegS vs LegT for Craftax? | Medium | Non-trivial architecture change, do in Phase 4 |
| target_kl threshold for combined reward? | Low | Start at 0.05, tune based on Phase 2 KL behavior |
