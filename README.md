# GEX
# GEX: Geodesic Exploration on the Hypersphere
## Pivot Plan

---

## What to KEEP

- `models/spherical_cauchy.py` — core distribution, sampling, KL
- `models/sc_unet_vae.py` — U-Net encoder/decoder
- `models/transition_sc_vae.py` — siamese transition encoder
- `models/memory.py` — SimHash (refactor to expose angular properties)
- `sparse_env.py` — DoorButton env (keep as unit test env, not main benchmark)
- `dataset.py` — transition dataset utilities
- `offline_rl/extractor.py` — SB3 feature extractor (modify input dim)
- SB3 PPO integration pattern (wrappers, callbacks, VecEnv)

## What to DROP

- `models/wyner.py` — entire GRU adaptive prior / Gaussian z module
- `online_rl/online_module.py` — online joint training (replace with simpler version)
- `online_rl/online_callback.py` — VAE+Wyner training callback
- The whole "Phase A / Phase B" training loop
- Teacher-Student mechanism (never implemented, remove from README)
- Variant B (use_skips) — dead code path
- KL(posterior || adaptive_prior) as surprise signal

## What to BUILD

### 1. Geodesic Bonus Module (replaces Wyner)

Core intrinsic reward computation using sphere geometry.
No GRU, no learned prior, no latent z — just distances on the sphere.

### 2. Spherical Episodic Memory (replaces SimHash HashMap)

k-NN on S^{d-1} with efficient retrieval.
Stores μ vectors directly, computes geodesic distances.

### 3. Angular Pseudo-Count Module (refactored SimHash)

Formalize SimHash buckets as spherical caps.
Track per-bucket counts for lifetime novelty.

### 4. Uniformity Regularizer (new training objective)

Add uniformity loss to SC-VAE training to encourage coverage.

### 5. Pixel Encoder Frontend (for Atari/Crafter)

CNN/ResNet → SC-VAE bottleneck for high-dim observations.

---

## New Architecture

```
                    ┌─────────────────────────┐
                    │    Observation Encoder   │
                    │  (CNN for pixels, or     │
                    │   CategoricalEmbed for   │
                    │   gridworlds)            │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Spherical Cauchy VAE    │
                    │  Transition Encoder      │
                    │  (s_t, a_t, s_{t+1})     │
                    │       → μ ∈ S^{d-1}      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Geodesic Bonus Head    │
                    │                          │
                    │  ┌─────────┐ ┌────────┐ │
                    │  │Episodic │ │Lifetime│ │
                    │  │ k-NN on │ │Angular │ │
                    │  │ S^{d-1} │ │PseudoCt│ │
                    │  └────┬────┘ └───┬────┘ │
                    │       │          │       │
                    │    r_epi    × 1/√(N+1)  │
                    │       └─────┬────┘       │
                    │          r_int           │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   PPO Policy            │
                    │   obs = [μ]  (or [μ,    │
                    │   obs_features] )       │
                    │   reward = r_ext +      │
                    │           η * r_int     │
                    └─────────────────────────┘
```

---

## Intrinsic Reward: Mathematical Definition

### Episodic Bonus (within-episode novelty)

Given episodic memory M_ep = {μ_1, ..., μ_t} on S^{d-1}:

```
d_geo(μ_a, μ_b) = arccos(clamp(μ_a · μ_b, -1, 1))

r_episodic(μ_t) = mean( d_geo(μ_t, μ_j) for j in kNN(μ_t, M_ep, k) )
```

Key property: r_episodic ∈ [0, π] — **naturally bounded, no normalization needed**.

When memory is empty (start of episode), r_episodic = π (maximum novelty).

### Lifetime Bonus (cross-episode novelty via angular pseudo-counts)

SimHash projects μ onto random hyperplanes → binary code → bucket.
Each bucket is a spherical cap on S^{d-1}.

```
bucket = SimHash(μ_t)
N(bucket) = lifetime count of transitions mapped to this bucket
r_lifetime(μ_t) = 1 / sqrt(N(bucket) + 1)
```

### Combined Intrinsic Reward

```
r_int = r_episodic × r_lifetime
```

No Welford normalization needed — both components are naturally bounded.
r_episodic ∈ [0, π], r_lifetime ∈ (0, 1].
Product r_int ∈ [0, π].

η scales the contribution: reward = r_ext + η * r_int

---

## SC-VAE Training: Add Uniformity Regularizer

Standard SC-VAE loss:
  L_vae = L_recon + β * KL(spCauchy || Uniform)

Add uniformity term (Wang & Isola 2020):
  L_uniform = log( E[exp(-t * d_geo(μ_i, μ_j)^2)] )  for i≠j in batch

Total:
  L = L_recon + β * L_kl + α * L_uniform

This explicitly encourages the encoder to spread transition
representations across S^{d-1}, maximizing exploration coverage.

The KL(spCauchy || Uniform) already pushes toward uniformity,
but L_uniform provides a direct, batch-level signal.

α should be small (0.01-0.1) — we want uniform spread but not
at the cost of destroying transition structure.

---

## Ablation Plan

| Variant | Episodic | Lifetime | Uniformity | Tests |
|---------|----------|----------|------------|-------|
| Full GEX | geodesic kNN | angular pseudo-ct | yes | Main result |
| No episodic | — | angular pseudo-ct | yes | Lifetime alone |
| No lifetime | geodesic kNN | — | yes | Episodic alone |
| No uniformity | geodesic kNN | angular pseudo-ct | no | Uniformity contribution |
| Euclidean baseline | L2 kNN | hash pseudo-ct | no | Sphere vs flat |
| RND-equivalent | — | — | no | Prediction error only |
| η sensitivity | geodesic kNN | angular pseudo-ct | yes | η ∈ {0.001..1.0} |
| Dimension sweep | geodesic kNN | angular pseudo-ct | yes | d ∈ {16,32,64,128} |

The **critical ablation** is Full GEX vs Euclidean baseline.
Same architecture, same counts, but μ ∈ R^d (Gaussian VAE) vs μ ∈ S^{d-1}.
This isolates the contribution of hyperspherical geometry.

---

## Benchmark Plan

### Tier 1: Fast iteration (days)
- DoorButton gridworld (10×10, 15×15, 20×20) — sanity check
- MiniGrid-KeyCorridor, MultiRoom — procedural gridworlds
- Craftax (JAX, 250× faster than Crafter) — primary benchmark

### Tier 2: Main results (weeks)
- Crafter (full, 1M steps) — standard benchmark
- MiniHack (navigation tasks) — procedural, partial observability
- Atari: Montezuma's Revenge (200M frames) — hard exploration gold standard

### Tier 3: Stress tests
- Atari: Pitfall, Private Eye, Gravitar — additional hard exploration
- Noisy TV variants of above — robustness to stochastic distractors

### Baselines (minimum)
- PPO (no intrinsic)
- RND (Burda et al. 2019)
- ICM (Pathak et al. 2017)
- NovelD (Zhang et al. 2021)
- E3B (Henaff et al. 2023)
- DRND (Yang et al. ICML 2024) — if feasible

---

## Implementation Priority Order

### Phase 1: Core GEX module (1-2 weeks)
1. `models/geodesic_bonus.py` — episodic kNN + pseudo-counts
2. `models/spherical_memory.py` — efficient kNN on S^{d-1}
3. `models/angular_counts.py` — formalized SimHash counting
4. Refactor `offline_rl/` to use GEX instead of Wyner
5. Validate on DoorButton (should match or beat current 96%)

### Phase 2: Uniformity + theory (1 week)
6. Add uniformity regularizer to SC-VAE training
7. Implement coverage metrics (uniformity on S^{d-1})
8. Run sphere vs Euclidean ablation on DoorButton
9. Visualize latent space coverage differences

### Phase 3: Scale to Craftax (2 weeks)
10. Build CNN frontend for pixel observations
11. Adapt SC-VAE for (84,84,3) or Craftax obs format
12. Implement Craftax integration
13. Run full benchmark suite
14. Tune η, d, k, hash_dim

### Phase 4: Atari + paper (3-4 weeks)
15. Atari preprocessing + integration
16. Montezuma's Revenge experiments (200M frames, 5 seeds)
17. Full ablation suite on all benchmarks
18. Write paper

---

## Paper Narrative (Draft)

**Title options:**
- "Geodesic Exploration: Intrinsic Motivation on the Hypersphere"
- "Exploring with Bounded Curiosity: Transition Encoding on S^{d-1}"
- "GEX: Geodesic Exploration Bonuses via Spherical Cauchy Transition Encoding"

**Story arc:**
1. Existing intrinsic motivation methods suffer from unbounded rewards,
   reward normalization sensitivity, and posterior collapse in VAE-based
   approaches.
2. We observe that encoding transitions on the hypersphere S^{d-1}
   provides three natural advantages: bounded geodesic distances
   (solving reward scaling), resistance to posterior collapse
   (maintaining informative representations), and a formal connection
   between uniformity on the sphere and exploration coverage.
3. We introduce GEX: a Spherical Cauchy VAE encodes transitions onto
   S^{d-1}, and exploration bonuses are computed via geodesic k-NN
   distance (episodic) and angular pseudo-counts (lifetime).
4. We prove that SimHash on the sphere corresponds to counting within
   spherical caps, connecting our method to the pseudo-count framework
   with explicit geometric characterization.
5. Experiments on Craftax, Crafter, MiniHack, and Montezuma's Revenge
   show GEX matches or exceeds state-of-the-art exploration methods
   while eliminating reward normalization hyperparameters.

**Key claims (must be validated empirically):**
- C1: Geodesic bonuses eliminate the need for reward normalization
- C2: Hyperspherical geometry outperforms Euclidean geometry for
      exploration (same architecture, same counting, sphere vs flat)
- C3: Uniformity regularization improves exploration coverage
- C4: GEX is competitive with RND/E3B on hard benchmarks




The one risk and how to mitigate it
The risk with GEX is that the episodic k-NN query gets expensive as episodes get long. In your 400-step gridworld episodes this is fine. In Montezuma's Revenge with episodes running to 100k+ steps, brute-force k-NN over all stored μ vectors becomes a bottleneck.
Two mitigations:

Cap the memory at the most recent N transitions (e.g., 1000). Oldest transitions fall off. This is fine since recent context matters more for episodic novelty.
Use approximate k-NN. Since μ is on the sphere, you can partition into angular cells (Voronoi on S^{d-1}) or use a simple grid of spherical caps. For d=32 or 64, even brute-force on 1000 vectors is fast (~0.1ms on CPU).


Why this pivot is stronger than what you had
Let me be direct about the tradeoffs:
What you gain:

A clean, testable novelty claim. "Geodesic exploration bonuses on S^{d-1}" — nobody has done this. It's one sentence, easy for reviewers to grasp, easy to verify empirically via the sphere-vs-Euclidean ablation.
You kill the normalization problem. Every existing intrinsic motivation method (RND, ICM, VIME, E3B) requires careful reward normalization — Welford running stats, clipping, etc. It's a tuning nightmare and a known failure mode. Your geodesic bonus is bounded in [0, π] by construction. That's a practical contribution reviewers will care about.
You remove ~60% of your codebase complexity. No Wyner module (GRU, Gaussian posterior, decoder, KL annealer, process_episode). No Phase A/Phase B training. No mu drift detection. No Welford normalization. The GEXModule is about 100 lines vs your OnlineHSWVIMEModule at 400+.
The uniformity-coverage connection is a real theoretical contribution. You're the first to connect Wang & Isola's uniformity metric to exploration coverage. This gives you a Proposition in the paper: "A transition encoder achieving uniformity on S^{d-1} maximizes the discriminability of visited transitions, providing an upper bound on exploration coverage." That's the kind of thing reviewers at ICLR love.
# GEX: Geometric Episodic eXploration

## One-Sentence Pitch

GEX encodes agent transitions on the hypersphere via a Spherical Cauchy VAE, then measures novelty through the *geometry itself* — episodic k-NN distance on the sphere as surprise, gated by SimHash lifetime novelty — eliminating the need for learned belief models (GRU/Wyner) that ablations proved unnecessary.

---

## 1. Why GEX Exists

### The Problem
Sparse-reward RL environments require intrinsic motivation for exploration. Standard methods (ICM, RND) suffer from the Noisy TV problem (rewarding unpredictable noise) and cyclic exploitation (revisiting the same states for repeated reward).

### What H-SW-VIME Proved
The predecessor system (H-SW-VIME) achieved 96%+ success rates with:
- Spherical Cauchy VAE encoding transitions → μ ∈ S^{d-1}
- GRU accumulating episode state W → adaptive prior p(z|W)
- Gaussian posterior q(z|W,μ) → KL surprise
- SimHash novelty gate

### What the Ablation Killed
Replacing the adaptive prior (GRU + Wyner) with a fixed N(0,I) prior produced **nearly identical results** (98.5% vs 99.5% train SR, same eval SR). The GRU's learned belief added nothing. What actually mattered:

1. **Transition encoding on the sphere** (SC-VAE) — validated, essential
2. **SimHash novelty gate** — doing most of the anti-cycling work
3. **Some surprise signal** — any form, normalized by Welford RMS

### The GEX Insight
If your representations already live on a well-structured sphere, you don't need a learned bottleneck to measure surprise. **The geometry provides the surprise signal directly** via k-NN distance in the episodic memory of stored μ vectors.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GEX Pipeline                             │
│                                                                 │
│  (s_t, a_t, s_{t+1})                                          │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────┐                                              │
│  │  SC-VAE      │  Frozen after Phase A training                │
│  │  encode()    │  Shared siamese encoder + transition bottleneck│
│  └──────┬───────┘                                              │
│         │ μ_{t+1} ∈ S^{d-1}                                   │
│         │                                                       │
│    ┌────┴────────────────┐                                     │
│    │                     │                                      │
│    ▼                     ▼                                      │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │  Episodic    │  │  Lifetime    │                            │
│  │  Memory      │  │  Novelty     │                            │
│  │  (k-NN on    │  │  (SimHash +  │                            │
│  │   sphere)    │  │   HashMap)   │                            │
│  └──────┬───────┘  └──────┬───────┘                            │
│         │ surprise         │ is_novel ∈ {0, 1}                  │
│         │                  │                                    │
│         └───────┬──────────┘                                   │
│                 │                                               │
│                 ▼                                               │
│     r_int = is_novel(μ) × surprise(μ)                          │
│                 │                                               │
│                 ▼                                               │
│     r_total = r_ext + η × normalize(r_int)                     │
│                 │                                               │
│                 ▼                                               │
│            PPO update                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Component 1: Transition SC-VAE (already built)

Compresses (s_t, a_t, s_{t+1}) → μ ∈ S^{d-1}.

- Siamese conv encoder (shared weights for s_t and s_{t+1})
- Action embedding fused at bottleneck
- Spherical Cauchy latent (Möbius reparameterization)
- Three-headed decoder (s_t, action, s_{t+1} — independent heads)
- KL(spCauchy ∥ Uniform) via Theorem 1 series (arXiv:2506.21278)
- Trained offline on collected rollouts, then **frozen** for RL

### 2.2 Component 2: Episodic Memory + Geometric Surprise

**This replaces the entire GRU/Wyner/adaptive-prior stack.**

Per episode:
1. Initialize empty memory M = []
2. At each step, compute μ_{t+1} = SC-VAE.encode(s_t, a_t, s_{t+1})
3. Query k nearest neighbors of μ_{t+1} in M (cosine distance on sphere)
4. Surprise = f(distances to k neighbors)
5. Append μ_{t+1} to M

When M is empty or has fewer than k entries, surprise is maximal (everything is novel at episode start).

**Surprise function options** (to ablate):

```
Option A (inverse mean distance):
  surprise = 1 / (mean_cosine_sim(μ, kNN(μ)) + ε)

Option B (kernel density, like NGU/RIDE):
  surprise = 1 / sqrt(mean(max(cosine_sim(μ, kNN(μ)) - threshold, 0)²) + ε)

Option C (count-based on sphere):
  surprise = 1 / sqrt(count_within_radius(μ, M, r) + 1)
```

Option B (kernel-based) is closest to what Never-Give-Up (NGU) and RIDE use, adapted for spherical geometry. This is likely the strongest default because it's been validated in prior work, just never with hyperspherical representations.

**Why this works:** μ vectors on the sphere have a natural distance metric (angular/cosine distance). Similar transitions cluster together. A transition the agent has seen many variants of will have many close neighbors → low surprise. A genuinely novel transition (button press, new room) will be far from all stored μ's → high surprise. The sphere's geometry does the work that the GRU was supposed to do but didn't.

### 2.3 Component 3: Lifetime Novelty Gate (SimHash + HashMap)

Prevents cyclic exploitation across episodes.

- SimHash: μ → binary code via random hyperplane projections (preserves angular distance)
- HashMap: set membership ("have I ever seen this hash?")
- Binary gate: r_int = 0 if hash seen before, else r_int = surprise

This is episodic novelty (Component 2) × lifetime novelty (Component 3). Two complementary mechanisms:
- Episodic: "Is this new *within this episode*?" (resets per episode)
- Lifetime: "Is this new *ever*?" (persists across episodes, never resets — or uses sliding window)

### 2.4 Component 4: Reward Normalization

Welford running RMS normalization on r_int before scaling by η. This is what made the H-SW-VIME ablation work — it absorbed the ~100× scale difference between adaptive-prior KL (~0.2) and fixed-prior KL (~20). Essential for stability.

### 2.5 Component 5: PPO Integration

Standard PPO (via SB3 custom wrapper or from-scratch). Policy receives augmented observation:

```
Variant A (minimal): obs = [emb(s_t).flat]
  → policy sees only current observation
  → simplest, cleanest, most comparable to baselines

Variant B (informed): obs = [emb(s_t).flat, μ_t]
  → policy gets transition encoding as extra context
  → μ encodes what just happened, might help credit assignment

Variant C (full context): obs = [emb(s_t).flat, μ_t, surprise_t]
  → policy also sees its own surprise signal
  → most information, but risk of overfitting to intrinsic signal
```

Start with Variant A. Add B if needed. Probably never need C.

---

## 3. What's Novel (Paper Story)

The contribution is NOT the individual components. k-NN episodic novelty exists (NGU, RIDE). SimHash exists. SC-VAE exists. The contribution is the specific combination and the theoretical argument for why it works:

1. **Transition encoding, not state encoding.** Curiosity about *what changed*, not *what I see*. The button press is a transition event invisible to state-only methods.

2. **Hyperspherical geometry as a natural metric space for novelty.** On the sphere, cosine distance is the canonical metric. k-NN on S^{d-1} directly measures angular novelty of transitions. No learned distance function needed.

3. **The ablation argument.** We empirically show that learned episodic belief models (GRU + adaptive prior) add no value over geometric episodic memory when the representations are well-structured. This simplifies the architecture dramatically while maintaining performance.

4. **Spherical Cauchy prevents the failure mode that makes this possible.** Gaussian VAEs suffer posterior collapse → all μ's cluster at origin → k-NN is meaningless. SC-VAE maintains well-spread, informative μ's on the sphere, which is what makes geometric surprise work.

### Comparison Table (for paper)

| Property | ICM | RND | NGU | RIDE | **GEX** |
|---|---|---|---|---|---|
| Encodes | states | states | states | states | **transitions** |
| Geometry | Euclidean | Euclidean | Euclidean | Euclidean | **Hyperspherical** |
| Surprise | pred. error | pred. error | k-NN embed. | k-NN embed. | **k-NN on S^{d-1}** |
| Episodic memory | ✗ | ✗ | ✓ (controllable) | ✓ | ✓ **(geometric)** |
| Lifetime novelty | ✗ | implicit | ✓ (RND arm) | ✗ | ✓ **(SimHash)** |
| Noisy TV robust | ✗ | partial | partial | partial | **✓ (transition enc.)** |
| Posterior collapse | N/A | N/A | possible | possible | **prevented (SC)** |
| Learned dynamics | ✓ (fwd model) | ✓ (predictor) | ✓ (embedding) | ✓ (fwd+inv) | **✗ (geometry only)** |

---

## 4. File Structure

```
gex/
├── plan.md                          ← this file
│
├── models/
│   ├── __init__.py
│   ├── config.py                    ← SCVAEConfig, MemoryConfig, RLConfig, Config
│   ├── spherical_cauchy.py          ← SC math: sample, kl_to_uniform, mobius_add
│   ├── sc_vae.py                    ← TransitionSCVAE (encoder/decoder/loss)
│   ├── episodic_memory.py           ← EpisodicMemory: k-NN on sphere + surprise
│   └── novelty_gate.py              ← SimHash + HashMap (lifetime novelty)
│
├── rl/
│   ├── __init__.py
│   ├── gex_module.py                ← GEXModule: wires SC-VAE + memory + gate
│   ├── vec_wrapper.py               ← SB3 VecEnv wrapper (reward augmentation)
│   ├── reward_normalizer.py         ← Welford running RMS
│   └── callbacks.py                 ← Logging, diagnostics, eval
│
├── training/
│   ├── __init__.py
│   ├── collect_rollouts.py          ← Keyboard / random / RND policy → .npz
│   ├── train_sc_vae.py              ← Phase A: train SC-VAE on rollout buffer
│   └── train_rl.py                  ← Phase B: PPO + GEX intrinsic reward
│
├── envs/
│   ├── __init__.py
│   └── door_button.py               ← SingleAgentDoorButton (sparse reward)
│
├── tests/
│   ├── test_sc_vae.py               ← Gradient sanity, reconstruction, unit norm
│   ├── test_episodic_memory.py      ← k-NN correctness, surprise decay, empty memory
│   ├── test_novelty_gate.py         ← Hash consistency, collision rate, lifecycle
│   ├── test_gex_module.py           ← End-to-end: transitions → intrinsic reward
│   └── test_reward_normalizer.py    ← Welford correctness, scale absorption
│
└── analysis/
    ├── compare_baselines.py          ← GEX vs RND vs ICM vs PPO
    ├── ablation.py                   ← Remove each component, measure impact
    └── visualize.py                  ← Latent space, exploration heatmaps, surprise
```

---

## 5. Module Sketches (No Implementation)

### 5.1 models/config.py

```python
@dataclass
class SCVAEConfig:
    view_size: int = 5
    n_object_types: int = 11
    n_colors: int = 6
    n_states: int = 4
    embed_per_ch: int = 4           # → embed_channels = 12
    conv_channels: tuple = (32, 64)
    hidden_dim: int = 256
    latent_dim: int = 32
    n_actions: int = 7
    action_embed_dim: int = 4
    rho_min: float = 0.001
    rho_max: float = 0.999
    kl_max_terms: int = 256         # series truncation for KL
    kl_use_asymptotic: bool = True  # switch to Prop 2 when rho > 0.9

@dataclass
class MemoryConfig:
    k_neighbors: int = 5            # k for k-NN surprise
    kernel_epsilon: float = 0.001   # denominator floor
    kernel_threshold: float = 0.008 # similarity clip for kernel-based surprise
    hash_dim: int = 128             # SimHash projection bits
    hash_input_dim: int = 32        # = latent_dim

@dataclass
class RLConfig:
    total_timesteps: int = 1_000_000
    n_envs: int = 1                 # start with 1 for debugging
    lr: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    eta_start: float = 0.01         # intrinsic reward weight
    eta_end: float = 0.001          # decay target
    eta_decay_steps: int = 500_000

@dataclass
class Config:
    sc_vae: SCVAEConfig = field(default_factory=SCVAEConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rl: RLConfig = field(default_factory=RLConfig)
```

### 5.2 models/spherical_cauchy.py

```python
"""Pure math — no neural networks. Already verified against arXiv:2506.21278."""

def mobius_add(a, x): ...
def sc_sample(mu, rho): ...
def sc_kl_uniform(rho, dim, *, max_terms=256, tol=1e-10): ...
def sc_kl_uniform_asymptotic(rho, dim): ...  # Proposition 2, for rho > 0.9
```

### 5.3 models/sc_vae.py

```python
"""Transition SC-VAE. Already built and verified."""

class SCVAEForwardOutput(NamedTuple): ...
class CategoricalEmbedding(nn.Module): ...
class ConvBlock(nn.Module): ...
class StateEncoder(nn.Module): ...
class TransitionBottleneck(nn.Module): ...
class FlatDecoder(nn.Module): ...

class TransitionSCVAE(nn.Module):
    def encode(self, s_t, a_t, s_next) -> mu: ...
    def forward(self, s_t, a_t, s_next) -> SCVAEForwardOutput: ...
    def loss(self, out) -> (l_recon, l_kl): ...
    def transition_target(self, s_t, a_t, s_next) -> target: ...
```

### 5.4 models/episodic_memory.py

```python
"""
Episodic memory with geometric surprise on S^{d-1}.

Replaces the GRU + Wyner + adaptive prior stack.
Resets at episode boundaries.
"""

class EpisodicMemory:
    """
    Stores μ vectors from the current episode.
    Computes surprise via k-NN cosine distance.
    """
    def __init__(self, k: int, kernel_epsilon: float, kernel_threshold: float): ...

    def reset(self) -> None:
        """Clear memory at episode start."""
        ...

    def query_and_add(self, mu: Tensor) -> float:
        """
        1. Compute cosine distances from mu to all stored vectors
        2. Find k nearest neighbors
        3. Compute surprise (kernel-based)
        4. Add mu to memory
        5. Return surprise scalar

        If memory has fewer than k entries, return max surprise.
        """
        ...

    def __len__(self) -> int: ...
```

### 5.5 models/novelty_gate.py

```python
"""
SimHash + HashMap for lifetime novelty detection.
Persists across episodes. Optional sliding window for very long training.
"""

class SimHash:
    """
    Random hyperplane projection: μ ∈ S^{d-1} → binary code.
    Preserves angular distance (LSH property).
    Fixed (non-trainable) projection matrix.
    """
    def __init__(self, input_dim: int, hash_dim: int, seed: int = 0): ...
    def __call__(self, mu: Tensor) -> int: ...

class NoveltyGate:
    """
    Wraps SimHash + set for binary novelty detection.
    """
    def __init__(self, input_dim: int, hash_dim: int): ...

    def is_novel(self, mu: Tensor) -> bool:
        """Check if hash(mu) has been seen before."""
        ...

    def add(self, mu: Tensor) -> None:
        """Mark hash(mu) as seen."""
        ...

    def reset(self) -> None:
        """Clear all entries (use sparingly or never)."""
        ...

    def count(self) -> int: ...
```

### 5.6 rl/gex_module.py

```python
"""
Wires together SC-VAE + EpisodicMemory + NoveltyGate.
Called by the VecEnv wrapper at each step.
"""

class GEXModule:
    """
    Stateful module that tracks episode context and computes intrinsic reward.

    Lifecycle:
      module = GEXModule(sc_vae, memory_cfg)
      aug_obs = module.reset(first_obs)       # episode start
      for each step:
          r_int, aug_obs, diag = module.step(prev_obs, action, next_obs)
    """
    def __init__(self, sc_vae: TransitionSCVAE, cfg: MemoryConfig): ...

    def reset(self, obs: ndarray) -> ndarray:
        """
        Episode boundary. Reset episodic memory.
        Encode initial dummy transition (s_0, no_op, s_0).
        Return augmented observation.
        """
        ...

    def step(self, prev_obs: ndarray, action: int, next_obs: ndarray) -> tuple:
        """
        1. Encode transition: mu = sc_vae.encode(prev_obs, action, next_obs)
        2. Episodic surprise: surp = episodic_memory.query_and_add(mu)
        3. Lifetime novelty: novel = novelty_gate.is_novel(mu)
        4. If novel: novelty_gate.add(mu)
        5. r_int = novel * surp
        6. Return (r_int, augmented_obs, diagnostics_dict)
        """
        ...
```

### 5.7 rl/vec_wrapper.py

```python
"""SB3-compatible VecEnv wrapper that injects GEX intrinsic rewards."""

class VecGEXWrapper(VecEnvWrapper):
    """
    Wraps a VecEnv to add GEX intrinsic rewards to extrinsic rewards.

    Handles:
    - Episode boundaries (terminal_observation from SB3 auto-reset)
    - Reward normalization (Welford RMS)
    - η annealing (intrinsic weight decay)
    - Diagnostic logging (surprise, novelty rate, etc.)
    """
    def __init__(self, venv, gex_module, eta_start, eta_end, eta_decay_steps): ...
    def reset(self, **kwargs) -> obs: ...
    def step_wait(self) -> (obs, rewards, dones, infos): ...
```

### 5.8 rl/reward_normalizer.py

```python
"""Welford online running RMS for reward normalization."""

class WelfordRMS:
    def __init__(self): ...
    def update(self, x: float) -> None: ...
    def normalize(self, x: float) -> float: ...

    @property
    def mean(self) -> float: ...
    @property
    def std(self) -> float: ...
```

---

## 6. Training Phases

### Phase A: SC-VAE (offline, ~30 min)

```
1. Collect rollouts via RND policy (best coverage) → .npz
2. Build transition buffer: (s_t, a_t, s_{t+1}) tuples
3. Train TransitionSCVAE:
   - Loss = MSE(recon, target) + β·KL(spCauchy ∥ Uniform)
   - β annealed from 0 → 0.005 over first 5k steps
   - Adam, lr=1e-3, batch=64, ~50-100 epochs
4. Validate:
   - μ unit norm ✓
   - Latent clustering by transition type ✓
   - Reconstruction quality ✓
   - No posterior collapse (KL > 0) ✓
5. Freeze SC-VAE weights. Never touch again.
```

### Phase B: RL with GEX (online, ~2-4 hrs)

```
1. Load frozen SC-VAE
2. Create GEXModule (SC-VAE + EpisodicMemory + NoveltyGate)
3. Wrap env in VecGEXWrapper
4. Train PPO:
   - r_total = r_ext + η·normalize(r_int)
   - η decays from 0.01 → 0.001 over 500k steps
   - Standard PPO hyperparams (lr=3e-4, n_steps=2048, etc.)
5. Log:
   - Success rate, steps to button, steps to goal
   - Mean surprise, novelty rate, η
   - Episode length, extrinsic vs intrinsic reward
```

---

## 7. Ablation Plan

Each ablation removes one component to prove its necessity:

| Ablation | What's removed | Expected failure mode |
|---|---|---|
| No intrinsic reward (η=0) | Everything | SR drops to ~40%, eval collapses |
| No episodic memory | k-NN surprise → fixed constant | No within-episode exploration guidance |
| No novelty gate | SimHash → always novel | Cyclic exploitation, reward farming |
| No normalization | Welford RMS → raw reward | Unstable training, scale sensitivity |
| Gaussian VAE | SC-VAE → Gaussian VAE | Posterior collapse, meaningless k-NN |
| State encoding | (s,a,s') → s only | Button press invisible, causal chain broken |

### Bonus comparisons:
| Comparison | Purpose |
|---|---|
| GEX vs RND | Does geometric surprise beat prediction error? |
| GEX vs ICM | Does transition encoding beat forward model? |
| GEX vs H-SW-VIME | Does removing GRU/Wyner actually hurt? (It shouldn't.) |
| GEX vs NGU-style | Ours with spherical geometry vs theirs with Euclidean |

---

## 8. Scalability Path

### Current: DoorButton gridworld (5×5 view, d=32)
- k-NN over ~400 vectors per episode: <0.1ms, trivially fast

### Next: Craftax / MiniHack
- Larger observation space, longer episodes
- SC-VAE needs larger conv encoder (view_size > 5)
- k-NN over ~1000-5000 vectors: still fast at d=32

### Future: Atari / Montezuma's Revenge
- 100k+ step episodes → must cap episodic memory
- Options:
  a. Rolling window: keep most recent N=1000 μ's
  b. Reservoir sampling: uniform subsample of episode
  c. Approximate k-NN: angular LSH or spherical Voronoi
- SC-VAE encoder: replace CategoricalEmbedding with standard CNN for pixel input

---

## 9. Key Design Decisions & Rationale

**Q: Why not just use RND?**
A: RND encodes states, not transitions. It can't distinguish "standing next to button" from "just pressed button." Also no episodic memory — same state is equally surprising every episode.

**Q: Why not NGU/RIDE with your SC-VAE?**
A: That's approximately what GEX is! The key differences: (1) transition encoding not state encoding, (2) hyperspherical geometry gives a principled distance metric, (3) SC prevents the posterior collapse that would make k-NN meaningless.

**Q: Why freeze the SC-VAE during RL?**
A: Core research principle. If SC-VAE trains online, the μ distribution drifts, which invalidates both the episodic memory's stored μ's and the SimHash projections. Freezing isolates the exploration mechanism from representation instability.

**Q: Why SimHash + HashMap instead of Bloom filter?**
A: HashMap is simpler, exact, and sufficient for gridworld episodes. Bloom filter was designed for multi-million-entry scenarios. If we scale to Atari, switch to sliding-window Bloom filter then. Don't over-engineer now.

**Q: Why not learn the surprise function?**
A: The entire point is that we DON'T need to learn it. The sphere's geometry IS the surprise function. Learned surprise (GRU/Wyner) was the thing the ablation killed.

---

## 10. Success Criteria

### Must-hit:
- [ ] GEX ≥ RND success rate on DoorButton
- [ ] Each ablation shows measurable degradation
- [ ] Intrinsic reward decreases over training (no farming)
- [ ] Stable training (no NaN, no collapse)

### Stretch:
- [ ] GEX > RND by ≥15% success rate
- [ ] GEX finds button faster (fewer steps)
- [ ] Clean latent space visualization (transition types cluster)
- [ ] Generalizes to at least one non-gridworld env