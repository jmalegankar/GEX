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
