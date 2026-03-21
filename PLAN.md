# Claude Execution Plan — spCauchy NeurIPS 2026
**Created: 2026-03-20 · Deadline: May 11, 2026**

---

## Current State

**Done:**
- TransitionSCVAE with forward prediction head, fc_rho bias init, GL cache fix
- TransitionGaussianVAE with free-bits KL
- HSWVimePPO with dual optimizers, GRU memory, intrinsic reward normalization (Welford RunningMeanStd)
- CategoricalGridWithDirEmbedding, PixelCNNEmbedding
- 37/38 math verification tests passing
- train.py CLI with MiniGrid suite, wandb support

**Not Done:**
- Smoke test (end-to-end training run)
- ENV_SUITE missing key environments from WORKSTREAMS.md (MultiRoom-N7-S4, N12-S10, KeyCorridorS4R3)
- 3x3 view size option (harder exploration — agent sees less, must explore more)
- CrafterCNNEmbedding + Crafter wrapper
- AtariCNNEmbedding + Atari wrappers
- RND baseline (Skip for now, I have a implementation somehwre less)
- Experiment launch scripts (HPC batch jobs)
- All figures, analysis, writing

---

## Architecture Reminder

```
obs → Embedding(obs) → ConvEncoder → h ∈ ℝ^128
(h_t, a_t, h_{t+1}) → spCauchy encoder → (μ, ρ) on S^{d-1}
                     → forward predictor → ĥ_{t+1}
intrinsic_reward = ||ĥ_{t+1} - sg(h_{t+1})||²
policy_features = concat(sg(μ), h_mem)  →  PPO actor-critic
```

Swap Embedding per domain. Everything else stays identical.

---

## Phase 1: First Signal (March 20-24)

### 1a. Smoke test on DoorButton ✅
- [x] Run train.py on DoorButton for ~2k steps, both latent types — no crashes
- [x] Verified: losses decrease, intrinsic rewards non-zero and decreasing (148→1.5 over 2k steps)
- [x] Verified: all metrics logged (kl_loss, rho_mean, fwd_loss, recon_loss, intrinsic_mean, etc.)
- [x] MiniGrid DoorKey-8x8 with view_size=3 also works end-to-end

### 1b. Add missing MiniGrid environments to ENV_SUITE ✅
- [x] Added `multiroom_n7s4`, `multiroom_n12s10`, `keycorridor_s4r3` to ENV_SUITE
- [x] N7-S4 and N12-S10 needed manual gymnasium.register() (not pre-registered in minigrid)
- [x] All 3 verified working with agent_view_size=3
- [x] Smoke test on multiroom_n7s4 passed

### 1c. 3x3 view size support ✅
- [x] ConvEncoder handles (3, 3) spatial input — verified via unit test and smoke test
  - 3x3 → DownBlock1 → 2x2 → DownBlock2 → 1x1 → DownBlock3 → 1x1 → global avg pool → (128,)
  - Third DownBlock is a no-op spatially but functionally correct, negligible overhead

### 1d. KL dynamics comparison — THE priority figure
- [ ] Run spCauchy vs Gaussian-vanilla vs Gaussian-stabilized on MultiRoom-N7-S4, view_size=3
- [ ] 20M frames, 3 seeds each (9 runs, fast on HPC)
- [ ] Plot: KL divergence over training steps for all 3 methods
- [ ] **Expected**: Gaussian-vanilla KL → 0 (collapse), spCauchy KL stays healthy
- [ ] This single figure motivates the entire paper

### 1e. First exploration comparison
- [ ] Same runs as 1d: compare episode return curves
- [ ] Does spCauchy solve MultiRoom-N7-S4 faster than Gaussian?

**Exit criteria**: KL dynamics plot shows clear Gaussian collapse vs spCauchy stability

---

## Phase 2: Full MiniGrid + Crafter Infra (March 24-31)

### 2a. Full MiniGrid sweep
- [ ] 5 methods × 5 envs × 5 seeds = 125 runs at 20M frames
- [ ] Methods: spCauchy, Gaussian-vanilla, Gaussian-stabilized, RND, PPO-no-intrinsic
- [ ] HPC batch script: parametric sweep over (method, env, seed)
- [ ] view_size=3 for harder exploration signal

### 2b. RND baseline implementation
- [ ] Implement RND module: random target network + predictor network
- [ ] intrinsic_reward = ||predictor(obs) - sg(target(obs))||²
- [ ] Same normalization (RunningMeanStd) as forward prediction
- [ ] Add `--intrinsic_type {forward_pred, rnd, none}` flag to train.py
- [ ] RND uses the same embedding → ConvEncoder → h pipeline

### 2c. CrafterCNNEmbedding ✅
- [x] 3-layer CNN: Conv(3,32,8,4) → Conv(32,64,4,2) → Conv(64,64,3,1) → (B,64,4,4)
- [x] Handles SB3 VecTransposeImage (input is already (B, C, H, W) for pixel obs)
- [x] ConvEncoder uses channels=[128] for Crafter (one DownBlock: 4→2 → pool → 128)
- [x] Smoke tested: both spCauchy and Gaussian run on Crafter end-to-end

### 2d. Crafter Gymnasium wrapper
- [x] `pip install crafter` — installed v1.8.3
- [x] CrafterTrainingWrapper: obs (64,64,3) uint8, Discrete(17), tracks achievements in info
- [x] Achievement tracking: cumulative max per episode, exposed in info["achievements"]
- [x] Added `"crafter"` to ENV_SUITE with domain="crafter" routing
- [ ] Achievement score logging callback (geometric mean of 22 success rates) — needed for eval

### 2e. Reward normalization specification
- [ ] Document: using Welford's running mean/std (already implemented)
- [ ] Verify: normalization is per-step, not per-episode
- [ ] Clip normalized intrinsic reward to [-5, 5] for stability
- [ ] Consider: percentile-based normalization as alternative if RunningMeanStd is unstable

**Exit criteria**: MiniGrid sweep launched, Crafter runs end-to-end

---

## Phase 3: Atari + Crafter Results (March 31 - April 7)

### 3a. AtariCNNEmbedding
- [ ] Nature DQN architecture: Conv(1,32,8,4) → Conv(32,64,4,2) → Conv(64,64,3,1) → flatten → Linear(3136, 512)
- [ ] Input: (B, 84, 84, 1) uint8 grayscale (after standard preprocessing)
- [ ] Implements EmbeddingInterface

### 3b. Atari wrappers
- [ ] `pip install gymnasium[atari] ale-py`
- [ ] Standard wrappers: NoopReset, MaxAndSkip(4), EpisodicLife, FireReset, WarpFrame(84), ClipReward, FrameStack(4)
- [ ] Atari 100k protocol (100k interactions = 400k frames with action repeat 4)
- [ ] Full 200M protocol for asymptotic comparison

### 3c. Launch Atari runs
- [ ] spCauchy + Gaussian-vanilla + RND on Montezuma, Venture, Gravitar
- [ ] 3 seeds each = 27 runs, 2-3 days per run
- [ ] Pipeline on HPC, parallelize across GPUs

### 3d. Crafter results analysis
- [ ] Achievement score (geometric mean of 22 success rates)
- [ ] Learning curves for total reward
- [ ] 5 methods × 5 seeds = 25 runs
- [ ] Compare against published DreamerV3, Plan2Explore, IMPALA numbers

**Exit criteria**: Atari running, Crafter results analyzed

---

## Phase 4: Critical Checkpoint (April 7-10)

### Checkpoint questions:
1. MiniGrid: spCauchy beats Gaussian on ≥2 hard exploration envs? ✓/✗
2. Crafter: spCauchy achieves more achievements than Gaussian? ✓/✗
3. KL dynamics: visible collapse in Gaussian that spCauchy avoids? ✓/✗
4. View size 3x3 makes the exploration problem harder (as expected)? ✓/✗

**If YES to ≥1**: proceed with confidence
**If NO to all**: emergency pivot — diagnose why, adjust hyperparameters, consider different intrinsic reward formulation

---

## Phase 5: Ablations + Analysis (April 10-17)

### 5a. 2×2 disentanglement (Table 2)
- [ ] {spCauchy, Gaussian} × {fwd-pred, RND} on MultiRoom-N12-S10 + Crafter
- [ ] 4 cells × 2 envs × 5 seeds = 40 runs
- [ ] Separates: "is it the latent space or the prediction head?"

### 5b. Latent dim ablation (Table A1)
- [ ] d ∈ {8, 16, 32, 64} on MultiRoom-N7-S4 + Crafter
- [ ] spCauchy only, 3 seeds each = 24 runs

### 5c. Noise robustness (Fig 8)
- [ ] MultiRoom-N7-S4, view_size=3
- [ ] Observation noise σ ∈ {0, 0.1, 0.3, 0.5}
- [ ] Add Gaussian noise to encoded observations before VAE
- [ ] spCauchy vs Gaussian-vanilla, 3 seeds each = 24 runs

### 5d. Statistical testing
- [ ] Bootstrap 95% confidence intervals on all metrics (10k bootstrap samples)
- [ ] Report CIs in all tables and shaded regions in all learning curves
- [ ] Welch's t-test or Mann-Whitney U for pairwise comparisons
- [ ] Flag results where p > 0.05

### 5e. Quantitative latent space metrics (supplement Fig 7)
- [ ] Uniformity metric (already in utils.py) over training — plot as Fig A2
- [ ] k-NN probe: train k-NN on μ vectors to predict game state, report accuracy
  - Stronger than t-SNE for showing "latent space is well-organized"
- [ ] Report both alongside t-SNE in paper

**Exit criteria**: all ablation runs complete, statistical analysis done

---

## Phase 6: Figures + Writing (April 17 - May 1)

### Figures (in priority order):
1. Fig 3: KL dynamics (spCauchy vs Gaussian across domains) — THE figure
2. Fig 4: MiniGrid learning curves (5 methods × 5 envs)
3. Fig 5: Crafter achievement scores (bar chart)
4. Table 1: Main results across all domains
5. Fig 6: Atari learning curves
6. Fig 2: Collapse curvature theory plot (already in plot_spcauchy_math.py)
7. Table 2: 2×2 disentanglement
8. Fig 7: Latent visualization (t-SNE + k-NN probe accuracy)
9. Fig 8: Noise robustness
10. Fig 1: Architecture diagram
11. Table 3: Crafter per-achievement breakdown
12. Appendix figures: ρ histograms, uniformity, hyperparameter tables

### Writing timeline:
- April 17-20: Method (2p), Background (1p)
- April 20-24: Experiments skeleton with figures
- April 24-28: Intro, Related Work, Analysis, Conclusion
- April 28: Send draft to advisor
- May 1-8: Revise based on feedback
- May 10: Submit

---

## Open Questions / Risks

1. **ConvEncoder with 3x3 input**: need to verify/fix architecture for small spatial dims
2. **Crafter achievement tracking**: need to hook into Crafter's info dict
3. **Atari frame stacking**: 4-frame stack means embedding input is (B, 84, 84, 4) not (B, 84, 84, 1) — adjust AtariCNNEmbedding input channels
4. **RND + spCauchy interaction**: in 2×2 table, RND operates on raw observations, not on spCauchy latent. This is intentional (tests whether improvement comes from latent space vs prediction method), but worth discussing in paper.
5. **HPC job management**: need SLURM scripts for batch sweeps

---

## HPC Launch Commands (template)

```bash
# MiniGrid sweep
for method in spcauchy gaussian; do
  for env in multiroom_n7s4 multiroom_n12s10 keycorridor keycorridor_s4r3 doorkey; do
    for seed in 0 1 2 3 4; do
      sbatch run.sh --env $env --latent_type $method --seed $seed --view_size 3 \
        --total_timesteps 20000000 --normalize_intrinsic --wandb
    done
  done
done

# Add RND and no-intrinsic variants similarly
```
