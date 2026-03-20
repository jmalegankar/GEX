# Week 1 Workstreams — spCauchy Exploration in POMDPs

**Branch:** `pure`
**Goal:** Clean rebuild of infrastructure for NeurIPS 2026 submission (deadline May 11, 2026)

---

## Workstream 1: Strip Wyner, SimHash, QA loss from codebase

**Status:** In Progress
**Est:** ~4 hrs

Strip all old intrinsic reward machinery. After this, the codebase is a clean PPO + spCauchy-VAE
with no intrinsic reward source (intrinsic rewards are zeroed until Workstream 2 adds the forward
prediction head).

### Tasks
- [x] 1a. Delete `models/qa_module.py` and remove all QA references in PPO/buffer
- [x] 1b. Delete `models/episodic_memory.py` and remove all references
- [x] 1c. Delete `models/wyner.py`, strip Wyner classes from `models/torch_layers.py`
- [x] 1d. Simplify `HSWVIMEActorCriticPolicy` — remove Wyner feature extractor, multihead
  attention; policy features = `mu` from VAE directly
- [x] 1e. Simplify `TransitionRolloutBuffer` — remove questions/answers/memories/timesteps
- [x] 1f. Simplify `HSWVimePPO` — remove Wyner/QA/episodic memory from collect_rollouts and
  train; zero intrinsic rewards for now
- [x] 1g. Update `train.py` — remove Wyner/episodic memory args and construction
- [x] 1h. Update `models/__init__.py` — remove Wyner exports
- [x] 1i. Update `config.py` — remove WynerConfig

### Files deleted
- `models/qa_module.py`
- `models/episodic_memory.py`
- `models/wyner.py`

### Files modified
- `models/torch_layers.py` — keep only `sinusoidal_timestep_encoding`
- `hswvime_ppo/hswvime_ppo.py` — strip Wyner/QA/episodic memory from PPO
- `hswvime_ppo/buffer.py` — remove questions/answers/memories/timesteps
- `hswvime_ppo/policies.py` — simplify to use mu directly
- `train.py` — remove Wyner/episodic memory construction
- `models/__init__.py` — remove Wyner exports
- `config.py` — remove WynerConfig

---

## Workstream 2: Add forward prediction head to TransitionSCVAE

**Status:** Done

Add a `ForwardPredictor` module: `(z, h_t, a_emb) -> h_{t+1}_pred`. This is the new intrinsic
reward signal: `r_int = ||pred - sg(h_{t+1})||^2`.

### Tasks
- [x] 2a. Create `ForwardPredictor(nn.Module)` — 2-layer MLP (z_dim + h_dim + a_dim -> hidden -> hidden -> h_dim)
- [x] 2b. Integrate into `TransitionSCVAE.__init__` and `forward()`
- [x] 2c. Add `fwd_loss` to `TransitionSCVAE.loss()` (MSE between predicted and stop-grad actual h_{t+1})
- [x] 2d. Expose `intrinsic_reward(s_t, a_t, s_tp1) -> Tensor` method for rollout collection
- [x] 2e. Wire intrinsic reward into `HSWVimePPO.collect_rollouts()` and `train()`
- [x] 2f. Add `--vae_fwd_coef` arg to `train.py` and pass through to PPO

### Files modified
- `models/vae.py` — added `ForwardPredictor`, `intrinsic_reward()`, `fwd_pred`/`fwd_target` in VAEOutput, `fwd_loss` in VAELoss
- `hswvime_ppo/hswvime_ppo.py` — compute intrinsic rewards during rollout, include fwd_loss in training, log it
- `train.py` — added `--vae_fwd_coef` argument

---

## Workstream 3: Implement Gaussian-TransitionVAE baseline

**Status:** Done

Create `TransitionGaussianVAE` — identical architecture to `TransitionSCVAE` but with Gaussian
latent space: `fc_mu` (NO L2-norm), `fc_logvar`, standard reparameterization, KL to N(0,I).

### Tasks
- [x] 3a. Create `TransitionGaussianVAE` class in `models/vae.py`
- [x] 3b. Implement Gaussian KL to N(0,I) with optional free-bits per dimension (`clamp_min`)
- [x] 3c. KL warmup already handled by PPO's `kl_use_schedule` / `effective_vae_kl_coef`
- [x] 3d. `ForwardPredictor` reused identically (same interface)
- [x] 3e. Add `--latent_type` flag to `train.py` (spcauchy / gaussian)

### Files modified
- `models/vae.py` — added `TransitionGaussianVAE` class
- `models/config.py` — added `GaussianVAEConfig` dataclass
- `models/__init__.py` — export new classes
- `train.py` — `--latent_type` arg, VAE selection logic
- `hswvime_ppo/hswvime_ppo.py` — handle `aux_loss=None` gracefully (Gaussian has no uniformity loss)

---

## Workstream 4: Math fixes from audit

**Status:** Pending
**Est:** ~2 hrs

Apply corrections identified in the math audit of the spCauchy distribution implementation.

### Tasks
- [ ] 4a. Initialize `fc_rho.bias = -2.0` in `TransitionSCVAE.__init__` (start rho ~ 0.12, near uniform)
- [ ] 4b. Cache Gauss-Legendre nodes per device in `sc_kl_uniform()` (avoid CPU->GPU transfer per forward pass)
- [ ] 4c. Unit test: verify quadrature/asymptotic KL branches agree at rho=0.9 +/- eps for d in {8,16,32,64}
- [ ] 4d. Derive and verify d^2 KL / d rho^2 at rho=0 = 2(d-1) numerically via finite differences
- [ ] 4e. Create `test_spcauchy_math.py` with all verification tests

---

## Workstream 5: New simplified spcauchy_ppo.py

**Status:** Pending
**Est:** ~6 hrs

Clean PPO class with two optimizers, GRU memory backbone, intrinsic reward normalization,
and comprehensive logging.

### Tasks
- [ ] 5a. Implement `RunningMeanStd` (Welford's algorithm) for intrinsic reward normalization
- [ ] 5b. Add GRU memory backbone to policy: `h_mem = GRU(sg(mu), h_{t-1}_mem)`
- [ ] 5c. Policy features = `concat(sg(mu), h_mem)` -> MLP -> pi(a|.), V(.)
- [ ] 5d. Two optimizers: one for PPO params, one for transition model params (VAE + forward predictor)
- [ ] 5e. Logging: fwd_pred_error, rho_mean, rho_std, kl, recon_loss, uniformity
- [ ] 5f. Update `train.py` — final integration with all new components
- [ ] 5g. Smoke test: run on DoorButtonEnv for ~1k steps, verify no crashes

---

## Dependency graph

```
Workstream 1 (strip)
    |
    +---> Workstream 2 (forward predictor)
    |         |
    |         +---> Workstream 3 (Gaussian baseline)
    |         |
    |         +---> Workstream 5 (new PPO)
    |
    +---> Workstream 4 (math fixes, independent)
```
