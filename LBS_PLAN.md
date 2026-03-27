# LBS Branch: Summary & Plan

## What WynerLBSVAE Does

Separates deterministic state (h_t via GRU) from stochastic latent (z via posterior/prior):

- **Prior** predicts from h_{t-1} (before GRU sees μ_t)
- **Posterior** infers from h_t (after GRU incorporates μ_t), optionally + μ_{t+1}
- **Memory buffer** stores h_t (deterministic), not z_mu (stochastic)
- **KL = Bayesian surprise** because prior and posterior condition on different information states

```
Policy path:  h_prev → GRU(μ_t) → h_t → attention(h_t, μ) → actor/critic
Wyner train:  h_prev → prior(h_prev) → prior_mu, prior_logvar
              h_prev → GRU(μ_t) → h_t → posterior(h_t, μ_{t+1}) → post_mu, post_logvar
              z ~ posterior → decoder(z, timestep) → recon
              KL(posterior || prior) = intrinsic reward
QA loss:      z sampled from posterior → decode(z, past_timestep) → reconstruct past transitions
```

## Experimental Results

| Run | Env | Key Observations |
|-----|-----|-----------------|
| door_button 500k (episodic+intrinsic) | door_button | KL ~12-14 (genuine surprise). Intrinsic reward grew to ~0.35, drowning extrinsic (~0.02). ep_rew collapsed after 150k. wyner_recon rose from 0.1→0.9. |
| MemoryS7 no_episodic | MemoryS7 | Removed episodic bonus, pure KL intrinsic. Same collapse pattern. |
| MemoryS7 anneal | MemoryS7 | Linear decay of intrinsic_scale. KL spiked to 8+, recon→1.2, explained_var→-4. Worse than no-anneal. |
| PPO baseline | MemoryS7 | Optimal episode length but ~0.5 reward (random 2-door guess). Can navigate but can't remember. |

## Root Cause: Shared Optimizer Gradient Conflict

During `evaluate_actions` (gradients ON), PPO losses backprop through `features_extractor` → into GRU weights and VAE encoder. In the same backward pass, Wyner/VAE losses also push gradients through those weights.

The GRU is optimized for two conflicting objectives:
1. "produce h_t that makes the policy work" (PPO)
2. "produce h_t that enables transition reconstruction" (Wyner recon/QA)

As the policy shifts, the GRU gets dragged along, and reconstruction/QA losses rise. This explains why wyner_recon goes up, QA goes up, and explained_variance goes down over time.

## Fix Plan

### Step 1: Detach fix in extract_features ✅
Add `.detach()` on `new_memory` and `mu` before passing to `features_extractor`. PPO treats h_t and μ as frozen feature inputs. Wyner GRU only gets gradients from its own losses.

**Status**: Implemented. Smoke test passed (5k steps). Awaiting full 200k run.

**Expected**: wyner_recon_loss stops rising, explained_variance stabilizes.

### Step 2: Intrinsic reward normalization
Replace `intrinsic_scale * KL` with `KL / running_std` (RND-style). Remove intrinsic_anneal_steps.

**Expected**: intrinsic_reward_mean stays bounded relative to extrinsic.

### Step 3: SlotMemory integration (replace QA)

Replace the QA mechanism with a fixed-size addressable slot memory that uses Wyner KL as a surprise-gated write signal. The policy reads from slots via cross-attention instead of attending only to the latest h_t.

#### Architecture

```
SlotMemory(num_slots=K, slot_dim=wyner_latent_dim, mu_dim=vae_latent_dim)

Write path (during rollout collection):
  kl = wyner_kl_per_step          # from Wyner forward pass
  gate = sigmoid(gate_scale * kl)  # surprise gate (detached mode)
  slots[lru_idx] = gate * z + (1 - gate) * slots[lru_idx]
  lru_ages updated

Read path (replaces HSWVIMEFeaturesExtractor):
  attn_out = cross_attention(query=μ, key=slots, value=slots)
  features = [attn_out || μ]       # same shape as before
```

#### Changes required

1. **Add `models/slot_memory.py`** — SlotMemory module with:
   - `write(z, mu, kl)` — soft-gated LRU write, returns gate values
   - `read(mu, slots)` — cross-attention read, returns (B, slot_dim + mu_dim)
   - Two gate modes: "detached" (sigmoid of scaled KL) and "learned" (MLP on mu)
   - `gate_correlation_loss()` for learned mode alignment

2. **Update `hswvime_ppo/policies.py`**:
   - Replace `HSWVIMEFeaturesExtractor` forward with `SlotMemory.read()`
   - `extract_features` takes slot state instead of single h_t
   - Track separate Wyner h_t (for Wyner GRU) and slot memory (for policy)

3. **Update `hswvime_ppo/hswvime_ppo.py`**:
   - Add slot write step in `collect_rollouts` after Wyner forward
   - Remove QA mechanism entirely (QASampler, QA buffer fields, QA loss in train())
   - Log slot gate stats (mean gate, num writes, slot utilization)

4. **Update `hswvime_ppo/buffer.py`**:
   - Replace single memory tensor with slot memory tensor (B, K, slot_dim)
   - Remove QA-specific buffer fields

5. **Update `train.py`**:
   - Add CLI args: `--num_slots`, `--gate_mode`, `--gate_scale`, `--gate_threshold`
   - Pass slot memory kwargs through policy

#### Experiment plan
- Run with `gate_mode="detached"` first (simpler, no extra loss)
- Compare against Step 1+2 baseline on MemoryS7
- If detached works, try `gate_mode="learned"` with `gate_correlation_loss`

**Expected**: Slot memory retains task-relevant observations (e.g., initial door color in MemoryS7). Policy reads from slots to break above 0.5 reward.
