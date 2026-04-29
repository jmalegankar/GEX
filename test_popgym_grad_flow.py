"""
Diagnostic G: does gradient flow from PPO loss back to the chunk-start
recurrent state?

After tests A-F + h_trace eliminated state-propagation bugs and
architectural-gap bugs, the only remaining hypothesis for "policy doesn't
learn to use memory" is that gradient signal isn't connecting through
the recurrent state during the PPO update.

Specific question: when evaluate_actions unrolls K steps from chunk-start
(lmu_h, lmu_m), does the gradient of policy/value loss w.r.t. those
chunk-start states reach them with non-zero magnitude?

If lmu_h.grad is all-zero or None after backward:
    The backward graph is severed somewhere in evaluate_actions. The
    actor never receives a gradient signal saying "this past state
    affected this action," so it learns to ignore the recurrent path.
    This would explain everything we've observed.

If lmu_h.grad is non-zero with reasonable magnitude:
    Gradient flows correctly through recurrence. The bug is downstream
    in PPO machinery (advantage computation, value clipping, return
    bootstrapping). We'd need to instrument those next.

Run:
    pytest test_popgym_grad_flow.py -v -s
"""

import warnings
from collections import deque

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.logger import Logger

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env


TASK = "popgym-RepeatPreviousMedium-v0"
N_ENVS = 4

CONFIG = dict(
    encoder_dim=64,
    hidden_size=128,
    memory_size=32,
    theta=40.0,
    n_steps=128,
    chunk_len=16,
    n_chunks_per_batch=4,
    n_epochs=1,
    lr=5e-5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    vf_coef=1.0,
    clip_range=0.2,
    clip_range_vf=0.2,
    max_grad_norm=0.5,
    target_kl=None,
    verbose=0,
    device="cpu",
)


def _build_and_collect():
    """Build a model, fill the buffer with one rollout. Return model + env."""
    env = DummyVecEnv([
        make_popgym_env(TASK, seed=42, rank=i) for i in range(N_ENVS)
    ])
    model = LMUPPO(env=env, cell_type="gru",
                   beta=0.0, beta_ep=0.0, seed=42, **CONFIG)

    # Initialize state without running learn().
    model._lmu_h, model._lmu_m = model.policy.initial_state(N_ENVS, model.device)
    model._lmu_t = None
    model._last_obs = env.reset()
    model._last_episode_starts = np.zeros(N_ENVS, dtype=bool)
    model.ep_info_buffer = deque(maxlen=100)   # real deque, not None
    model.ep_success_buffer = deque(maxlen=100)
    model._n_updates = 0
    model.num_timesteps = 0
    model._ep_bonus = None
    model._b_running_std = None
    model._phi_encoder = None
    # No-op logger — satisfies self.logger.record() calls without needing
    # tensorboard or stdout output. SB3's _setup_learn would normally make this.
    model._logger = Logger(folder=None, output_formats=[])

    # Do one rollout via the actual collect_rollouts path.
    from stable_baselines3.common.callbacks import CallbackList
    cb = CallbackList([])
    cb.init_callback(model)
    model.collect_rollouts(env, cb, model.rollout_buffer,
                           n_rollout_steps=CONFIG['n_steps'])
    return model, env


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: gradient flows through chunk-start lmu_h
# ─────────────────────────────────────────────────────────────────────────────

def test_grad_flows_to_chunk_start_lmu_h():
    """
    Fetch one chunk, run evaluate_actions with lmu_h.requires_grad_(True),
    backward through log_probs.sum() + values.sum(), inspect lmu_h.grad.

    We use both log_probs and values in the loss because real PPO loss is
    a combination of policy and value loss; either alone would only test
    one half of the gradient path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        model, env = _build_and_collect()
        try:
            # Fetch one chunk batch from the buffer.
            for batch in model.rollout_buffer.get(model.n_chunks_per_batch):
                break  # take just the first batch

            # Make lmu_h a leaf with requires_grad=True so we can read its grad.
            # (The original tensor in batch is not a leaf — it came out of to_torch
            #  which detaches.)
            lmu_h = batch.lmu_h.detach().clone().requires_grad_(True)
            lmu_m = batch.lmu_m.detach().clone().requires_grad_(True)

            # Run evaluate_actions with this chunk.
            model.policy.set_training_mode(True)
            values, log_probs, entropy, r_intrs = model.policy.evaluate_actions(
                obs_seq=batch.observations,
                lmu_h=lmu_h,
                lmu_m=lmu_m,
                episode_starts=batch.episode_starts,
                actions_seq=batch.actions,
                lmu_t=None,
            )

            # Loss = sum of log_probs + sum of values. Both should backward
            # through the unroll and hit lmu_h.
            loss = log_probs.sum() + values.sum()
            loss.backward()

            print("\n" + "=" * 70)
            print("DIAGNOSTIC G: Gradient flow to chunk-start lmu_h")
            print("=" * 70)

            print(f"\n  Chunk shape: B={batch.lmu_h.shape[0]} hidden={batch.lmu_h.shape[1]}")
            print(f"  Chunk K: {batch.episode_starts.shape[1]}")
            print(f"  log_probs.sum() = {log_probs.sum().item():.4f}")
            print(f"  values.sum()    = {values.sum().item():.4f}")

            print(f"\n  lmu_h.grad statistics:")
            if lmu_h.grad is None:
                print(f"    lmu_h.grad is None — GRADIENT NOT REACHING CHUNK START")
                print(f"    The unroll loop's backward path is severed somewhere.")
                pytest.fail("lmu_h.grad is None — backward path severed")
            else:
                g = lmu_h.grad
                print(f"    shape:    {tuple(g.shape)}")
                print(f"    mean:     {g.mean().item():+.3e}")
                print(f"    std:      {g.std().item():+.3e}")
                print(f"    abs.mean: {g.abs().mean().item():+.3e}")
                print(f"    abs.max:  {g.abs().max().item():+.3e}")
                print(f"    nonzero fraction: {(g != 0).float().mean().item():.4f}")

            print(f"\n  lmu_m.grad statistics:")
            if lmu_m.grad is None:
                print(f"    lmu_m.grad is None — m doesn't connect either")
            else:
                g = lmu_m.grad
                print(f"    shape:    {tuple(g.shape)}")
                print(f"    abs.mean: {g.abs().mean().item():+.3e}")
                print(f"    abs.max:  {g.abs().max().item():+.3e}")
                print(f"    nonzero fraction: {(g != 0).float().mean().item():.4f}")

            # ── Compare to encoder grad as reference ──
            # The encoder is on the "fast path" — every step's loss gradient
            # touches it directly through the current obs. Its grad is the
            # "good" gradient magnitude. lmu_h.grad should be in similar
            # ballpark if recurrence is being used; if it's orders of magnitude
            # smaller, recurrence is being effectively ignored.
            encoder_grads = [
                p.grad.abs().mean().item()
                for p in model.policy.encoder.parameters()
                if p.grad is not None
            ]
            actor_grads = [
                p.grad.abs().mean().item()
                for p in model.policy.actor.parameters()
                if p.grad is not None
            ]
            cell_grads = [
                p.grad.abs().mean().item()
                for p in model.policy.lmu_cell.parameters()
                if p.grad is not None and p.requires_grad
            ]

            print(f"\n  Reference parameter gradient magnitudes (abs.mean):")
            print(f"    encoder: {np.mean(encoder_grads) if encoder_grads else 0:+.3e}  "
                  f"(n_params={len(encoder_grads)})")
            print(f"    actor:   {np.mean(actor_grads) if actor_grads else 0:+.3e}  "
                  f"(n_params={len(actor_grads)})")
            print(f"    cell:    {np.mean(cell_grads) if cell_grads else 0:+.3e}  "
                  f"(n_params={len(cell_grads)})")

            # ── Interpretation ──
            lmu_h_grad_mag = (lmu_h.grad.abs().mean().item()
                              if lmu_h.grad is not None else 0.0)
            encoder_grad_mag = float(np.mean(encoder_grads)) if encoder_grads else 0.0

            print(f"\n  Interpretation:")
            if lmu_h_grad_mag == 0.0:
                print(f"    BROKEN: lmu_h gradient is zero. Backward graph severed.")
            elif lmu_h_grad_mag < 1e-10:
                print(f"    NEARLY BROKEN: lmu_h gradient is {lmu_h_grad_mag:.3e}.")
                print(f"    Effectively zero. Policy can't use memory.")
            elif encoder_grad_mag > 0 and lmu_h_grad_mag < encoder_grad_mag * 1e-3:
                ratio = lmu_h_grad_mag / encoder_grad_mag
                print(f"    LIKELY ISSUE: lmu_h grad is {ratio:.2e}× encoder grad.")
                print(f"    Recurrence is connected but gradient signal is much")
                print(f"    weaker than encoder's. Policy will preferentially")
                print(f"    learn from encoder.")
            else:
                print(f"    OK: lmu_h gradient flows with reasonable magnitude.")
                print(f"    Bug is NOT in evaluate_actions backward path.")
                print(f"    Investigate PPO machinery (advantage / value).")
        finally:
            env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: same thing but for K=1 chunk
# ─────────────────────────────────────────────────────────────────────────────

def test_grad_flows_minimal_K1():
    """
    Sanity: with K=1 (chunk_len=1, single step), gradient SHOULD flow
    cleanly. If even K=1 produces zero grad, something fundamental is
    broken in evaluate_actions's loop, not in the multi-step BPTT.

    K=1 reduces evaluate_actions to a single forward pass, so its grad
    path is the simplest possible.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        config = dict(CONFIG)
        config['chunk_len'] = 1
        config['n_chunks_per_batch'] = 4
        config['n_steps'] = 16

        env = DummyVecEnv([
            make_popgym_env(TASK, seed=42, rank=i) for i in range(N_ENVS)
        ])
        model = LMUPPO(env=env, cell_type="gru",
                       beta=0.0, beta_ep=0.0, seed=42, **config)
        model._lmu_h, model._lmu_m = model.policy.initial_state(N_ENVS, model.device)
        model._lmu_t = None
        model._last_obs = env.reset()
        model._last_episode_starts = np.zeros(N_ENVS, dtype=bool)
        model.ep_info_buffer = deque(maxlen=100)
        model.ep_success_buffer = deque(maxlen=100)
        model._n_updates = 0
        model.num_timesteps = 0
        model._ep_bonus = None
        model._b_running_std = None
        model._phi_encoder = None
        model._logger = Logger(folder=None, output_formats=[])

        from stable_baselines3.common.callbacks import CallbackList
        cb = CallbackList([])
        cb.init_callback(model)
        try:
            model.collect_rollouts(env, cb, model.rollout_buffer,
                                   n_rollout_steps=config['n_steps'])

            for batch in model.rollout_buffer.get(model.n_chunks_per_batch):
                break

            lmu_h = batch.lmu_h.detach().clone().requires_grad_(True)
            lmu_m = batch.lmu_m.detach().clone().requires_grad_(True)

            model.policy.set_training_mode(True)
            values, log_probs, entropy, r_intrs = model.policy.evaluate_actions(
                obs_seq=batch.observations,
                lmu_h=lmu_h, lmu_m=lmu_m,
                episode_starts=batch.episode_starts,
                actions_seq=batch.actions,
                lmu_t=None,
            )
            loss = log_probs.sum() + values.sum()
            loss.backward()

            print("\n" + "=" * 70)
            print("DIAGNOSTIC G2: K=1 sanity check")
            print("=" * 70)
            print(f"\n  K=1 lmu_h.grad: ", end="")
            if lmu_h.grad is None:
                print("None — backward severed at K=1")
                pytest.fail("K=1 lmu_h.grad is None")
            else:
                print(f"shape={tuple(lmu_h.grad.shape)}  "
                      f"abs.mean={lmu_h.grad.abs().mean().item():+.3e}  "
                      f"abs.max={lmu_h.grad.abs().max().item():+.3e}  "
                      f"nonzero={(lmu_h.grad != 0).float().mean().item():.4f}")
        finally:
            env.close()