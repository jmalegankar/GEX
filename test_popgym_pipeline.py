"""
Smoke tests for the POPGym training pipeline.

What this catches:
    - Env wrapping (SingleKeyDict, TupleToMultiDiscrete) producing an obs
      space that flows cleanly into LMURolloutBuffer.
    - Encoder dispatch (Discrete / MultiDiscrete) on real POPGym obs shapes.
    - Cell construction + forward pass for all four cell types.
    - PPO update round-trip: rollout → buffer chunk → evaluate_actions →
      backward → W_pre Riemannian step (where applicable) → optimizer step.
    - Cell-type × phi-source compatibility validation in _setup_learn.
    - LegS step-counter plumbing.
    - SubprocVecEnv worker bringup (catches the import-popgym-in-_init bug
      class on spawn-based starts).

What this does NOT check:
    - Convergence. No reward is asserted.
    - GPU correctness. CPU-only.
    - Long-run stability. 256 timesteps is far too short for that.

Runtime budget:
    ~3-5 minutes on CPU for the full file. Each (task, cell) combo runs
    ~256 env-steps which is 2 rollouts × 2 PPO epochs at the test config.

Run:
    pytest tests/test_popgym_pipeline.py -v
    pytest tests/test_popgym_pipeline.py -v -k "matrix"
    pytest tests/test_popgym_pipeline.py::test_subproc_bringup -v
"""

import copy
from typing import List

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env


# ─────────────────────────────────────────────────────────────────────────────
# Test config
# ─────────────────────────────────────────────────────────────────────────────

TASKS: List[str] = [
    "popgym-RepeatPreviousMedium-v0",
    "popgym-AutoencodeMedium-v0",       # Tuple → MultiDiscrete via wrapper
    "popgym-CountRecallMedium-v0",
    "popgym-ConcentrationMedium-v0",
]

CELLS: List[str] = ["gated_lmu", "vanilla_lmu", "gru", "lstm"]

# Tiny config — exercises the full update path without burning compute.
SMOKE_CONFIG = dict(
    encoder_dim=32,
    hidden_size=32,
    memory_size=16,
    theta=40.0,
    n_steps=64,
    chunk_len=8,
    n_chunks_per_batch=4,
    n_epochs=2,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,           # off — entropy can mask gradient-flow bugs
    vf_coef=0.5,
    clip_range=0.2,
    clip_range_vf=0.2,
    max_grad_norm=0.5,
    target_kl=None,         # no early stop — we want the full update to run
    verbose=0,
    device="cpu",
)

TOTAL_TIMESTEPS = 256       # 2 rollouts of n_steps=64 per env at n_envs=2
N_ENVS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_vec_env(env_id: str, seed: int = 0, n_envs: int = N_ENVS):
    """In-process VecEnv for fast, traceback-friendly smoke testing."""
    return DummyVecEnv([
        make_popgym_env(env_id, seed=seed, rank=i)
        for i in range(n_envs)
    ])


def _param_signature(model: LMUPPO) -> float:
    """L2 norm of all trainable policy params, scalar. For change-detection."""
    total = 0.0
    for p in model.policy.parameters():
        if p.requires_grad:
            total += float(p.detach().double().pow(2).sum().item())
    return total ** 0.5


def _snapshot_params(model: LMUPPO):
    """Cloned dict of trainable params keyed by name. For change-detection."""
    return {
        name: p.detach().clone()
        for name, p in model.policy.named_parameters()
        if p.requires_grad
    }


def _params_changed(before: dict, after_model: LMUPPO,
                    threshold: float = 1e-6) -> List[str]:
    """
    Return list of param names whose mean-abs delta exceeds threshold.
    A non-empty list = gradients flowed somewhere. An empty list = nothing
    moved, which means a frozen module or a detached graph or no optimizer.
    """
    changed = []
    after = dict(after_model.policy.named_parameters())
    for name, p_before in before.items():
        p_after = after[name].detach()
        delta = (p_after - p_before).abs().mean().item()
        if delta > threshold:
            changed.append(name)
    return changed



def _build_model(env, cell_type: str, **overrides) -> LMUPPO:
    """
    Construct an LMUPPO with the smoke config, overridable per test.

    Uses setdefault for beta/beta_ep so callers can override them via
    `overrides` without colliding with hardcoded defaults. The previous
    pattern of passing them as direct kwargs broke whenever a test tried
    to set beta_ep > 0.
    """
    cfg = dict(SMOKE_CONFIG)
    cfg.update(overrides)
    cfg.setdefault("beta", 0.0)
    cfg.setdefault("beta_ep", 0.0)
    return LMUPPO(
        env=env,
        cell_type=cell_type,
        seed=0,
        **cfg,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1. Subprocess bringup — separate from matrix, exercises spawn-import path
# ─────────────────────────────────────────────────────────────────────────────

def test_subproc_bringup():
    """
    SubprocVecEnv worker can import popgym, register envs, reset, step, close.

    This is the only test that uses subprocess workers. Failure here means
    the `import popgym` inside _init() in popgym_envs.py is not running
    correctly per-worker, which would surface in production runs as obscure
    EOFError or env-not-found errors at `gym.make` time.
    """
    env = SubprocVecEnv([
        make_popgym_env("popgym-RepeatPreviousMedium-v0", seed=0, rank=i)
        for i in range(2)
    ])
    try:
        obs = env.reset()
        assert obs is not None
        action_space = env.action_space
        actions = np.array([action_space.sample() for _ in range(2)])
        new_obs, rewards, dones, infos = env.step(actions)
        assert new_obs is not None
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. The 4×4 matrix
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("cell", CELLS)
def test_matrix(task: str, cell: str):
    """
    Smoke: env + encoder + cell + buffer + PPO update, no E3B, β=0.

    Asserts:
        1. Construction does not raise.
        2. learn() completes without exception.
        3. At least one trainable parameter changed during training.
    """
    env = _make_dummy_vec_env(task)
    try:
        model = _build_model(env, cell_type=cell)

        before = _snapshot_params(model)

        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)

        changed = _params_changed(before, model)
        assert len(changed) > 0, (
            f"No parameter changed during training for "
            f"task={task} cell={cell}. Either the optimizer didn't run, "
            f"the graph was detached, or no gradient flowed. "
            f"Snapshot diff was below threshold for all "
            f"{sum(1 for _ in model.policy.named_parameters())} params."
        )
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. E3B smoke — exercise random_encoder phi path on the agency-rich task
# ─────────────────────────────────────────────────────────────────────────────

def test_e3b_random_encoder():
    """
    E3B with random-encoder phi on Concentration: the planned Tier 2 config.

    Exercises:
        - EllipticalEpisodicBonus construction + Sherman-Morrison update
        - RunningStd batched normalization
        - phi_encoder = random frozen encoder, used per step
        - Reward combination including the b_norm * mask term
        - M reset on episode boundaries
    """
    env = _make_dummy_vec_env("popgym-ConcentrationMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="gated_lmu",
            beta_ep=0.1,
            lambda_reg=1.0,
            phi_source="random_encoder",
        )

        before = _snapshot_params(model)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
        changed = _params_changed(before, model)
        assert len(changed) > 0

        # The frozen phi_encoder must NOT be in the trainable set.
        # If it is, gradients will leak into it and the bonus signal will
        # drift uncontrollably during training.
        if model._phi_encoder is not None:
            for p in model._phi_encoder.parameters():
                assert not p.requires_grad, (
                    "phi_encoder should be frozen after _setup_learn; "
                    "found a parameter with requires_grad=True."
                )
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Compatibility-validation regression test
# ─────────────────────────────────────────────────────────────────────────────

def test_phi_source_validation_rejects_vanilla_lmu_with_y_readout():
    """
    phi_source='y_readout' requires cell_type='gated_lmu' with read_head='dynamic'
    because it dereferences self.lmu_cell.W_query, which doesn't exist on
    vanilla_lmu (read_head='first_coef' skips the W_query allocation).

    The validation in _setup_learn must reject this combination at learn-time
    (with a clear message), not at first-rollout-time (with an opaque
    AttributeError deep inside collect_rollouts).
    """
    env = _make_dummy_vec_env("popgym-RepeatPreviousMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="vanilla_lmu",
            beta_ep=0.1,                # forces the validation to run
            phi_source="y_readout",     # incompatible with vanilla_lmu
        )
        with pytest.raises(AssertionError, match="phi_source"):
            model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    finally:
        env.close()


def test_phi_source_validation_passes_when_beta_ep_zero():
    """
    The same incompatible combination should NOT raise when beta_ep=0,
    because the validation is gated on actually using E3B. This protects
    Tier 1 sweeps where users don't think about phi_source.
    """
    env = _make_dummy_vec_env("popgym-RepeatPreviousMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="vanilla_lmu",
            beta_ep=0.0,                # E3B disabled — validation skipped
            phi_source="y_readout",     # would be incompatible if used
        )
        # Should not raise.
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. LegS smoke — exercises the per-env step-counter plumbing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason="lmu_s.py has not yet been updated with the read_head arg "
           "(intentionally deferred — see batch 2 header). Remove this "
           "decorator once LegSCell accepts read_head and exposes "
           "read_state(h, m) and head_input_size."
)
def test_legs_smoke():
    """
    LegS adds a per-env int32 step counter (_lmu_t) that is:
        - Initialized to 1 in _setup_learn
        - Threaded through forward / evaluate_actions / predict_values
        - Stored per-step in the rollout buffer
        - Reset to 1 on episode boundary
        - Incremented per step otherwise

    Asserts the run completes; param-change check covers gradient flow.
    """
    env = _make_dummy_vec_env("popgym-RepeatPreviousMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="gated_lmu",
            measure="LegS",
        )

        before = _snapshot_params(model)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)

        assert model._lmu_t is not None, (
            "_lmu_t was not allocated for measure='LegS' with an LMU cell"
        )
        assert model._lmu_t.dtype == th.int32
        assert model._lmu_t.shape == (N_ENVS,)
        assert (model._lmu_t >= 1).all()

        changed = _params_changed(before, model)
        assert len(changed) > 0
    finally:
        env.close()

def test_legs_not_allocated_for_gru():
    """
    measure='LegS' is meaningless for GRU/LSTM (they have no theta/measure
    distinction). The _lmu_t allocation must be gated on is_lmu so GRU/LSTM
    runs don't carry around an unused step-counter tensor that would cause
    a TypeError if accidentally threaded into forward().
    """
    env = _make_dummy_vec_env("popgym-RepeatPreviousMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="gru",
            measure="LegS",   # specified but should be ignored
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
        assert model._lmu_t is None, (
            "_lmu_t was allocated for cell_type='gru' even though "
            "GRU has no LegT/LegS distinction"
        )
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Vanilla-LMU forced-config sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_vanilla_lmu_forces_baseline_config():
    """
    cell_type='vanilla_lmu' must override gate_type/residual_scale/read_head
    to ('none', 0.0, 'first_coef') regardless of what was passed. Documented
    behavior of make_cell — this test pins it.
    """
    env = _make_dummy_vec_env("popgym-RepeatPreviousMedium-v0")
    try:
        model = _build_model(
            env,
            cell_type="vanilla_lmu",
            gate_type="softsign_sum",       # should be ignored
            residual_scale=0.05,            # should be ignored
            read_head="dynamic",            # should be ignored
        )
        cell = model.policy.lmu_cell
        assert cell.gate_type == "none"
        assert cell.residual_scale == 0.0
        assert cell.read_head == "first_coef"
        # And W_query must NOT have been allocated (read_head='first_coef').
        assert cell.W_query is None
    finally:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Buffer dimensionality sanity — catches obs-space vs buffer mismatch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_buffer_obs_dtype_matches_encoder_input(task: str):
    """
    A class of subtle bugs: the buffer stores observations as some dtype
    (often float32 by SB3 default), but the encoder expects int64 for
    Discrete/MultiDiscrete. If those don't agree, the encoder either
    silently does an unsafe cast or raises a dtype error inside
    evaluate_actions during PPO update (which is far from the source).

    Run one rollout, fetch a chunk from the buffer, run it through the
    encoder, assert no exception. This is essentially what evaluate_actions
    does, but isolated.
    """
    env = _make_dummy_vec_env(task)
    try:
        model = _build_model(env, cell_type="gated_lmu")
        # Trigger one rollout to fill the buffer.
        model._last_obs = env.reset()
        model._last_episode_starts = np.zeros(N_ENVS, dtype=bool)
        model._setup_learn(total_timesteps=TOTAL_TIMESTEPS, callback=None,
                           reset_num_timesteps=True, tb_log_name="smoke",
                           progress_bar=False)

        # Build a callback that does nothing.
        from stable_baselines3.common.callbacks import CallbackList
        cb = CallbackList([])
        cb.init_callback(model)

        ok = model.collect_rollouts(env, cb, model.rollout_buffer,
                                    n_rollout_steps=model.n_steps)
        assert ok

        # Fetch one chunk and run through the encoder.
        for batch in model.rollout_buffer.get(model.n_chunks_per_batch):
            obs_seq = batch.observations
            # obs_seq[key] has shape (B, K, ...). Take k=0 slice.
            obs_k = {key: obs_seq[key][:, 0] for key in obs_seq}
            x = model.policy.encoder(obs_k)
            assert x.shape[-1] == model.policy.encoder_dim
            break
    finally:
        env.close()