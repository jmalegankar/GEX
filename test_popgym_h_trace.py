"""
Diagnostic: trace the recurrent state h through the rollout-buffer-update
pipeline and identify where (if anywhere) it diverges.

Hypothesis after E/F: the cell's hidden state isn't connecting to gradient
in a way that lets the actor learn to use it. There are three places this
can fail:

    (1) Buffer roundtrip corrupts state
        h stored at rollout time != h read back during update.
        Test: read lmu_h[chunk_start, env_idx] back from buffer, compare
        bit-for-bit to the h tensor we recorded during rollout.

    (2) Cell forward produces different outputs in train vs eval mode
        rollout uses set_training_mode(False), PPO update uses
        set_training_mode(True). spectral_norm-wrapped layers behave
        differently in these modes (during training they update SV
        estimates; during eval they use the cached SV). For our POPGym
        path, GRU/LSTM cells have no spectral_norm so this should be zero.
        Worth verifying.
        Test: replay cell forward over a chunk in both modes from the
        same starting state, compare h_k for each k.

    (3) Logits differ between rollout and update on the same obs+state
        Even if h matches, if actor(h) produces different logits in the
        two modes, PPO is regressing the policy onto a different
        distribution than the one that generated the rollout actions.
        Test: compute logits on identical (obs, h, m) inputs in both modes.

If (1) is broken: buffer/serialization bug. Fix in buffer.py.
If (2) is broken: train-mode-only modules in cell. Probably need to set
    cell to eval-mode during PPO update or accept the divergence.
If (3) is broken even when (1)+(2) are clean: there's something downstream
    of the cell (postprocessor, actor) doing something train-mode-specific.

If ALL THREE pass: the bug isn't in state propagation. It's elsewhere
    (most likely the gradient path through the unroll loop in
    evaluate_actions, or a numerical stability issue).

Run:
    pytest test_popgym_h_trace.py -v -s
"""

import warnings

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env
from stable_baselines3.common.utils import obs_as_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Config — match POPGym recipe
# ─────────────────────────────────────────────────────────────────────────────

TASK = "popgym-RepeatPreviousMedium-v0"
N_ENVS = 4   # smaller than usual — we don't need many for state diagnostics

POPGYM_CONFIG = dict(
    encoder_dim=64,
    hidden_size=128,
    memory_size=32,
    theta=40.0,
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_and_collect(chunk_len: int, n_steps: int, cell_type: str = "gru"):
    """
    Build a fresh model, run one rollout, return (model, env, rollout_record).

    rollout_record is a list of dicts, one per step, with keys:
        'obs', 'h_pre', 'm_pre', 'h_post', 'm_post', 'logits', 'action', 'value'
    where the *_pre tensors are inputs to the cell at that step (i.e. the
    state going INTO step t) and *_post are outputs (state going INTO step t+1).
    All tensors are detached and on CPU for storage.
    """
    train_env = DummyVecEnv([
        make_popgym_env(TASK, seed=42, rank=i) for i in range(N_ENVS)
    ])

    config = dict(POPGYM_CONFIG)
    config['n_steps'] = n_steps
    config['chunk_len'] = chunk_len

    model = LMUPPO(
        env=train_env,
        cell_type=cell_type,
        beta=0.0,
        beta_ep=0.0,
        seed=42,
        **config,
    )

    # Manually do _setup_learn-equivalent state initialization without running
    # learn() (which would do PPO updates and confound the test).
    model._lmu_h, model._lmu_m = model.policy.initial_state(N_ENVS, model.device)
    if model.measure == 'LegS' and model.is_lmu:
        model._lmu_t = th.ones(N_ENVS, dtype=th.int32, device=model.device)
    else:
        model._lmu_t = None
    model._last_obs = train_env.reset()
    model._last_episode_starts = np.zeros(N_ENVS, dtype=bool)
    model.ep_info_buffer = None  # not needed
    model._n_updates = 0
    model.num_timesteps = 0
    model._ep_bonus = None
    model._b_running_std = None
    model._phi_encoder = None

    # Collect one rollout while recording everything.
    rollout_record = []

    model.policy.set_training_mode(False)
    obs_now = model._last_obs
    h_now = model._lmu_h.clone()
    m_now = model._lmu_m.clone()

    for step in range(n_steps):
        with th.no_grad():
            obs_t = obs_as_tensor(obs_now, model.device)
            x = model.policy.encoder(obs_t)
            if model.measure == 'LegS' and model.is_lmu:
                h_post, m_post, _, _, _, _ = model.policy.lmu_cell(
                    x, h_now, m_now, model._lmu_t.float()
                )
            else:
                h_post, m_post, _, _, _, _ = model.policy.lmu_cell(x, h_now, m_now)
            head = model.policy._critic_input(h_post, m_post)
            logits = model.policy.actor(head)
            value = model.policy.critic(head).squeeze(-1)
            dist = th.distributions.Categorical(logits=logits)
            action = dist.sample()

        rollout_record.append({
            'step': step,
            'obs': {k: v.copy() if hasattr(v, 'copy') else v
                    for k, v in obs_now.items()},
            'h_pre':  h_now.clone().cpu(),
            'm_pre':  m_now.clone().cpu(),
            'x':      x.detach().cpu(),
            'h_post': h_post.detach().cpu(),
            'm_post': m_post.detach().cpu(),
            'logits': logits.detach().cpu(),
            'action': action.detach().cpu(),
            'value':  value.detach().cpu(),
            'episode_starts': model._last_episode_starts.copy(),
        })

        # Add to buffer.
        actions_np = action.cpu().numpy()
        new_obs, rewards, dones, infos = train_env.step(actions_np)
        log_probs = dist.log_prob(action)

        model.rollout_buffer.add(
            obs_now,
            actions_np.reshape(-1, 1),
            rewards,
            model._last_episode_starts,
            value,
            log_probs,
            h_now,
            m_now,
            lmu_t=model._lmu_t,
        )

        # Advance state.
        h_now = h_post.clone()
        m_now = m_post.clone()
        obs_now = new_obs
        model._last_episode_starts = dones

        # Reset on episode end.
        for i in np.where(dones)[0]:
            h_now[i].zero_()
            m_now[i].zero_()

    model._lmu_h = h_now
    model._lmu_m = m_now

    return model, train_env, rollout_record


def _max_abs_diff(a: th.Tensor, b: th.Tensor) -> float:
    """Element-wise max-abs-diff. Both tensors moved to CPU first."""
    if a.shape != b.shape:
        return float('nan')
    return float((a.cpu() - b.cpu()).abs().max().item())


def _replay_chunk_through_cell(
    model: LMUPPO,
    chunk_t_start: int,
    env_idx: int,
    chunk_len: int,
    rollout_record: list,
    training_mode: bool,
) -> list:
    """
    Replay the cell forward for one chunk, starting from the chunk-start
    state read out of the buffer. Returns a list of dicts mirroring
    rollout_record's structure but for the replay run.

    training_mode controls policy.set_training_mode().
    """
    model.policy.set_training_mode(training_mode)

    # Pull chunk-start state from the buffer (this is what evaluate_actions
    # would do during PPO update).
    h = th.from_numpy(
        model.rollout_buffer.lmu_h[chunk_t_start, env_idx]
    ).unsqueeze(0)   # (1, hidden)
    m = th.from_numpy(
        model.rollout_buffer.lmu_m[chunk_t_start, env_idx]
    ).unsqueeze(0)   # (1, d, C)

    replay = []
    for k in range(chunk_len):
        step = chunk_t_start + k
        rec = rollout_record[step]

        # Build single-env obs dict.
        obs_singleenv = {
            key: th.from_numpy(rec['obs'][key][env_idx:env_idx+1])
            for key in rec['obs']
        }

        # Episode start handling — same as evaluate_actions does.
        ep_start = rec['episode_starts'][env_idx]
        if ep_start:
            h = h * 0
            m = m * 0

        with th.set_grad_enabled(False):
            x = model.policy.encoder(obs_singleenv)
            if model.measure == 'LegS' and model.is_lmu:
                # We don't have lmu_t in this minimal replay; skip LegS in
                # this diagnostic.
                raise NotImplementedError("LegS not in this diagnostic")
            else:
                h_post, m_post, _, _, _, _ = model.policy.lmu_cell(x, h, m)
            head = model.policy._critic_input(h_post, m_post)
            logits = model.policy.actor(head)

        replay.append({
            'k': k,
            'h_pre':  h.clone(),
            'm_pre':  m.clone(),
            'x':      x.clone(),
            'h_post': h_post.clone(),
            'm_post': m_post.clone(),
            'logits': logits.clone(),
        })

        h, m = h_post, m_post

    return replay


def _print_chunk_diff_table(
    label: str,
    chunk_t_start: int,
    env_idx: int,
    chunk_len: int,
    rollout_record: list,
    replay: list,
):
    """Side-by-side diff table for one chunk."""
    print(f"\n  Chunk diff: {label}")
    print(f"    chunk_t_start={chunk_t_start}  env_idx={env_idx}  chunk_len={chunk_len}")
    print(f"    {'k':>3s}  {'x_diff':>12s}  {'h_pre_diff':>12s}  "
          f"{'h_post_diff':>12s}  {'logits_diff':>12s}")
    for k in range(chunk_len):
        step = chunk_t_start + k
        roll = rollout_record[step]
        rep = replay[k]

        x_diff = _max_abs_diff(rep['x'][0], roll['x'][env_idx])
        h_pre_diff = _max_abs_diff(rep['h_pre'][0], roll['h_pre'][env_idx])
        h_post_diff = _max_abs_diff(rep['h_post'][0], roll['h_post'][env_idx])
        logits_diff = _max_abs_diff(rep['logits'][0], roll['logits'][env_idx])

        print(f"    {k:>3d}  {x_diff:>12.3e}  {h_pre_diff:>12.3e}  "
              f"{h_post_diff:>12.3e}  {logits_diff:>12.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: short chunk (smoke-tested), GRU
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk_len,n_steps", [(8, 64), (128, 256)])
def test_h_trace_gru(chunk_len: int, n_steps: int):
    """
    Trace recurrent state h through buffer roundtrip + replay, both modes.

    For a working pipeline:
        - eval-mode replay h matches rollout h (h_diff < 1e-5 for all k)
        - train-mode replay h either also matches (no train-mode-specific
          modules) or shows a defined divergence pattern (train-mode dropout,
          spectral_norm SV updates, etc.)
        - logits diff at k=0 == 0 (deterministic forward from same input)

    For a broken pipeline:
        - h_diff > 1e-3 anywhere — buffer or replay inconsistent
        - logits_diff between modes when h matches — actor has a train-mode
          confound

    GRU has no spectral_norm or dropout, so eval-mode and train-mode replay
    SHOULD produce identical results bit-for-bit. Any divergence is a real bug.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        model, env, rollout_record = _build_and_collect(
            chunk_len=chunk_len, n_steps=n_steps, cell_type="gru"
        )
        try:
            # Pick chunks: t_start=0 (first chunk) and one middle chunk.
            chunks_to_test = [
                (0, 0),           # very first chunk, env 0
                (chunk_len, 1),   # second chunk, env 1
                (chunk_len * 2, 2) if (chunk_len * 2 + chunk_len) <= n_steps
                                   else (chunk_len, 0),
            ]

            print(f"\n" + "=" * 75)
            print(f"H-TRACE: GRU, chunk_len={chunk_len}, n_steps={n_steps}")
            print("=" * 75)

            print(f"\n  Number of buffer steps recorded: {len(rollout_record)}")
            print(f"  Number of envs: {N_ENVS}")
            print(f"  Hidden size: {POPGYM_CONFIG['hidden_size']}")

            # ── Buffer roundtrip check ──
            # Compare rollout-recorded h_pre[chunk_t_start] against buffer-stored
            # lmu_h[chunk_t_start]. They should be bit-equal (numpy is float32,
            # rollout tensors are float32).
            print(f"\n  ── Buffer roundtrip check ──")
            for t_start, env_idx in chunks_to_test:
                rollout_h = rollout_record[t_start]['h_pre'][env_idx]
                buffer_h = th.from_numpy(model.rollout_buffer.lmu_h[t_start, env_idx])
                diff = _max_abs_diff(rollout_h, buffer_h)
                status = "OK" if diff < 1e-6 else "FAIL"
                print(f"    t_start={t_start:3d} env={env_idx}: "
                      f"max_abs_diff={diff:.3e}  [{status}]")

            # ── Eval-mode replay (matches rollout collection) ──
            print(f"\n  ── Eval-mode replay (set_training_mode(False)) ──")
            for t_start, env_idx in chunks_to_test:
                if t_start + chunk_len > n_steps:
                    continue
                replay = _replay_chunk_through_cell(
                    model, t_start, env_idx, chunk_len, rollout_record,
                    training_mode=False,
                )
                _print_chunk_diff_table(
                    f"eval-mode, t_start={t_start} env={env_idx}",
                    t_start, env_idx, chunk_len, rollout_record, replay,
                )

            # ── Train-mode replay (matches PPO update) ──
            print(f"\n  ── Train-mode replay (set_training_mode(True)) ──")
            for t_start, env_idx in chunks_to_test:
                if t_start + chunk_len > n_steps:
                    continue
                replay = _replay_chunk_through_cell(
                    model, t_start, env_idx, chunk_len, rollout_record,
                    training_mode=True,
                )
                _print_chunk_diff_table(
                    f"train-mode, t_start={t_start} env={env_idx}",
                    t_start, env_idx, chunk_len, rollout_record, replay,
                )

            print(f"\n  Interpretation:")
            print(f"    All diffs should be < 1e-5 for GRU (no spectral_norm/dropout).")
            print(f"    Diff > 1e-3 anywhere = bug in either buffer or cell forward.")
            print(f"    eval-mode and train-mode should be IDENTICAL for GRU.")
        finally:
            env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: same trace but cell=lstm — different state-packing logic
# ─────────────────────────────────────────────────────────────────────────────

def test_h_trace_lstm():
    """
    Same trace as test 1 but with LSTM. LSTM packs c-state into m's flat
    slots — if the c-state isn't surviving the buffer roundtrip, this test
    will show m_pre_diff > 0 at k=0 of any chunk after the first.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        chunk_len = 8
        n_steps = 64
        model, env, rollout_record = _build_and_collect(
            chunk_len=chunk_len, n_steps=n_steps, cell_type="lstm"
        )
        try:
            print(f"\n" + "=" * 75)
            print(f"H-TRACE: LSTM, chunk_len={chunk_len}")
            print("=" * 75)

            chunks = [(0, 0), (chunk_len, 1), (chunk_len * 4, 2)]

            print(f"\n  ── Buffer roundtrip check (focus on m, which packs c-state) ──")
            for t_start, env_idx in chunks:
                if t_start >= n_steps:
                    continue
                rollout_m = rollout_record[t_start]['m_pre'][env_idx]
                buffer_m = th.from_numpy(model.rollout_buffer.lmu_m[t_start, env_idx])
                diff = _max_abs_diff(rollout_m, buffer_m)
                # First hidden_size flat slots = c-state.
                hidden = POPGYM_CONFIG['hidden_size']
                c_only_diff = _max_abs_diff(
                    rollout_m.reshape(-1)[:hidden],
                    buffer_m.reshape(-1)[:hidden],
                )
                status = "OK" if diff < 1e-6 else "FAIL"
                print(f"    t_start={t_start:3d} env={env_idx}: "
                      f"full_m diff={diff:.3e}  c-state diff={c_only_diff:.3e}  [{status}]")

            print(f"\n  ── Eval-mode replay ──")
            for t_start, env_idx in chunks:
                if t_start + chunk_len > n_steps:
                    continue
                replay = _replay_chunk_through_cell(
                    model, t_start, env_idx, chunk_len, rollout_record,
                    training_mode=False,
                )
                _print_chunk_diff_table(
                    f"LSTM eval, t_start={t_start} env={env_idx}",
                    t_start, env_idx, chunk_len, rollout_record, replay,
                )

            print(f"\n  ── Train-mode replay ──")
            for t_start, env_idx in chunks:
                if t_start + chunk_len > n_steps:
                    continue
                replay = _replay_chunk_through_cell(
                    model, t_start, env_idx, chunk_len, rollout_record,
                    training_mode=True,
                )
                _print_chunk_diff_table(
                    f"LSTM train, t_start={t_start} env={env_idx}",
                    t_start, env_idx, chunk_len, rollout_record, replay,
                )
        finally:
            env.close()