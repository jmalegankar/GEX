"""
Diagnostics B, C, D: isolate whether recurrence is being used at all.

A1 showed: GRU on RepeatPreviousMedium with POPGym recipe lifts reward +0.036
above random at 100k steps, with statistically-significant non-uniform action
distribution (chi2 = 17, p < 0.001). Two interpretations:

    (a) Agent learned to use memory; just slow + budget-limited.
    (b) Agent learned the marginal distribution of correct answers
        (= "always-pick-the-most-common-suit" baseline) and stopped.

These three tests separate (a) from (b):

    B: GRU + Medium + chunk_len=16 (vs A1's chunk_len=128)
       Same per-rollout compute (n_steps reduced proportionally).
       If reward + chi2 are unchanged, the long BPTT window doesn't matter,
       which means the agent isn't using recurrent state across long
       horizons. Strong evidence for (b).

    C: LSTM + Medium + chunk_len=128 (same as A1 but different cell)
       If LSTM matches GRU's small lift, recurrence-class is the issue, not
       cell-specific. If LSTM clearly outperforms, GRU has a specific bug.

    D: GRU + RepeatPreviousEasy (k=1) + chunk_len=128 (otherwise = A1)
       k=1 is one-step recall — the simplest possible memory task.
       Any working recurrence should solve this in 100k steps.
       If D also lifts only +0.04, recurrence is fundamentally broken in
       our pipeline. If D shows +0.3 to +0.5, Medium just needs more budget.

Run all three:
    pytest test_popgym_diagnostic_bcd.py -v -s

Each test is independent — failure in one doesn't block others. Each prints
its own diagnostic block at the end.
"""

import warnings

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env
from lmu_ppo.mmer_callback import MMERCallback


# ─────────────────────────────────────────────────────────────────────────────
# Common config
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSTIC_STEPS = 100_000
N_ENVS = 8
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 10
N_BASELINE_EPISODES = 50

POPGYM_RECIPE_BASE = dict(
    encoder_dim=64,
    hidden_size=128,
    memory_size=32,
    theta=40.0,
    n_steps=2048,
    chunk_len=128,
    n_chunks_per_batch=32,
    n_epochs=30,
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
# Helpers (same as test_popgym_diagnostic.py — duplicated for self-containment)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_random_baseline(env, n_episodes):
    n_actions = env.action_space.n
    action_counts = np.zeros(n_actions, dtype=np.int64)
    episode_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action = np.array([env.action_space.sample()])
            action_counts[int(action[0])] += 1
            obs, reward, done_arr, info = env.step(action)
            ep_return += float(reward[0])
            done = bool(done_arr[0])
        episode_returns.append(ep_return)
    returns = np.asarray(episode_returns)
    total = action_counts.sum()
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(returns.mean()),
        "std_reward": float(returns.std()),
        "action_counts": action_counts,
        "action_dist": action_counts / max(total, 1),
    }


def _measure_policy_actions(model, env, n_episodes):
    n_actions = env.action_space.n
    action_counts = np.zeros(n_actions, dtype=np.int64)
    episode_returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        state = None
        episode_start = np.array([True])
        ep_return = 0.0
        done = False
        while not done:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=False
            )
            action_counts[int(action[0])] += 1
            obs, reward, done_arr, info = env.step(action)
            ep_return += float(reward[0])
            done = bool(done_arr[0])
            episode_start = np.array([done])
        episode_returns.append(ep_return)
    returns = np.asarray(episode_returns)
    total = action_counts.sum()
    return {
        "n_episodes": n_episodes,
        "mean_reward": float(returns.mean()),
        "std_reward": float(returns.std()),
        "action_counts": action_counts,
        "action_dist": action_counts / max(total, 1),
    }


def _chi_squared_uniform(counts):
    n = len(counts)
    total = counts.sum()
    expected = total / n
    if expected < 1:
        return {"chi2": 0.0, "consistent_with_uniform": True}
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    crit_p05 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070}
    critical = crit_p05.get(n - 1, 11.070)
    return {
        "chi2": chi2,
        "dof": n - 1,
        "critical_p05": critical,
        "consistent_with_uniform": chi2 < critical,
    }


class _RewardTrackingCallback(MMERCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self._n_evals_seen_local = 0

    def _on_step(self):
        result = super()._on_step()
        if self._n_evals_seen > self._n_evals_seen_local:
            self.history.append(
                (int(self.num_timesteps), float(self.last_mean_reward))
            )
            self._n_evals_seen_local = self._n_evals_seen
        return result


def _run_one_diagnostic(
    label: str,
    task: str,
    cell_type: str,
    config_overrides: dict,
    expected_lift_for_solved: float,
):
    """
    Run one diagnostic: train for DIAGNOSTIC_STEPS, measure random baseline
    on same env, print structured report, return interpretation dict.

    No pytest assertions — diagnostic only. The test functions wrap this
    and print results.

    expected_lift_for_solved is the reward delta above random that we'd
    expect IF the task were genuinely solved. Used in the interpretation
    block to distinguish "small lift" from "task solved." Per-task:
        RepeatPreviousMedium k=4: solved ≈ +0.5 (75% accuracy = 3/4 correct)
        RepeatPreviousEasy k=1:   solved ≈ +0.7 (90%+ accuracy is achievable)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        config = dict(POPGYM_RECIPE_BASE)
        config.update(config_overrides)

        # Random baseline on this task.
        baseline_env = DummyVecEnv([make_popgym_env(task, seed=999, rank=0)])
        try:
            baseline = _measure_random_baseline(baseline_env, N_BASELINE_EPISODES)
        finally:
            baseline_env.close()

        # Train.
        train_env = DummyVecEnv([
            make_popgym_env(task, seed=0, rank=i) for i in range(N_ENVS)
        ])
        eval_env = DummyVecEnv([make_popgym_env(task, seed=1000, rank=0)])

        try:
            cb = _RewardTrackingCallback(
                eval_env,
                eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                n_eval_episodes=N_EVAL_EPISODES,
                verbose=0,
            )
            model = LMUPPO(
                env=train_env, cell_type=cell_type,
                beta=0.0, beta_ep=0.0, seed=0, **config,
            )
            model.learn(
                total_timesteps=DIAGNOSTIC_STEPS,
                callback=cb, progress_bar=True,
            )

            # Pull rollout reward from SB3's ep_info_buffer.
            ep_buf = model.ep_info_buffer
            rollout_returns = [ep['r'] for ep in ep_buf if 'r' in ep]
            rollout_mean = float(np.mean(rollout_returns)) if rollout_returns else float('nan')

            # Eval history.
            eval_history = cb.history
            mean_eval = (
                float(np.mean([h[1] for h in eval_history])) if eval_history else float('nan')
            )

            # Trained-policy action histogram.
            trained = _measure_policy_actions(model, eval_env, n_episodes=20)
            chi2 = _chi_squared_uniform(trained["action_counts"])

            # ── Print ──
            print("\n" + "=" * 70)
            print(f"DIAGNOSTIC {label}")
            print("=" * 70)
            print(f"  task:    {task}")
            print(f"  cell:    {cell_type}")
            print(f"  config overrides: {config_overrides}")
            print()

            print(f"  Random baseline:  mean={baseline['mean_reward']:+.3f} ± "
                  f"{baseline['std_reward']:.3f}")
            print(f"  Rollout reward:   mean={rollout_mean:+.3f}")
            print(f"  Eval reward:      mean={mean_eval:+.3f}")
            print(f"  Lift over random: rollout={rollout_mean - baseline['mean_reward']:+.3f}  "
                  f"eval={mean_eval - baseline['mean_reward']:+.3f}")
            print(f"  Expected lift if task solved: ~{expected_lift_for_solved:+.2f}")

            print(f"\n  Trained-policy action distribution:")
            for a in range(len(trained["action_counts"])):
                count = trained["action_counts"][a]
                pct = trained["action_dist"][a]
                bar = "#" * int(40 * pct)
                print(f"    action {a}: {count:5d}  ({pct:.2%})  {bar}")
            print(f"  Chi2 vs uniform: chi2={chi2['chi2']:.3f}  "
                  f"consistent_with_uniform={chi2['consistent_with_uniform']}")

            # ── Interpretation ──
            lift = mean_eval - baseline['mean_reward']
            lift_threshold_solved = 0.6 * expected_lift_for_solved   # 60% of solved
            lift_threshold_marginal = 0.10                             # tiny lift

            print(f"\n  Interpretation:")
            if lift > lift_threshold_solved:
                print(f"    GENUINELY LEARNING the task (lift {lift:+.3f} >= "
                      f"{lift_threshold_solved:+.2f}). Recurrence is being used.")
                interpretation = "solved"
            elif lift > lift_threshold_marginal:
                print(f"    PARTIAL LEARNING (lift {lift:+.3f}, between marginal "
                      f"and solved thresholds). Some memory use but not converged.")
                interpretation = "partial"
            elif chi2['consistent_with_uniform']:
                print(f"    NO LEARNING — actions are uniform. Either policy")
                print(f"    head not connected, or no gradient reaching it.")
                interpretation = "no_learning"
            else:
                print(f"    MARGINAL-PRIOR LEARNING — actions are non-uniform")
                print(f"    (chi2={chi2['chi2']:.1f}) but reward lift is tiny")
                print(f"    ({lift:+.3f}). Agent learned the marginal action")
                print(f"    distribution without using recurrence.")
                interpretation = "marginal_prior"

            return {
                "label": label,
                "task": task,
                "cell": cell_type,
                "rollout_mean": rollout_mean,
                "eval_mean": mean_eval,
                "random_baseline": baseline['mean_reward'],
                "lift": lift,
                "chi2": chi2['chi2'],
                "consistent_with_uniform": chi2['consistent_with_uniform'],
                "interpretation": interpretation,
            }
        finally:
            train_env.close()
            eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# B: chunk_len=16 (vs A1's chunk_len=128). n_steps reduced to keep
#    chunks-per-rollout-per-env constant at 16 (= 256/16 = 2048/128).
# ─────────────────────────────────────────────────────────────────────────────

def test_B_chunk_len_16():
    """
    Same as A1 but with chunk_len=16 instead of 128. n_steps=256 to keep
    chunks-per-env constant (so per-rollout SGD compute is the same).

    If reward and chi2 match A1 (~+0.04 lift, chi2 ~17), the long BPTT
    window from A1 was not used — agent learned the same marginal-prior
    pattern with 16-step memory as with 128-step memory.

    If B is significantly worse than A1, the long BPTT was helping (and
    we have evidence that recurrence IS being used, just not effectively).
    """
    result = _run_one_diagnostic(
        label="B: chunk_len=16 on RepeatPreviousMedium",
        task="popgym-RepeatPreviousMedium-v0",
        cell_type="gru",
        config_overrides=dict(
            n_steps=256,            # 16 chunks per env, matches A1 ratio
            chunk_len=16,
            n_chunks_per_batch=32,  # keep batch shape similar
        ),
        expected_lift_for_solved=0.5,  # k=4, ~75% accuracy = +0.5 above random
    )

    # Print summary line for cross-test comparison.
    print(f"\n  B summary: lift={result['lift']:+.3f}  chi2={result['chi2']:.1f}  "
          f"interp={result['interpretation']}")


# ─────────────────────────────────────────────────────────────────────────────
# C: LSTM (vs A1's GRU), same task and config otherwise
# ─────────────────────────────────────────────────────────────────────────────

def test_C_lstm_medium():
    """
    Same as A1 but cell_type='lstm'. If LSTM matches GRU's small lift
    (+0.04, chi2 ~17), the issue isn't cell-specific — recurrence in
    general isn't being used.

    If LSTM significantly outperforms (+0.2 or more), GRU has a specific
    issue in our pipeline (e.g., GRUCellWrapper's m=zero-pad design has
    a subtle gradient problem we missed).

    LSTM packs cell state c into m's first hidden_size flat slots — so
    this also indirectly tests whether the LSTM-specific buffer handling
    works at chunk_len=128 on real PPO updates (smoke ran chunk_len=8).
    """
    result = _run_one_diagnostic(
        label="C: LSTM on RepeatPreviousMedium",
        task="popgym-RepeatPreviousMedium-v0",
        cell_type="lstm",
        config_overrides={},  # all defaults from A1
        expected_lift_for_solved=0.5,
    )

    print(f"\n  C summary: lift={result['lift']:+.3f}  chi2={result['chi2']:.1f}  "
          f"interp={result['interpretation']}")


# ─────────────────────────────────────────────────────────────────────────────
# D: RepeatPreviousEasy (k=1) — the simplest memory task
# ─────────────────────────────────────────────────────────────────────────────

def test_D_easy_task():
    """
    Same as A1 but on RepeatPreviousEasy (k=1, one-step lookback).

    This is the cleanest discriminator. RepeatPreviousEasy is dead simple:
    output the suit of the IMMEDIATELY previous card. Any working recurrence
    should solve this in 100k steps.

    If D shows lift > +0.4: recurrence works fine; Medium just needs more
        budget. The MMER plateau on Medium is a budget/optimization issue.

    If D shows lift ~ +0.04 like A1: recurrence is fundamentally broken
        in our pipeline. The chi2 = 17 result on A1 was learning the
        marginal answer distribution, NOT learning the task. This is the
        bad outcome — implies a real bug in the cell/buffer/PPO interaction
        that has to be found before any sweep.
    """
    result = _run_one_diagnostic(
        label="D: GRU on RepeatPreviousEasy (k=1)",
        task="popgym-RepeatPreviousEasy-v0",
        cell_type="gru",
        config_overrides={},  # default POPGym recipe
        expected_lift_for_solved=0.7,  # k=1 should be near-perfect
    )

    print(f"\n  D summary: lift={result['lift']:+.3f}  chi2={result['chi2']:.1f}  "
          f"interp={result['interpretation']}")