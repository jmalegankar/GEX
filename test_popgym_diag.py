"""
Diagnostics E and F: Postprocessor MLP experiment.

Hypothesis: the previous failures (A1, B, C, D) showed agents barely lifting
above random because the actor/critic heads were single Linear layers reading
directly from the cell's hidden state. The recurrent gradient path was so
attenuated relative to the encoder path that the actor effectively learned
to ignore the cell state and key off encoder features alone.

POPGym's published baselines architect the policy as:
    encoder → cell → postprocessor MLP → actor / critic
where postprocessor is e.g. nn.Sequential(nn.Linear(hidden, 64), nn.ReLU()).

This adds a learnable nonlinear projection between the cell output and the
heads, giving the gradient a richer path through the recurrent state.

E: GRU + Medium + postprocessor_dim=64
   Compare to A1 (no postprocessor, lift +0.036, chi2 17). If E shows
   lift > +0.2, the missing postprocessor was the issue.

F: GRU + Easy + postprocessor_dim=64
   Compare to D (no postprocessor, lift +0.056, chi2 6.2). Easy should
   solve trivially with a working architecture; lift > +0.4 expected.

If E shows partial lift but F shows clear solving, postprocessor was the
fix and Medium just needs longer training. If F also fails, the issue is
deeper than postprocessor and we instrument the cell.

Run:
    pytest test_popgym_diagnostic_ef.py -v -s
"""

import warnings

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from lmu_ppo.lmu_ppo import LMUPPO
from lmu_ppo.popgym_envs import make_popgym_env
from lmu_ppo.mmer_callback import MMERCallback


# ─────────────────────────────────────────────────────────────────────────────
# Common config (= A1's recipe + postprocessor_dim=64)
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSTIC_STEPS = 100_000
N_ENVS = 8
EVAL_FREQ = 25_000
N_EVAL_EPISODES = 10
N_BASELINE_EPISODES = 50
POSTPROCESSOR_DIM = 64

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
# Helpers (duplicated from earlier diagnostic — kept for self-containment)
# ─────────────────────────────────────────────────────────────────────────────

def _measure_random_baseline(env, n_episodes):
    n_actions = env.action_space.n
    counts = np.zeros(n_actions, dtype=np.int64)
    rets = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0.0
        done = False
        while not done:
            action = np.array([env.action_space.sample()])
            counts[int(action[0])] += 1
            obs, reward, done_arr, info = env.step(action)
            ret += float(reward[0])
            done = bool(done_arr[0])
        rets.append(ret)
    rets = np.asarray(rets)
    total = counts.sum()
    return {"mean_reward": float(rets.mean()), "std_reward": float(rets.std()),
            "action_counts": counts, "action_dist": counts / max(total, 1)}


def _measure_policy_actions(model, env, n_episodes):
    n_actions = env.action_space.n
    counts = np.zeros(n_actions, dtype=np.int64)
    rets = []
    for _ in range(n_episodes):
        obs = env.reset()
        state = None
        episode_start = np.array([True])
        ret = 0.0
        done = False
        while not done:
            action, state = model.predict(
                obs, state=state, episode_start=episode_start, deterministic=False
            )
            counts[int(action[0])] += 1
            obs, reward, done_arr, info = env.step(action)
            ret += float(reward[0])
            done = bool(done_arr[0])
            episode_start = np.array([done])
        rets.append(ret)
    rets = np.asarray(rets)
    total = counts.sum()
    return {"mean_reward": float(rets.mean()), "std_reward": float(rets.std()),
            "action_counts": counts, "action_dist": counts / max(total, 1)}


def _chi_squared_uniform(counts):
    n = len(counts)
    total = counts.sum()
    if total / n < 1:
        return {"chi2": 0.0, "consistent_with_uniform": True}
    expected = total / n
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    crit_p05 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070}
    critical = crit_p05.get(n - 1, 11.070)
    return {"chi2": chi2, "dof": n - 1, "critical_p05": critical,
            "consistent_with_uniform": chi2 < critical}


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


def _run_diagnostic(label, task, cell_type, config_overrides, expected_solved_lift):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        config = dict(POPGYM_RECIPE_BASE)
        config.update(config_overrides)

        baseline_env = DummyVecEnv([make_popgym_env(task, seed=999, rank=0)])
        try:
            baseline = _measure_random_baseline(baseline_env, N_BASELINE_EPISODES)
        finally:
            baseline_env.close()

        train_env = DummyVecEnv([
            make_popgym_env(task, seed=0, rank=i) for i in range(N_ENVS)
        ])
        eval_env = DummyVecEnv([make_popgym_env(task, seed=1000, rank=0)])

        try:
            cb = _RewardTrackingCallback(
                eval_env,
                eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                n_eval_episodes=N_EVAL_EPISODES, verbose=0,
            )
            model = LMUPPO(
                env=train_env, cell_type=cell_type,
                beta=0.0, beta_ep=0.0, seed=0, **config,
            )
            model.learn(total_timesteps=DIAGNOSTIC_STEPS, callback=cb,
                        progress_bar=True)

            ep_buf = model.ep_info_buffer
            rollout_returns = [ep['r'] for ep in ep_buf if 'r' in ep]
            rollout_mean = float(np.mean(rollout_returns)) if rollout_returns else float('nan')

            mean_eval = (
                float(np.mean([h[1] for h in cb.history])) if cb.history else float('nan')
            )

            trained = _measure_policy_actions(model, eval_env, n_episodes=20)
            chi2 = _chi_squared_uniform(trained["action_counts"])

            print("\n" + "=" * 70)
            print(f"DIAGNOSTIC {label}")
            print("=" * 70)
            print(f"  task:    {task}")
            print(f"  cell:    {cell_type}")
            print(f"  config overrides: {config_overrides}")
            print()

            print(f"  Random baseline:  mean={baseline['mean_reward']:+.3f}")
            print(f"  Rollout reward:   mean={rollout_mean:+.3f}")
            print(f"  Eval reward:      mean={mean_eval:+.3f}")
            print(f"  Lift over random: rollout="
                  f"{rollout_mean - baseline['mean_reward']:+.3f}  "
                  f"eval={mean_eval - baseline['mean_reward']:+.3f}")
            print(f"  Expected lift if task solved: ~{expected_solved_lift:+.2f}")

            print(f"\n  Trained-policy action distribution:")
            for a in range(len(trained["action_counts"])):
                count = trained["action_counts"][a]
                pct = trained["action_dist"][a]
                bar = "#" * int(40 * pct)
                print(f"    action {a}: {count:5d}  ({pct:.2%})  {bar}")
            print(f"  Chi2 vs uniform: chi2={chi2['chi2']:.3f}  "
                  f"consistent_with_uniform={chi2['consistent_with_uniform']}")

            lift = mean_eval - baseline['mean_reward']
            solved_threshold = 0.6 * expected_solved_lift

            print(f"\n  Interpretation:")
            if lift > solved_threshold:
                interp = "solved"
                print(f"    GENUINELY LEARNING (lift {lift:+.3f} > {solved_threshold:+.2f}).")
            elif lift > 0.10:
                interp = "partial"
                print(f"    PARTIAL LEARNING (lift {lift:+.3f}).")
            elif chi2['consistent_with_uniform']:
                interp = "no_learning"
                print(f"    NO LEARNING — uniform actions.")
            else:
                interp = "marginal_prior"
                print(f"    MARGINAL-PRIOR ONLY — non-uniform actions but tiny lift.")

            return {"label": label, "lift": lift, "chi2": chi2["chi2"],
                    "interp": interp}
        finally:
            train_env.close()
            eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# E: GRU + Medium + postprocessor
# ─────────────────────────────────────────────────────────────────────────────

def test_E_gru_medium_with_postprocessor():
    """
    Same as A1 but with postprocessor_dim=64.
    Compare lift and chi2 to A1's (+0.036, 17.0).
    """
    result = _run_diagnostic(
        label="E: GRU + Medium + postprocessor=64",
        task="popgym-RepeatPreviousMedium-v0",
        cell_type="gru",
        config_overrides={"postprocessor_dim": POSTPROCESSOR_DIM},
        expected_solved_lift=0.5,
    )
    print(f"\n  E summary: lift={result['lift']:+.3f}  chi2={result['chi2']:.1f}  "
          f"interp={result['interp']}")
    print(f"  A1 reference: lift=+0.036  chi2=17.0  interp=marginal_prior")


# ─────────────────────────────────────────────────────────────────────────────
# F: GRU + Easy + postprocessor
# ─────────────────────────────────────────────────────────────────────────────

def test_F_gru_easy_with_postprocessor():
    """
    Same as D but with postprocessor_dim=64.
    Compare lift and chi2 to D's (+0.056, 6.2).

    Easy (k=1) should solve trivially with a working architecture. If F shows
    lift > +0.4, postprocessor was the fix. If F still shows tiny lift,
    the issue is deeper than postprocessor.
    """
    result = _run_diagnostic(
        label="F: GRU + Easy + postprocessor=64",
        task="popgym-RepeatPreviousEasy-v0",
        cell_type="gru",
        config_overrides={"postprocessor_dim": POSTPROCESSOR_DIM},
        expected_solved_lift=0.7,
    )
    print(f"\n  F summary: lift={result['lift']:+.3f}  chi2={result['chi2']:.1f}  "
          f"interp={result['interp']}")
    print(f"  D reference: lift=+0.056  chi2=6.2  interp=no_learning")