"""
MMER (max-mean-episodic-reward) tracking callback.

POPGym's headline metric is MMER: the max over evaluation epochs of the
mean episodic reward at that epoch. This is the comparison axis for their
published Table 2 / Figure 3 results.

A subclass of SB3's EvalCallback that preserves all standard eval behavior
(periodic eval, best-model checkpointing, reward logging) and additionally
tracks the running max. After each eval, logs:

    eval/mmer       — running max of mean reward across all evals so far
    eval/mmer_step  — the timestep at which the current max was achieved

After training, the final MMER is available as callback.mmer.

Detection mechanism:
    EvalCallback runs an eval whenever (n_calls % eval_freq == 0). It always
    sets self.last_mean_reward after each eval. We use n_calls // eval_freq
    as a monotone "evals seen" counter and read last_mean_reward as the
    eval result. This works regardless of whether log_path is set, unlike
    self.evaluations_results (which only populates when log_path is given).

Usage:
    eval_cb = MMERCallback(eval_env, eval_freq=10000, n_eval_episodes=20)
    model.learn(total_timesteps=5_000_000, callback=eval_cb)
    print(f"Final MMER: {eval_cb.mmer:.3f}")
"""

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class MMERCallback(EvalCallback):
    """
    EvalCallback augmented with MMER tracking.

    MMER = max over evals of mean(episodic_returns at that eval).

    Eval detection uses n_calls // eval_freq, NOT self.evaluations_results
    (which is empty unless log_path is provided to the parent class).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mmer = -np.inf
        self._mmer_step = 0
        self._n_evals_seen = 0

    def _on_step(self) -> bool:
        # Run base eval logic first — EvalCallback._on_step updates
        # self.last_mean_reward whenever it just ran an eval.
        result = super()._on_step()

        if self.eval_freq is not None and self.eval_freq > 0:
            n_evals_now = self.n_calls // self.eval_freq

            if n_evals_now > self._n_evals_seen:
                # An eval just completed. self.last_mean_reward is now fresh.
                current = float(self.last_mean_reward)

                # Guard against the initial -inf and against any NaN that
                # could come out of a degenerate eval (e.g. all-zero returns).
                if np.isfinite(current) and current > self._mmer:
                    self._mmer = current
                    self._mmer_step = self.num_timesteps

                # Always log — gives a continuous TB curve even when MMER
                # plateaus, which is what we want for plot interpretability.
                if self.logger is not None:
                    self.logger.record('eval/mmer', self._mmer)
                    self.logger.record('eval/mmer_step', self._mmer_step)

                self._n_evals_seen = n_evals_now

        return result

    @property
    def mmer(self) -> float:
        """Final MMER value — read after model.learn returns."""
        return self._mmer

    @property
    def mmer_step(self) -> int:
        """Timestep at which the current MMER was achieved."""
        return self._mmer_step