"""
LMU-PPO: PPO with an LMU recurrent policy.

Changes from baseline:
──────────────────────
1. collect_rollouts: unpacks 7-value policy.forward (adds r_intr).
   r_intr accumulated over rollout and logged as mean/max — NOT added to
   rewards (β=0, Step 1 validation only).

2. train: unpacks 4-value evaluate_actions (adds r_intrs).
   r_intrs logged per update. W_pre.ortho_update called after every
   loss.backward(). Periodic orthogonality check every 100 updates with
   hard SVD reset fallback.

3. predict: unpacks 7-value policy.forward, discards r_intr with _.

Step 1 diagnostics logged in collect_rollouts (all computable without
API changes to lmu_cell.forward):
  debug/r_intr_mean      — novelty signal, should decrease in corridor
  debug/r_intr_max       — upper bound check (must stay ≤ √C)
  debug/e_x_norm         — must = 1.0 always (F.normalize guard)
  debug/W_pre_ortho_error — must stay < 1e-3 (Cayley lr check)
  debug/m_norm           — memory magnitude; crash if growing/shrinking

Old lines marked  # [OLD]  and kept for diff/reversion.
"""

from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv

from .buffer import LMURolloutBuffer
from .policies import LMUActorCriticPolicy
from torch.distributions import Categorical, kl_divergence
import os


class LMUPPO(PPO):
    """
    PPO with LMU recurrent policy (gated write variant).

    Key differences from standard PPO:
      - LMU state (h, m) tracked per env during rollout
      - State reset to zero on episode boundaries
      - State stored in buffer; re-run with gradient during update
      - W_pre maintained on O(C) via Riemannian updates (ortho_update)
    """

    policy: LMUActorCriticPolicy
    rollout_buffer: LMURolloutBuffer

    def __init__(
        self,
        env:                GymEnv,
        policy=None,
        lr:                 Union[float, Schedule] = 3e-4,
        n_steps:            int = 2048,
        batch_size:         int = 256,
        n_epochs:           int = 10,
        gamma:              float = 0.999,
        gae_lambda:         float = 0.95,
        clip_range:         Union[float, Schedule] = 0.2,
        clip_range_vf:      Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef:           float = 0.01,
        vf_coef:            float = 0.5,
        max_grad_norm:      float = 0.5,
        target_kl:          Optional[float] = None,
        encoder_dim:        int = 64,
        hidden_size:        int = 64,
        memory_size:        int = 32,
        theta:              float = 50.0,
        chunk_len:          int = 16,
        n_chunks_per_batch: int = 16,
        tensorboard_log:    Optional[str] = None,
        verbose:            int = 1,
        seed:               Optional[int] = None,
        device:             Union[th.device, str] = "auto",
        _init_setup_model:  bool = True,
    ):
        self.encoder_dim        = encoder_dim
        self.hidden_size        = hidden_size
        self.memory_size        = memory_size
        self.theta              = theta
        self.chunk_len          = chunk_len
        self.n_chunks_per_batch = n_chunks_per_batch

        super().__init__(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=n_chunks_per_batch * chunk_len,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )
        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = LMUActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr=self.learning_rate if isinstance(self.learning_rate, float)
               else self.learning_rate(1.0),
            encoder_dim=self.encoder_dim,
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            theta=self.theta,
        ).to(self.device)

        self.rollout_buffer = LMURolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            encoder_dim=self.encoder_dim,
            chunk_len=self.chunk_len,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        self._lmu_h: Optional[th.Tensor] = None
        self._lmu_m: Optional[th.Tensor] = None

    # ------------------------------------------------------------------
    # Learn setup
    # ------------------------------------------------------------------

    def _setup_learn(self, total_timesteps, callback=None,
                     reset_num_timesteps=True, tb_log_name="lmu_ppo",
                     progress_bar=False):
        ret = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar
        )
        self._lmu_h, self._lmu_m = self.policy.initial_state(self.n_envs, self.device)
        return ret

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        rollout_buffer.reset()
        callback.on_rollout_start()

        # [NEW] Accumulate r_intr over the rollout — logging inside the loop
        # would overwrite the same key each step; only the last value would
        # survive to the flush.  Collect here, log the mean after the loop.
        r_intr_buf = []

        n_steps = 0
        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)
                # [OLD] actions, values, log_probs, h_new, m_new, logits_t = self.policy.forward(...)
                actions, values, log_probs, h_new, m_new, logits_t, r_intr = \
                    self.policy.forward(obs_t, self._lmu_h, self._lmu_m)

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            # [NEW] Accumulate — do NOT log inside the loop
            r_intr_buf.append(r_intr.cpu())

            n_steps += 1

            rollout_buffer.add(
                self._last_obs,
                actions_np.reshape(-1, 1),
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                self._lmu_h,
                self._lmu_m,
            )

            self._lmu_h = h_new.clone()
            self._lmu_m = m_new.clone()

            for i, done in enumerate(dones):
                if done:
                    self._lmu_h[i].zero_()
                    self._lmu_m[i].zero_()

            self._last_obs            = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_t  = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_t, self._lmu_h, self._lmu_m)

        rollout_buffer.compute_returns_and_advantage(values, dones)

        # [NEW] Step 1 diagnostics — logged once per rollout, not per step.
        # These are the minimum required to validate the gated write (Section 12).
        r_intr_all = th.stack(r_intr_buf)          # (n_steps, n_envs)
        self.logger.record("debug/r_intr_mean",      r_intr_all.mean().item())
        self.logger.record("debug/r_intr_max",        r_intr_all.max().item())

        # e_x_norm must = 1.0 always — any drift means F.normalize is broken
        e_x_norm = F.normalize(self.policy.lmu_cell.e_x, dim=0).norm().item()
        self.logger.record("debug/e_x_norm", e_x_norm)

        # W_pre orthogonality — must stay < 1e-3; fire warning if it drifts
        ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
        self.logger.record("debug/W_pre_ortho_error", ortho_err)
        if ortho_err > 1e-3:
            if self.verbose >= 1:
                print(f"  [warn] W_pre ortho error = {ortho_err:.2e} — running SVD reset")
            self.policy.lmu_cell.W_pre.reorthogonalize()

        # m_norm — memory magnitude; should be stable throughout training.
        # Monotone growth → u_actual too large. Monotone shrink → write collapsing.
        m_norm = self._lmu_m.norm(dim=(1, 2)).mean().item()
        self.logger.record("debug/m_norm", m_norm)

        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Predict (EvalCallback)
    # ------------------------------------------------------------------

    def predict(self, observation, state=None, episode_start=None,
                deterministic: bool = False):
        self.policy.set_training_mode(False)

        obs_tensor = obs_as_tensor(observation, self.device)

        if isinstance(observation, dict):
            n = next(iter(observation.values())).shape[0]
        else:
            n = observation.shape[0]

        if state is None:
            h, m = self.policy.initial_state(n, self.device)
        else:
            h, m = state

        if episode_start is not None:
            dones = th.as_tensor(episode_start, dtype=th.bool, device=self.device)
            h = h.clone(); m = m.clone()
            h[dones] = 0.0; m[dones] = 0.0

        with th.no_grad():
            # [OLD] actions, _, _, h_new, m_new, logits_t = self.policy.forward(...)
            actions, _, _, h_new, m_new, logits_t, _ = \
                self.policy.forward(obs_tensor, h, m)

        actions = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions, (h_new, m_new)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.policy.set_training_mode(True)

        lr = self.lr_schedule(self._current_progress_remaining)
        for pg in self.policy.optimizer.param_groups:
            pg["lr"] = lr

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses, value_losses, entropy_losses = [], [], []
        clip_fractions, approx_kl_divs, grad_norms = [], [], []
        r_intrs_log = []   # [NEW] accumulate across batches

        continue_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.n_chunks_per_batch):

                # [OLD] values, log_prob, entropy = self.policy.evaluate_actions(...)
                values, log_prob, entropy, r_intrs = self.policy.evaluate_actions(
                    obs_seq=batch.observations,
                    lmu_h=batch.lmu_h,
                    lmu_m=batch.lmu_m,
                    episode_starts=batch.episode_starts,
                    actions_seq=batch.actions,
                )

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio       = th.exp(log_prob - batch.old_log_prob)
                policy_loss = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.old_values + th.clamp(
                        values - batch.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss   = F.mse_loss(batch.returns, values_pred)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef  * value_loss
                )

                with th.no_grad():
                    log_ratio     = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"  Early stopping epoch {epoch}, KL={approx_kl_div:.3f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # [NEW] Riemannian update for W_pre — after backward, before
                # clip_grad_norm_.  Zeros W_pre.grad so Adam never sees it.
                # NOTE: clip_grad_norm_ will therefore see 0 for W_pre's
                # contribution, slightly underestimating the true grad norm.
                self.policy.lmu_cell.W_pre.ortho_update(lr=1e-3)

                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                # [NEW] Periodic hard orthogonality check (every 100 updates)
                if self._n_updates % 100 == 0:
                    ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
                    if ortho_err > 1e-3:
                        if self.verbose >= 1:
                            print(f"  [warn] W_pre ortho error={ortho_err:.2e} "
                                  f"at update {self._n_updates} — SVD reset")
                        self.policy.lmu_cell.W_pre.reorthogonalize()

                r_intrs_log.append(r_intrs.mean().item())   # [NEW]
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                grad_norms.append(grad_norm.item())
                self._n_updates += 1

            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/policy_loss",        np.mean(pg_losses))
        self.logger.record("train/value_loss",         np.mean(value_losses))
        self.logger.record("train/entropy_loss",       np.mean(entropy_losses))
        self.logger.record("train/approx_kl",          np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction",      np.mean(clip_fractions))
        self.logger.record("train/grad_norm",          np.mean(grad_norms))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates",          self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate",      lr)
        self.logger.record("train/clip_range",         clip_range)
        self.logger.record("debug/r_intrs_train_mean", np.mean(r_intrs_log))   # [NEW]