"""
LMU-PPO: PPO with an LMU recurrent policy.

Inherits from SB3's PPO for: VecEnv management, callbacks, TensorBoard
logging, learning rate schedules, and clip_range scheduling.

Overrides: _setup_model, collect_rollouts, train.
The policy is our standalone LMUActorCriticPolicy (not SB3's ActorCriticPolicy).
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


class LMUPPO(PPO):
    """
    PPO with LMU recurrent policy.

    Key differences from standard PPO:
      - LMU state (h, m) tracked per env during rollout
      - State reset to zero on episode boundaries
      - State stored in buffer; re-run with gradient during update
    """

    policy: LMUActorCriticPolicy
    rollout_buffer: LMURolloutBuffer

    def __init__(
        self,
        env:                GymEnv,
        lr:                 Union[float, Schedule] = 3e-4,
        n_steps:            int = 2048,
        batch_size:         int = 256,
        n_epochs:           int = 10,
        gamma:              float = 0.999,      # high gamma for long-horizon credit
        gae_lambda:         float = 0.95,
        clip_range:         Union[float, Schedule] = 0.2,
        clip_range_vf:      Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef:           float = 0.01,
        vf_coef:            float = 0.5,
        max_grad_norm:      float = 0.5,
        target_kl:          Optional[float] = None,
        # LMU / policy architecture
        encoder_dim:        int = 64,
        hidden_size:        int = 64,
        memory_size:        int = 32,
        theta:              float = 50.0,
        # SB3 plumbing
        tensorboard_log:    Optional[str] = None,
        verbose:            int = 1,
        seed:               Optional[int] = None,
        device:             Union[th.device, str] = "auto",
    ):
        # Store arch params before super().__init__ so _setup_model can use them
        self.encoder_dim  = encoder_dim
        self.hidden_size  = hidden_size
        self.memory_size  = memory_size
        self.theta        = theta

        # SB3's PPO.__init__ calls _setup_model at the end — we pass a dummy
        # policy string so it doesn't crash before we override _setup_model.
        super().__init__(
            policy="MultiInputPolicy",  # placeholder; overridden in _setup_model
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
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
            _init_setup_model=False,  # we call it ourselves below
        )
        self._setup_model()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # Minimal version of SB3's _setup_model:
        # set random seeds, resolve device, build policy and buffer.
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
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Clip range schedule
        self.clip_range    = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # LMU state per env — initialised in _setup_learn
        self._lmu_h: Optional[th.Tensor] = None
        self._lmu_m: Optional[th.Tensor] = None

    # ------------------------------------------------------------------
    # Learn setup  (called at the start of .learn())
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

    def collect_rollouts(
        self,
        env:              VecEnv,
        callback:         BaseCallback,
        rollout_buffer:   LMURolloutBuffer,
        n_rollout_steps:  int,
    ) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_t  = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, h_new, m_new = self.policy.forward(
                    obs_t, self._lmu_h, self._lmu_m
                )

            actions_np = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            # Timeout bootstrapping (truncated episodes)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    with th.no_grad():
                        raw = infos[idx]["terminal_observation"]
                        term_obs = obs_as_tensor(
                            {k: np.array(v)[None] for k, v in raw.items()}, self.device
                        )
                        term_val = self.policy.predict_values(
                            term_obs, h_new[idx:idx+1], m_new[idx:idx+1]
                        ).item()
                    rewards[idx] += self.gamma * term_val

            rollout_buffer.add(
                self._last_obs,
                actions_np.reshape(-1, 1),   # (n_envs, 1) for Discrete
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                self._lmu_h,                 # h_{t-1} stored, not h_t
                self._lmu_m,
            )

            # Advance state; zero out finished episodes
            self._lmu_h = h_new.clone()
            self._lmu_m = m_new.clone()
            for idx, done in enumerate(dones):
                if done:
                    self._lmu_h[idx].zero_()
                    self._lmu_m[idx].zero_()

            self._last_obs            = new_obs
            self._last_episode_starts = dones

        # GAE bootstrap
        with th.no_grad():
            obs_t  = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_t, self._lmu_h, self._lmu_m)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )

        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
  # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        """
        Override BaseAlgorithm.predict so EvalCallback works.
        state = (h, m) tuple; None means start of episode.
        """
        self.policy.set_training_mode(False)

        # obs_as_tensor handles both dict and array observations
        obs_tensor = obs_as_tensor(observation, self.device)

        # Initialise or unpack recurrent state
        # observation may be (n_envs, ...) so infer batch size from it
        if isinstance(observation, dict):
            n = next(iter(observation.values())).shape[0]
        else:
            n = observation.shape[0]

        if state is None:
            h, m = self.policy.initial_state(n, self.device)
        else:
            h, m = state

        # Reset state for finished episodes
        if episode_start is not None:
            dones = th.as_tensor(episode_start, dtype=th.bool, device=self.device)
            h = h.clone()
            m = m.clone()
            h[dones] = 0.0
            m[dones] = 0.0

        with th.no_grad():
            actions, _, _, h_new, m_new = self.policy.forward(obs_tensor, h, m)

        actions = actions.cpu().numpy()

        # Clip for continuous action spaces (not needed for Discrete but harmless)
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions, (h_new, m_new)
    
    def train(self) -> None:
        self.policy.set_training_mode(True)

        # Update learning rate
        lr = self.lr_schedule(self._current_progress_remaining)
        for pg in self.policy.optimizer.param_groups:
            pg["lr"] = lr

        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses, value_losses, entropy_losses = [], [], []
        clip_fractions, approx_kl_divs, grad_norms = [], [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                actions = batch.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    batch.observations,
                    batch.lmu_h,
                    batch.lmu_m,
                    actions,
                )

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss (clipped surrogate)
                ratio          = th.exp(log_prob - batch.old_log_prob)
                policy_loss    = -th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ).mean()

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.old_values + th.clamp(
                        values - batch.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(batch.returns, values_pred)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.ent_coef  * entropy_loss
                    + self.vf_coef   * value_loss
                )

                # Early stopping on KL divergence
                with th.no_grad():
                    log_ratio     = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"  Early stopping at epoch {epoch}, approx KL={approx_kl_div:.3f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

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

        self.logger.record("train/policy_loss",       np.mean(pg_losses))
        self.logger.record("train/value_loss",        np.mean(value_losses))
        self.logger.record("train/entropy_loss",      np.mean(entropy_losses))
        self.logger.record("train/approx_kl",         np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction",     np.mean(clip_fractions))
        self.logger.record("train/grad_norm",         np.mean(grad_norms))
        self.logger.record("train/explained_variance",explained_var)
        self.logger.record("train/n_updates",         self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate",     lr)
        self.logger.record("train/clip_range",        clip_range)