from copy import deepcopy
from typing import Any, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO

from .buffer import TransitionRolloutBuffer
from .policies import HSWVIMEActorCriticPolicy

from models.vae import VAEInterface, TransitionSCVAE
from models.utils import RunningMeanStd

from typing import Optional, Tuple, Union, Any

SelfHSWVimePPO = TypeVar("SelfHSWVimePPO", bound="HSWVimePPO")


class HSWVimePPO(PPO):
    policy: HSWVIMEActorCriticPolicy
    rollout_buffer: TransitionRolloutBuffer

    def __init__(
        self,
        policy: HSWVIMEActorCriticPolicy,
        env: Union[GymEnv, str],
        null_action: np.ndarray,
        learning_rate: Union[float, Schedule] = 3e-4,
        vae_lr: float = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        vae_recon_coef: float = 1.0,
        vae_kl_coef: float = 0.1,
        vae_fwd_coef: float = 1.0,
        kl_use_schedule: bool = False,
        kl_anneal_steps: int = 50_000,
        max_grad_norm: float = 0.5,
        aux_max_grad_norm: float = 5.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TransitionRolloutBuffer]] = TransitionRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        vae_features_extractor_class: VAEInterface = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_intrinsic: bool = True,
        gru_hidden_dim: int = 0,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["vae_features_extractor_class"] = vae_features_extractor_class
        policy_kwargs["vae_features_extractor_kwargs"] = vae_features_extractor_kwargs
        policy_kwargs["gru_hidden_dim"] = gru_hidden_dim

        rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        rollout_buffer_kwargs["gru_hidden_dim"] = gru_hidden_dim

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
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
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        self.vae_lr = vae_lr
        self.vae_recon_coef = vae_recon_coef
        self.vae_kl_coef = vae_kl_coef
        self.vae_fwd_coef = vae_fwd_coef
        self.kl_use_schedule = kl_use_schedule
        self.kl_anneal_steps = kl_anneal_steps
        self.aux_max_grad_norm = aux_max_grad_norm
        self.null_action = null_action
        self.normalize_intrinsic = normalize_intrinsic
        self.gru_hidden_dim = gru_hidden_dim

        self._prev_last_obs = None
        self._prev_action = None
        self._gru_hidden = None  # (1, n_envs, gru_hidden_dim)

        # Intrinsic reward running stats
        self.intrinsic_rms = RunningMeanStd() if normalize_intrinsic else None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Create separate VAE optimizer (two-optimizer setup)
        vae_params = list(self.policy.vae_feature_extractor.parameters())
        self.vae_optimizer = th.optim.Adam(vae_params, lr=self.vae_lr)

    def set_env(self, env, force_reset: bool = True):
        ret = super().set_env(env, force_reset)
        if force_reset:
            self._prev_last_obs = None
            self._prev_action = None
            self._gru_hidden = None
        return ret

    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True, tb_log_name="run", progress_bar=False):
        ret = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self._prev_last_obs = deepcopy(self._last_obs)
        self._prev_action = np.tile(self.null_action, (self.n_envs, 1))
        if self.gru_hidden_dim > 0:
            self._gru_hidden = th.zeros(1, self.n_envs, self.gru_hidden_dim, device=self.device)
        return ret

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TransitionRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect experiences using the current policy and fill a RolloutBuffer."""
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._prev_last_obs is not None, "No previous observation was provided"
        assert self._prev_action is not None, "No previous action was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                s_tm1 = obs_as_tensor(self._prev_last_obs, self.device)
                a_tm1 = obs_as_tensor(self._prev_action, self.device)
                s_t = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, h_mem = self.policy.forward(
                    s_tm1, a_tm1, s_t, h_prev=self._gru_hidden,
                )
            actions = actions.cpu().numpy()

            # Save GRU state before stepping
            gru_h_np = None
            if self.gru_hidden_dim > 0:
                gru_h_np = self._gru_hidden.squeeze(0).cpu().numpy()  # (n_envs, H)

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Intrinsic reward = forward prediction error
            with th.no_grad():
                a_t_tensor = th.as_tensor(actions, device=self.device).float()
                intrinsic_rewards = self.policy.vae_feature_extractor.intrinsic_reward(
                    s_t, a_t_tensor, obs_as_tensor(new_obs, self.device),
                ).cpu().numpy()

            # Normalize intrinsic rewards
            if self.intrinsic_rms is not None:
                self.intrinsic_rms.update(intrinsic_rewards)
                intrinsic_rewards = self.intrinsic_rms.normalize(intrinsic_rewards)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    with th.no_grad():
                        s_tp1 = obs_as_tensor(new_obs, self.device)
                        a_t = obs_as_tensor(actions, self.device)
                        terminal_value = self.policy.predict_values(
                            s_t[idx:idx+1], a_t[idx:idx+1], s_tp1[idx:idx+1],
                        ).item()
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                self._prev_last_obs,
                new_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                self._prev_action,
                intrinsic_rewards,
                gru_h_np,
            )

            # Update GRU hidden state (keep h_mem from policy forward)
            if self.gru_hidden_dim > 0:
                self._gru_hidden = h_mem
                # Reset GRU state for done envs
                for idx, done in enumerate(dones):
                    if done:
                        self._gru_hidden[:, idx, :] = 0.0

            self._prev_last_obs = self._last_obs
            self._prev_action = actions
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            s_tp1 = obs_as_tensor(new_obs, self.device)
            a_t = obs_as_tensor(actions, self.device)
            values = self.policy.predict_values(s_t, a_t, s_tp1)
            values = values.view(-1).cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        self.policy.set_training_mode(True)

        # Update PPO optimizer LR (VAE optimizer has fixed LR)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        if self.kl_use_schedule:
            kl_progress = min(self.num_timesteps / max(self.kl_anneal_steps, 1), 1.0)
            effective_vae_kl_coef = self.vae_kl_coef * kl_progress
        else:
            effective_vae_kl_coef = self.vae_kl_coef

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        vae_losses = []
        vae_recon_losses, vae_kl_losses, vae_fwd_losses = [], [], []
        vae_aux_losses = []
        ppo_grad_norms, vae_grad_norms = [], []
        all_approx_kl_divs = []
        rho_means, rho_stds = [], []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Reconstruct GRU hidden for single-step BPTT
                h_prev = None
                if self.gru_hidden_dim > 0:
                    # (B, H) -> (1, B, H), detached so no BPTT beyond one step
                    h_prev = rollout_data.gru_hidden_states.unsqueeze(0).detach()

                values, log_prob, entropy, _ = self.policy.evaluate_actions(
                    rollout_data.prev_observations,
                    rollout_data.prev_actions,
                    rollout_data.observations,
                    actions,
                    h_prev,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ── PPO loss + optimization ──
                ppo_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                ppo_loss.backward()

                ppo_params = (list(self.policy.mlp_extractor.parameters()) +
                              list(self.policy.value_net.parameters()) +
                              list(self.policy.action_net.parameters()))
                if self.policy.gru is not None:
                    ppo_params += list(self.policy.gru.parameters())

                ppo_gn = th.nn.utils.clip_grad_norm_(ppo_params, self.max_grad_norm)
                ppo_grad_norms.append(ppo_gn.item())
                self.policy.optimizer.step()

                # ── VAE loss + optimization (separate optimizer) ──
                vae_out = self.policy.vae_feature_extractor.forward(
                    rollout_data.prev_observations,
                    rollout_data.prev_actions,
                    rollout_data.observations,
                )
                vae_loss_obj = self.policy.vae_feature_extractor.loss(vae_out)
                vae_loss = (
                    self.vae_recon_coef * vae_loss_obj.recon_loss
                    + effective_vae_kl_coef * vae_loss_obj.kl_loss
                )
                if vae_loss_obj.aux_loss is not None:
                    vae_loss = vae_loss + vae_loss_obj.aux_loss
                    vae_aux_losses.append(vae_loss_obj.aux_loss.item())
                if vae_loss_obj.fwd_loss is not None:
                    vae_loss = vae_loss + self.vae_fwd_coef * vae_loss_obj.fwd_loss
                    vae_fwd_losses.append(vae_loss_obj.fwd_loss.item())

                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                vae_gn = th.nn.utils.clip_grad_norm_(
                    self.policy.vae_feature_extractor.parameters(), self.aux_max_grad_norm,
                )
                vae_grad_norms.append(vae_gn.item())
                self.vae_optimizer.step()

                vae_losses.append(vae_loss.item())
                vae_recon_losses.append(vae_loss_obj.recon_loss.item())
                vae_kl_losses.append(vae_loss_obj.kl_loss.item())

                # Track rho stats
                if hasattr(vae_out, 'rho') and vae_out.rho is not None:
                    with th.no_grad():
                        rho = vae_out.rho
                        # For spCauchy, rho is concentration; for Gaussian, rho slot holds logvar
                        if rho.shape[-1] == 1:  # spCauchy: rho is (B, 1)
                            rho_means.append(rho.mean().item())
                            rho_stds.append(rho.std().item())

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    all_approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # ── Logging ──
        # Rewards
        self.logger.record("rewards/intrinsic_mean", self.rollout_buffer.intrinsic_rewards.mean())
        self.logger.record("rewards/intrinsic_std", self.rollout_buffer.intrinsic_rewards.std())
        self.logger.record("rewards/extrinsic_mean", self.rollout_buffer.rewards.mean())
        self.logger.record("rewards/extrinsic_std", self.rollout_buffer.rewards.std())

        # PPO
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/ppo_grad_norm", np.mean(ppo_grad_norms))

        # VAE
        self.logger.record("vae/total_loss", np.mean(vae_losses))
        self.logger.record("vae/recon_loss", np.mean(vae_recon_losses))
        self.logger.record("vae/kl_loss", np.mean(vae_kl_losses))
        self.logger.record("vae/kl_coef", effective_vae_kl_coef)
        self.logger.record("vae/grad_norm", np.mean(vae_grad_norms))
        if vae_fwd_losses:
            self.logger.record("vae/fwd_loss", np.mean(vae_fwd_losses))
        if vae_aux_losses:
            self.logger.record("vae/uniformity_loss", np.mean(vae_aux_losses))
        if rho_means:
            self.logger.record("vae/rho_mean", np.mean(rho_means))
            self.logger.record("vae/rho_std", np.mean(rho_stds))

        # Meta
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
