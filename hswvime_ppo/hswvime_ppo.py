from copy import deepcopy
from typing import Any, Optional, Tuple, TypeVar, Union

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
from models.wyner import WynerInterface, WynerLoss, WynerVAE
from models.slot_memory import SlotMemory

class RunningMeanStd:
    """Welford online estimator for mean / std of a scalar stream."""
    def __init__(self, warmup: int = 64):
        self.mean = 0.0
        self.var = 0.0
        self.count = 0
        self.warmup = warmup  # min samples before std is trusted
        self._buffer: list[np.ndarray] = []
        self._warmed_up = False

    def update(self, batch: np.ndarray):
        batch = batch.ravel()
        if not self._warmed_up:
            self._buffer.append(batch)
            total = sum(len(b) for b in self._buffer)
            if total >= self.warmup:
                all_data = np.concatenate(self._buffer)
                self.mean = float(all_data.mean())
                self.var = float(all_data.var())
                self.count = len(all_data)
                self._buffer.clear()
                self._warmed_up = True
            return
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> float:
        return max(float(np.sqrt(self.var)), 1e-6)


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
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        vae_recon_coef: float = 1.0,
        wyner_recon_coef: float = 1.0,
        vae_kl_coef: float = 0.1,
        wyner_kl_coef: float = 0.1,
        kl_use_schedule: bool = False,
        kl_anneal_steps: int = 50_000,
        intrinsic_scale: float = 1.0,
        intrinsic_anneal_steps: int = 0,
        intrinsic_normalize: bool = False,
        max_grad_norm: float = 500.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TransitionRolloutBuffer]] = TransitionRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        memory_shape: Tuple[int, ...] = (1, 64),
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        vae_features_extractor_class: VAEInterface  = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        wyner_features_extractor_class: WynerInterface = WynerVAE,
        wyner_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        aux_max_grad_norm: float = 500.0,
        mu_dim: int = 32,
        num_slots: int = 8,
        gate_mode: str = "detached",
        gate_scale: float = 1.0,
        gate_threshold: float = 0.0,
        wyner_kl_target: float = 0.0,
        wyner_kl_target_coef: float = 0.0,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["vae_features_extractor_class"] = vae_features_extractor_class
        policy_kwargs["vae_features_extractor_kwargs"] = vae_features_extractor_kwargs
        policy_kwargs["wyner_features_extractor_class"] = wyner_features_extractor_class
        policy_kwargs["wyner_features_extractor_kwargs"] = wyner_features_extractor_kwargs

        self.num_slots = num_slots
        self.gate_mode = gate_mode
        self.gate_scale = gate_scale
        self.gate_threshold = gate_threshold
        self.wyner_kl_target = wyner_kl_target
        self.wyner_kl_target_coef = wyner_kl_target_coef

        # Slots store μ (VAE latent, dim=mu_dim), not h_t (Wyner GRU state)
        self._slot_memory_shape = (num_slots, mu_dim)

        rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        rollout_buffer_kwargs["memory_shape"] = memory_shape
        rollout_buffer_kwargs["slot_memory_shape"] = self._slot_memory_shape

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

        self.vae_recon_coef = vae_recon_coef
        self.wyner_recon_coef = wyner_recon_coef
        self.vae_kl_coef = vae_kl_coef
        self.wyner_kl_coef = wyner_kl_coef
        self.kl_use_schedule = kl_use_schedule
        self.kl_anneal_steps = kl_anneal_steps

        self.intrinsic_scale = intrinsic_scale
        self.intrinsic_anneal_steps = intrinsic_anneal_steps
        self.intrinsic_normalize = intrinsic_normalize
        self._kl_running_stats = RunningMeanStd()
        self._gate_kl_running_stats = RunningMeanStd()  # Always-on stats for gate normalization

        self.memory_shape = memory_shape

        self.null_action = null_action

        self.aux_max_grad_norm = aux_max_grad_norm

        self._last_memory = None
        self._last_slots = None
        self._last_ages = None
        self._prev_last_obs = None
        self._prev_action = None
        self._timesteps = None

        if _init_setup_model:
            self._setup_model()

        self.slot_memory = SlotMemory(
            num_slots=self.num_slots,
            slot_dim=self._slot_memory_shape[1],
            gate_mode=self.gate_mode,
            gate_scale=self.gate_scale,
            gate_threshold=self.gate_threshold,
        ).to(self.device)
    
    def set_env(self, env, force_reset: bool = True):
        ret = super().set_env(env, force_reset)
        if force_reset:
            self._prev_last_obs = None
            self._last_memory = None
            self._last_slots = None
            self._last_ages = None
            self._prev_action = None
            self._timesteps = None
        return ret
    
    def _setup_learn(self, total_timesteps, callback = None, reset_num_timesteps = True, tb_log_name = "run", progress_bar = False):
        ret = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self._last_memory = th.zeros((self.n_envs, *self.memory_shape), device=self.device)
        self._last_slots, self._last_ages = self.slot_memory.init_state(self.n_envs, self.device)
        self._prev_last_obs = deepcopy(self._last_obs)
        self._prev_action = np.tile(self.null_action, (self.n_envs, 1))
        self._timesteps = np.zeros(self.n_envs, dtype=np.int64)

        return ret

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TransitionRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._prev_last_obs is not None, "No previous observation was provided"
        assert self._last_memory is not None, "No previous memory was provided"
        assert self._last_slots is not None, "No slot memory was provided"
        assert self._prev_action is not None, "No previous action was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        _wyner_kl_log: list[float] = []
        _gate_log: list[float] = []

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
                timestep_tensor = th.tensor(self._timesteps, device=self.device, dtype=th.long)
                actions, memory, values, log_probs = self.policy.forward(
                    s_tm1, a_tm1, s_t, self._last_memory, self._last_slots, timestep=timestep_tensor
                )
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Calculate intrinsic reward
            with th.no_grad():
                s_tp1 = obs_as_tensor(new_obs, self.device)
                a_t = obs_as_tensor(actions, self.device)
                vae_tp1 = self.policy.vae_feature_extractor.forward(s_t, a_t, s_tp1)
                wyner_loss: WynerLoss = self.policy.wyner_feature_extractor.loss(
                    self.policy.wyner_feature_extractor.forward(memory, vae_tp1.mu, None, vae_tp1.skips, timestep=timestep_tensor+1),
                )
                wyner_kl = wyner_loss.kl_loss.view(-1).cpu()  # (n_envs,)

                kl_np = wyner_kl.numpy()
                _wyner_kl_log.append(kl_np.mean().item())

                if self.intrinsic_normalize:
                    self._kl_running_stats.update(kl_np)
                    if self._kl_running_stats._warmed_up:
                        intrinsic_rewards = self.intrinsic_scale * kl_np / self._kl_running_stats.std
                    else:
                        intrinsic_rewards = np.zeros_like(kl_np)
                else:
                    if self.intrinsic_anneal_steps > 0:
                        decay = max(1.0 - self.num_timesteps / self.intrinsic_anneal_steps, 0.0)
                    else:
                        decay = 1.0
                    intrinsic_rewards = (self.intrinsic_scale * decay) * kl_np

            assert rewards.shape == intrinsic_rewards.shape == (self.n_envs,)

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
                    memory[idx, ...] = 0.0
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            s_t[idx:idx+1], a_t[idx:idx+1], s_tp1[idx:idx+1],
                            self._last_memory[idx:idx+1], self._last_slots[idx:idx+1],
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
                self._last_memory,
                memory,
                self._last_slots,
                self._prev_action,
                self._timesteps,
                intrinsic_rewards,
            )

            # Slot write: store π_t (policy projection) gated by Wyner KL surprise
            with th.no_grad():
                vae_t = self.policy.vae_feature_extractor.forward(s_tm1, a_tm1, s_t)
                wyner_out_t = self.policy.wyner_feature_extractor.forward(
                    self._last_memory, vae_t.mu, None, vae_t.skips, timestep=timestep_tensor,
                )
                current_kl = self.policy.wyner_feature_extractor.loss(wyner_out_t).kl_loss.view(-1)
                # Always z-score normalize KL for gating so the gate discriminates
                # between relatively surprising vs boring steps, even when mean KL is high.
                gate_kl_np = current_kl.cpu().numpy()
                self._gate_kl_running_stats.update(gate_kl_np)
                if self._gate_kl_running_stats._warmed_up:
                    gate_kl = (current_kl - self._gate_kl_running_stats.mean) / self._gate_kl_running_stats.std
                else:
                    gate_kl = current_kl
                # Write π_t (policy projection) — same space as attention query,
                # but decoupled from μ so PPO gradients don't corrupt the VAE.
                new_slots, new_ages, gate = self.slot_memory.write(
                    self._last_slots, self._last_ages, vae_t.pi.detach(), gate_kl,
                )
                _gate_log.append(gate.mean().item())

            # Increment timesteps, reset for done envs
            self._timesteps += 1
            for idx, done in enumerate(dones):
                if done:
                    self._timesteps[idx] = 0
                    memory[idx, ...] = 0.0
                    new_slots[idx] = 0.0
                    new_ages[idx] = th.arange(self.num_slots, device=self.device).float()

            self._prev_last_obs = self._last_obs
            self._prev_action = actions
            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_memory.copy_(memory)
            self._last_slots = new_slots
            self._last_ages = new_ages
            del memory

        with th.no_grad():
            values = self.policy.predict_values(s_t, a_t, s_tp1, self._last_memory, self._last_slots)
            values = values.view(-1).cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        self.logger.record("intrinsic/wyner_kl_mean", np.mean(_wyner_kl_log))
        if self.intrinsic_normalize:
            self.logger.record("intrinsic/kl_running_mean", self._kl_running_stats.mean)
            self.logger.record("intrinsic/kl_running_std", self._kl_running_stats.std)
        if self.intrinsic_anneal_steps > 0:
            self.logger.record("intrinsic/effective_scale", self.intrinsic_scale * max(1.0 - self.num_timesteps / self.intrinsic_anneal_steps, 0.0))
        self.logger.record("slots/gate_mean", np.mean(_gate_log))
        self.logger.record("slots/gate_kl_running_mean", self._gate_kl_running_stats.mean)
        self.logger.record("slots/gate_kl_running_std", self._gate_kl_running_stats.std)
        self.logger.record("debug/wyner_h_norm", float(th.norm(self._last_memory).item()))
        self.logger.record("debug/slots_norm", float(th.norm(self._last_slots).item()))

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        if self.kl_use_schedule:
            kl_progress = min(self.num_timesteps / max(self.kl_anneal_steps, 1), 1.0)
            effective_vae_kl_coef = self.vae_kl_coef * kl_progress
            effective_wyner_kl_coef = self.wyner_kl_coef * kl_progress
        else:
            effective_vae_kl_coef = self.vae_kl_coef
            effective_wyner_kl_coef = self.wyner_kl_coef

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        vae_losses, wyner_losses = [], []
        vae_recon_losses, vae_kl_losses = [], []
        wyner_recon_losses, wyner_kl_losses, wyner_recon_next_losses = [], [], []
        grad_norms = []
        all_approx_kl_divs = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy, _ = self.policy.evaluate_actions(
                    rollout_data.prev_observations,   # s_{t-1}
                    rollout_data.prev_actions,        # a_{t-1}
                    rollout_data.observations,        # s_t
                    rollout_data.memories,            # h_{t-1}
                    rollout_data.slot_memories,       # slot state at t
                    actions,
                    timestep=rollout_data.timesteps.long(),
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())


                vae_t = self.policy.vae_feature_extractor.forward(
                    rollout_data.prev_observations,
                    rollout_data.prev_actions,
                    rollout_data.observations,
                )

                vae_tp1 = self.policy.vae_feature_extractor.forward(
                    rollout_data.observations,
                    actions,
                    rollout_data.next_observations,
                )

                vae_loss_obj = self.policy.vae_feature_extractor.loss(vae_t)

                vae_loss = (
                    self.vae_recon_coef * vae_loss_obj.recon_loss
                    + effective_vae_kl_coef * vae_loss_obj.kl_loss
                )

                vae_losses.append(vae_loss.item())
                vae_recon_losses.append(vae_loss_obj.recon_loss.item())
                vae_kl_losses.append(vae_loss_obj.kl_loss.item())

                # Wyner forward: h_t already has μ_t baked in, so feed μ_{t+1}
                # to advance GRU to h_{t+1} — matching rollout KL semantics.
                if len(wyner_kl_losses) == 0:
                    self.logger.record("debug/train_h_input_shape", str(tuple(rollout_data.wyner_h.shape)))
                    self.logger.record("debug/train_h_input_norm", float(th.norm(rollout_data.wyner_h).item()))
                wyner_out = self.policy.wyner_feature_extractor.forward(
                    rollout_data.wyner_h,
                    vae_tp1.mu,
                    None,
                    vae_tp1.skips,
                    timestep=rollout_data.timesteps.long() + 1,
                )
                wyner_loss_obj = self.policy.wyner_feature_extractor.loss(
                    wyner_out,
                    recon_target=vae_tp1.recon_target,
                    recon_next_target=None,
                )

                kl_mean = wyner_loss_obj.kl_loss.mean()

                if self.wyner_kl_target_coef > 0:
                    # KL targeting: quadratic penalty pulling KL toward target.
                    # Can't collapse to 0, can't explode to 300.
                    kl_target_loss = (kl_mean - self.wyner_kl_target).pow(2)
                    wyner_loss = (
                        self.wyner_recon_coef * wyner_loss_obj.recon_loss.mean()
                        + self.wyner_kl_target_coef * kl_target_loss
                    )
                else:
                    # Detached-posterior KL: only train prior to match posterior.
                    post_mu_d = (wyner_out.posterior_mu if wyner_out.posterior_mu is not None
                                 else wyner_out.w.squeeze(1)).detach()
                    post_logvar_d = wyner_out.logvar.detach()
                    kl_for_prior = 0.5 * (
                        wyner_out.prior_logvar - post_logvar_d
                        + (post_logvar_d.exp() + (post_mu_d - wyner_out.prior_mu).pow(2))
                          / wyner_out.prior_logvar.exp()
                        - 1.0
                    ).sum(dim=-1).mean()
                    wyner_loss = (
                        self.wyner_recon_coef * wyner_loss_obj.recon_loss.mean()
                        + effective_wyner_kl_coef * kl_for_prior
                    )

                wyner_losses.append(wyner_loss.item())
                wyner_recon_losses.append(wyner_loss_obj.recon_loss.mean().item())
                wyner_kl_losses.append(kl_mean.detach().item())
                wyner_recon_next_losses.append(0.0)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + vae_loss + wyner_loss


                                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
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

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                ppo_params = (list(self.policy.features_extractor.parameters()) +
                              list(self.policy.mlp_extractor.parameters()) +
                              list(self.policy.value_net.parameters()) +
                              list(self.policy.action_net.parameters()))
                vae_params = list(self.policy.vae_feature_extractor.parameters())
                wyner_params = list(self.policy.wyner_feature_extractor.parameters())

                for name, params in [("ppo", ppo_params), ("vae", vae_params), ("wyner", wyner_params)]:
                    total = th.nn.utils.clip_grad_norm_(params, float("inf"))  # measure only
                    self.logger.record(f"debug/{name}_grad_norm", total.item())

                # Independent clipping: PPO and aux models clip against their own norms
                grad_norm = th.nn.utils.clip_grad_norm_(ppo_params, self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(vae_params + wyner_params, self.aux_max_grad_norm)
                grad_norms.append(grad_norm.item())

                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("rewards/intrinsic_reward_mean", self.rollout_buffer.intrinsic_rewards.mean())
        self.logger.record("rewards/intrinsic_reward_std", self.rollout_buffer.intrinsic_rewards.std())
        self.logger.record("rewards/extrinsic_reward_mean", self.rollout_buffer.rewards.mean())
        self.logger.record("rewards/extrinsic_reward_std", self.rollout_buffer.rewards.std())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("vae/vae_loss", np.mean(vae_losses))
        self.logger.record("vae/vae_recon_loss", np.mean(vae_recon_losses))
        self.logger.record("vae/vae_kl_loss", np.mean(vae_kl_losses))
        self.logger.record("wyner/wyner_loss", np.mean(wyner_losses))
        self.logger.record("wyner/wyner_recon_loss", np.mean(wyner_recon_losses))
        self.logger.record("wyner/wyner_kl_loss", np.mean(wyner_kl_losses))
        self.logger.record("wyner/wyner_recon_next_loss", np.mean(wyner_recon_next_losses))
        self.logger.record("train/approx_kl", np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/vae_kl_coef", effective_vae_kl_coef)
        self.logger.record("train/wyner_kl_coef", effective_wyner_kl_coef)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

