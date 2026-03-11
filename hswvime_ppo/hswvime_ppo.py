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
from models.wyner import WynerInterface, WynerLoss, WynerVAE
from models.episodic_memory import EpisodicMemoryInterface, BatchedNoveltyMemory


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
        wyner_recon_coef: float = 1.0,
        vae_kl_coef: float = 0.1,
        wyner_kl_coef: float = 0.1,
        kl_use_schedule: bool = False,
        kl_anneal_steps: int = 50_000,
        intrinsic_scale: float = 1.0,
        max_grad_norm: float = 0.5,
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
        episodic_memory_class: type = BatchedNoveltyMemory,
        episodic_memory_kwargs: Optional[dict[str, Any]] = None,
        aux_max_grad_norm: float = 5.0,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["vae_features_extractor_class"] = vae_features_extractor_class
        policy_kwargs["vae_features_extractor_kwargs"] = vae_features_extractor_kwargs
        policy_kwargs["wyner_features_extractor_class"] = wyner_features_extractor_class
        policy_kwargs["wyner_features_extractor_kwargs"] = wyner_features_extractor_kwargs

        rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        rollout_buffer_kwargs["memory_shape"] = memory_shape

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

        self.memory_shape = memory_shape

        self.null_action = null_action

        self.episodic_memory_class = episodic_memory_class
        self.episodic_memory_kwargs = episodic_memory_kwargs or {}
        self._episodic_memory: Optional[EpisodicMemoryInterface] = None

        self.aux_max_grad_norm = aux_max_grad_norm
        # TODO: add aux_learning_rate param and forward via policy_kwargs when ready

        self._last_memory = None
        self._prev_last_obs = None
        self._prev_action = None

        if _init_setup_model:
            self._setup_model()
    
    def set_env(self, env, force_reset: bool = True):
        ret = super().set_env(env, force_reset)
        if force_reset:
            self._prev_last_obs = None
        if force_reset:
            self._last_memory = None
        if force_reset:
            self._prev_action = None
        return ret
    
    def _setup_learn(self, total_timesteps, callback = None, reset_num_timesteps = True, tb_log_name = "run", progress_bar = False):
        ret = super()._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar)
        self._last_memory = th.zeros((self.n_envs, *self.memory_shape), device=self.device)
        self._prev_last_obs = deepcopy(self._last_obs)
        self._prev_action = np.tile(self.null_action, (self.n_envs, 1))

        self._episodic_memory = self.episodic_memory_class(
            n_envs=self.n_envs,
            **self.episodic_memory_kwargs,
        )

        return ret

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TransitionRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        assert self._prev_last_obs is not None, "No previous observation was provided"
        assert self._last_memory is not None, "No previous memory was provided"
        assert self._prev_action is not None, "No previous action was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        _wyner_kl_log: list[float] = []
        _episodic_novel_log: list[float] = []

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                s_tm1 = obs_as_tensor(self._prev_last_obs, self.device)  # type: ignore[arg-type]
                a_tm1 = obs_as_tensor(self._prev_action, self.device)  # type: ignore[arg-type]
                s_t = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions, memory, values, log_probs = self.policy.forward(s_tm1, a_tm1, s_t, self._last_memory)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Calculate intrinsic reward
            with th.no_grad():
                s_tp1 = obs_as_tensor(new_obs, self.device)
                a_t = obs_as_tensor(actions, self.device)
                vae_tp1 = self.policy.vae_feature_extractor.forward(s_t, a_t, s_tp1)
                wyner_loss: WynerLoss = self.policy.wyner_feature_extractor.loss(
                    self.policy.wyner_feature_extractor.forward(memory, vae_tp1.mu, None, vae_tp1.skips),
                )
                wyner_kl = wyner_loss.kl_loss.view(-1).cpu()  # (n_envs,)

                # Episodic bonus: 1.0 if hash bucket is novel this episode, else 0.0
                episodic_bonus = self._episodic_memory.query_and_add(vae_tp1.mu)  # (n_envs,)

                intrinsic_rewards = self.intrinsic_scale * (wyner_kl * episodic_bonus).numpy()
                _wyner_kl_log.append(wyner_kl.mean().item())
                _episodic_novel_log.append(episodic_bonus.mean().item())

            assert rewards.shape == intrinsic_rewards.shape == (self.n_envs,), f"Reward shape mismatch: {rewards.shape} vs {intrinsic_rewards.shape}"


            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    # Reset the memory for the next episode
                    memory[idx, ...] = 0.0
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            s_t[idx:idx+1], a_t[idx:idx+1], s_tp1[idx:idx+1], self._last_memory[idx:idx+1]
                        ).item()
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                self._prev_last_obs,  # type: ignore[arg-type]
                new_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                self._last_memory,  # type: ignore[call-overload]
                self._prev_action,  # type: ignore[arg-type]
                intrinsic_rewards,
            )
            self._prev_last_obs = self._last_obs  # type: ignore[assignment]
            self._prev_action = actions
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._last_memory.copy_(memory)  # type: ignore[call-overload]
            del memory

            # Reset episodic memory for finished episodes
            if dones.any():
                self._episodic_memory.reset_envs(th.from_numpy(dones))

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(s_t, a_t, s_tp1, self._last_memory)  # type: ignore[arg-type]
            values = values.view(-1).cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        self.logger.record("intrinsic/wyner_kl_mean", np.mean(_wyner_kl_log))
        self.logger.record("intrinsic/episodic_novel_frac", np.mean(_episodic_novel_log))

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
                    rollout_data.memories,            # memory at t
                    actions,
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

                wyner_out = self.policy.wyner_feature_extractor.forward(
                    rollout_data.memories,
                    vae_t.mu,
                    vae_tp1.mu,
                    vae_t.skips,
                )
                wyner_loss_obj = self.policy.wyner_feature_extractor.loss(
                    wyner_out,
                    recon_target=vae_t.recon_target,
                    recon_next_target=vae_tp1.recon_target,
                )

                wyner_loss = (
                    self.wyner_recon_coef * (wyner_loss_obj.recon_loss.mean() + wyner_loss_obj.recon_next_loss.mean())
                    + effective_wyner_kl_coef * wyner_loss_obj.kl_loss.mean()
                )

                wyner_losses.append(wyner_loss.item())
                wyner_recon_losses.append(wyner_loss_obj.recon_loss.mean().item())
                wyner_kl_losses.append(wyner_loss_obj.kl_loss.mean().item())
                wyner_recon_next_losses.append(wyner_loss_obj.recon_next_loss.mean().item())

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

                ppo_params = (list(self.policy.mlp_extractor.parameters()) +
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
