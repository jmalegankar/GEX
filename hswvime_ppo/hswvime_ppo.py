from copy import deepcopy
from typing import Any, TypeVar, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
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
from models.wyner import WynerVAE, WynerConfig, WynerLoss, WynerContextVAE
from models.episodic_memory import EpisodicMemoryInterface, BatchedNoveltyMemory

SelfHSWVimePPO = TypeVar("SelfHSWVimePPO", bound="HSWVimePPO")


class HSWVimePPO(PPO):
    """
    PPO with HSWVIME intrinsic motivation.

    Two auxiliary models sit alongside the policy:
      - TransitionSCVAE  : encodes (s_{t-1}, a_{t-1}, s_t) → mu_t  (spherical Cauchy latent)
      - WynerVAE         : maintains slow-path context h_t (GRU) and learns z_t,
                           the Wyner common cause of mu_t and mu_{t+1}.

    Memory in the rollout buffer:
      Stores h_{t-1} (the WynerVAE slow-path hidden state BEFORE step t).
      memory_shape must match WynerConfig.context_dim.
      e.g.  memory_shape=(256,)  for context_dim=256.

    Intrinsic reward:
      r_int = intrinsic_scale * KL[q(z_t|h_t, mu_t, mu_{t+1}) || p(z_t|h_t)] * episodic_bonus
      KL is the per-sample Wyner information gain — high when the agent
      encounters transitions the world model cannot predict from context alone.
    """

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
        aux_max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[TransitionRolloutBuffer]] = TransitionRolloutBuffer,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        # memory_shape must equal (context_dim,) — stores context_t per step
        memory_shape: Tuple[int, ...] = (64,),
        mu_buffer_k: int = 64,
        context_dim: int = 64,
        vae_latent_dim: int = 32,
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
        wyner_features_extractor_class = WynerVAE,
        wyner_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        episodic_memory_class: type = BatchedNoveltyMemory,
        episodic_memory_kwargs: Optional[dict[str, Any]] = None,
        aux_lr: float = 3e-4,
        ema_decay: float = 0.995,
    ):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs["vae_features_extractor_class"]    = vae_features_extractor_class
        policy_kwargs["vae_features_extractor_kwargs"]   = vae_features_extractor_kwargs
        policy_kwargs["wyner_features_extractor_class"]  = wyner_features_extractor_class
        policy_kwargs["wyner_features_extractor_kwargs"] = wyner_features_extractor_kwargs

        rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        rollout_buffer_kwargs["memory_shape"] = memory_shape
        rollout_buffer_kwargs["vae_latent_dim"] = vae_latent_dim

        super().__init__(
            policy=policy, env=env, learning_rate=learning_rate,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range,
            clip_range_vf=clip_range_vf, normalize_advantage=normalize_advantage,
            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
            use_sde=use_sde, sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl, stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs,
            verbose=verbose, seed=seed, device=device,
            _init_setup_model=False,
        )

        self.vae_recon_coef   = vae_recon_coef
        self.wyner_recon_coef = wyner_recon_coef
        self.vae_kl_coef      = vae_kl_coef
        self.wyner_kl_coef    = wyner_kl_coef
        self.kl_use_schedule  = kl_use_schedule
        self.kl_anneal_steps  = kl_anneal_steps
        self.intrinsic_scale  = intrinsic_scale
        self.aux_max_grad_norm = aux_max_grad_norm
        self.memory_shape     = memory_shape
        self.mu_buffer_k      = mu_buffer_k
        self.context_dim      = context_dim
        self.vae_latent_dim   = vae_latent_dim
        self.null_action      = null_action

        self.aux_lr    = aux_lr
        self.ema_decay = ema_decay

        self.episodic_memory_class  = episodic_memory_class
        self.episodic_memory_kwargs = episodic_memory_kwargs or {}
        self._episodic_memory: Optional[EpisodicMemoryInterface] = None
        self._target_vae: Optional[nn.Module] = None

        # Persistent across rollouts; shape (n_envs, context_dim)
        self._last_memory: Optional[th.Tensor] = None
        self._mu_buffer: Optional[th.Tensor] = None
        self._episode_start_mu: Optional[th.Tensor] = None
        self._prev_last_obs = None
        self._prev_action   = None

        self.aux_optimizer: Optional[th.optim.Optimizer] = None

        if _init_setup_model:
            self._setup_model()

    # ── Setup / env reset ────────────────────────────────────────────────────

    def _setup_model(self) -> None:
        super()._setup_model()

        # Split optimizers: PPO params get one Adam, VAE+Wyner get another.
        # SB3 creates self.policy.optimizer over ALL policy parameters.
        # We rebuild it to exclude aux params, then create a separate aux optimizer.
        aux_param_ids = set(
            id(p) for p in self.policy.vae_feature_extractor.parameters()
        ) | set(
            id(p) for p in self.policy.wyner_feature_extractor.parameters()
        )

        ppo_params = [p for p in self.policy.parameters() if id(p) not in aux_param_ids]
        aux_params = [p for p in self.policy.parameters() if id(p) in aux_param_ids]

        lr = self.lr_schedule(1)
        self.policy.optimizer = self.policy.optimizer_class(
            ppo_params, lr=lr, **self.policy.optimizer_kwargs
        )
        # Fixed LR for aux optimizer — not tied to PPO schedule
        self.aux_optimizer = th.optim.Adam(aux_params, lr=self.aux_lr)

        # Target SCVAE: EMA copy that produces stable mu targets for Wyner reconstruction.
        # Without this, the SCVAE mu vectors shift every epoch, making Wyner's recon target
        # non-stationary and preventing the decoder from learning.
        self._target_vae = deepcopy(self.policy.vae_feature_extractor)
        self._target_vae.requires_grad_(False)
        self._target_vae.eval()

    @th.no_grad()
    def _update_target_vae(self):
        """Polyak-average the live SCVAE into the target SCVAE."""
        for p_live, p_tgt in zip(
            self.policy.vae_feature_extractor.parameters(),
            self._target_vae.parameters(),
        ):
            p_tgt.data.mul_(self.ema_decay).add_(p_live.data, alpha=1 - self.ema_decay)

    def set_env(self, env, force_reset: bool = True):
        ret = super().set_env(env, force_reset)
        if force_reset:
            self._prev_last_obs    = None
            self._last_memory      = None
            self._prev_action      = None
            self._mu_buffer        = None
            self._episode_start_mu = None
        return ret

    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True,
                     tb_log_name="run", progress_bar=False):
        ret = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        # context_t: (n_envs, context_dim) — starts zeroed each learn() call
        self._last_memory   = th.zeros((self.n_envs, *self.memory_shape), device=self.device)
        self._mu_buffer     = th.zeros(
            (self.n_envs, self.mu_buffer_k, self.vae_latent_dim), device=self.device
        )
        self._episode_start_mu = th.zeros(
            (self.n_envs, self.vae_latent_dim), device=self.device
        )
        self._prev_last_obs = deepcopy(self._last_obs)
        self._prev_action   = np.tile(self.null_action, (self.n_envs, 1))

        self._episodic_memory = self.episodic_memory_class(
            n_envs=self.n_envs,
            **self.episodic_memory_kwargs,
        )
        return ret

    # ── Rollout collection ───────────────────────────────────────────────────

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TransitionRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs     is not None, "No previous observation"
        assert self._prev_last_obs is not None, "No previous observation (prev)"
        assert self._last_memory  is not None, "No previous memory"
        assert self._prev_action  is not None, "No previous action"

        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        _wyner_kl_log:       list[float] = []
        _episodic_novel_log: list[float] = []

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                s_tm1 = obs_as_tensor(self._prev_last_obs, self.device)
                a_tm1 = obs_as_tensor(self._prev_action,   self.device)
                s_t   = obs_as_tensor(self._last_obs,      self.device)

                # Policy forward: uses precomputed context_t as memory.
                # Second return is the context passed through unchanged.
                actions, _, values, log_probs = self.policy.forward(
                    s_tm1, a_tm1, s_t, self._last_memory
                )

            actions_np = actions.cpu().numpy()
            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # ── Intrinsic reward ─────────────────────────────────────────────
            #
            # Now that we have s_{t+1}, we can compute mu_{t+1} and run
            # forward_train to get the true KL(q||p) as intrinsic reward.
            #
            # h_prev = self._last_memory  (= h_{t-1}, before this step)
            # This is consistent with how forward_train is called in train():
            # both use the stored h_{t-1} from the buffer.
            with th.no_grad():
                s_tp1 = obs_as_tensor(new_obs, self.device)
                a_t   = obs_as_tensor(actions_np, self.device)

                # mu_t: SCVAE latent for (s_{t-1} → s_t) transition
                mu_t, _, _ = self.policy.vae_feature_extractor.encode(s_tm1, a_tm1, s_t)

                # mu_{t+1}: SCVAE latent for (s_t → s_{t+1}) transition
                vae_tp1 = self.policy.vae_feature_extractor.forward(s_t, a_t, s_tp1)

                # 1. Compute intrinsic reward BEFORE inserting mu_{t+1}
                #    mu_buffer contains [mu_{t-K}, ..., mu_{t-1}] (causal)
                wyner_out = self.policy.wyner_feature_extractor.forward(
                    vae_tp1.mu, self._mu_buffer
                )
                wyner_l = self.policy.wyner_feature_extractor.loss(wyner_out)

                # 2. Compute context for policy features (used next iteration)
                context_t = wyner_out.h_slow  # (n_envs, context_dim)

                # 3. Roll buffer and insert AFTER forward (no self-attention leak)
                self._mu_buffer = th.roll(self._mu_buffer, -1, dims=1)
                self._mu_buffer[:, -1, :] = vae_tp1.mu.detach()

                # Per-env KL scalar; intrinsic_reward is already detached and ≥ 0
                wyner_kl      = wyner_l.intrinsic_reward.cpu()    # (n_envs,)
                episodic_bonus = self._episodic_memory.query_and_add(mu_t)  # (n_envs,)

                intrinsic_rewards = self.intrinsic_scale * wyner_kl.numpy()

                _wyner_kl_log.append(wyner_kl.mean().item())
                _episodic_novel_log.append(float(episodic_bonus.mean()))

            assert rewards.shape == intrinsic_rewards.shape == (self.n_envs,), \
                f"Reward shape mismatch: {rewards.shape} vs {intrinsic_rewards.shape}"

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # ── Episode-end handling ─────────────────────────────────────────
            #
            # Two cases for done envs:
            #   1. TimeLimit truncation: bootstrap terminal value, then zero context.
            #   2. Regular termination: just zero context + mu_buffer.
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None \
                        and infos[idx].get("TimeLimit.truncated", False):
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(
                            s_t[idx:idx+1], a_t[idx:idx+1],
                            s_tp1[idx:idx+1], self._last_memory[idx:idx+1]
                        ).item()
                    rewards[idx] += self.gamma * terminal_value

            # 4. On episode done: zero that env's mu_buffer, recompute episode_start_mu
            for idx, done in enumerate(dones):
                if done:
                    self._mu_buffer[idx].zero_()
                    # Encode (s_{t+1}, no-op, s_{t+1}) as the new episode's start anchor
                    with th.no_grad():
                        s_start = obs_as_tensor(new_obs[idx:idx+1], self.device)
                        a_noop  = th.zeros(1, *self.null_action.shape, device=self.device)
                        start_mu, _, _ = self.policy.vae_feature_extractor.encode(
                            s_start, a_noop, s_start
                        )
                    self._episode_start_mu[idx] = start_mu.squeeze(0).detach()

            # Zero context for ALL done envs (truncated or not) before storing
            if dones.any():
                done_mask = th.from_numpy(dones).bool()
                context_t[done_mask] = 0.0

            # ── Store transition ─────────────────────────────────────────────
            # self._last_memory stores context_t (precomputed cross-attention output)
            rollout_buffer.add(
                self._last_obs,         # s_t
                self._prev_last_obs,    # s_{t-1}
                new_obs,                # s_{t+1}
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                self._last_memory,      # context_{t-1}: stored as 'memories' in buffer
                self._prev_action,
                intrinsic_rewards,
                self._episode_start_mu.cpu().numpy(),  # episode_start_mu per env
            )

            self._prev_last_obs = self._last_obs
            self._prev_action   = actions_np
            self._last_obs      = new_obs
            self._last_episode_starts = dones
            self._last_memory.copy_(context_t)  # advance: store context_t for next step

            if dones.any():
                self._episodic_memory.reset_envs(th.from_numpy(dones))

        # Final value estimate for GAE
        with th.no_grad():
            values = self.policy.predict_values(s_t, a_t, s_tp1, self._last_memory)
            values = values.view(-1).cpu().numpy()

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        self.logger.record("intrinsic/wyner_kl_mean",       np.mean(_wyner_kl_log))
        self.logger.record("intrinsic/episodic_novel_frac", np.mean(_episodic_novel_log))

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ── Training update ───────────────────────────────────────────────────────

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        # aux_optimizer uses fixed LR — no schedule update
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # KL annealing schedule (both VAE and Wyner KL)
        if self.kl_use_schedule:
            kl_progress = min(self.num_timesteps / max(self.kl_anneal_steps, 1), 1.0)
            effective_vae_kl_coef   = self.vae_kl_coef   * kl_progress
            effective_wyner_kl_coef = self.wyner_kl_coef * kl_progress
        else:
            effective_vae_kl_coef   = self.vae_kl_coef
            effective_wyner_kl_coef = self.wyner_kl_coef

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        vae_losses, vae_recon_losses, vae_kl_losses = [], [], []
        wyner_losses         = []
        wyner_kl_losses      = []
        wyner_recon_past_losses, wyner_recon_future_losses = [], []
        wyner_recon_prior_losses = []
        grad_norms           = []
        all_approx_kl_divs   = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # ── PPO losses ───────────────────────────────────────────────
                values, log_prob, entropy, _ = self.policy.evaluate_actions(
                    rollout_data.prev_observations,   # s_{t-1}
                    rollout_data.prev_actions,        # a_{t-1}
                    rollout_data.observations,        # s_t
                    rollout_data.memories,            # h_{t-1}  (stored in buffer)
                    actions,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio          = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1  = advantages * ratio
                policy_loss_2  = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss    = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                # ── SCVAE losses ─────────────────────────────────────────────
                # vae_t  : encodes (s_{t-1}, a_{t-1}, s_t)  → mu_t
                # vae_tp1: encodes (s_t,     a_t,     s_{t+1}) → mu_{t+1}
                # Both are needed for the Wyner posterior.
                vae_t   = self.policy.vae_feature_extractor.forward(
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

                # Target SCVAE: stable mu targets for Wyner reconstruction.
                # These don't shift every batch, so the Wyner decoder can learn.
                with th.no_grad():
                    tgt_vae_t = self._target_vae.forward(
                        rollout_data.prev_observations,
                        rollout_data.prev_actions,
                        rollout_data.observations,
                    )
                    tgt_vae_tp1 = self._target_vae.forward(
                        rollout_data.observations,
                        actions,
                        rollout_data.next_observations,
                    )
                vae_loss = (
                    self.vae_recon_coef * vae_loss_obj.recon_loss
                    + effective_vae_kl_coef * vae_loss_obj.kl_loss
                )
                if vae_loss_obj.aux_loss is not None:
                    vae_loss = vae_loss + vae_loss_obj.aux_loss

                vae_losses.append(vae_loss.item())
                vae_recon_losses.append(vae_loss_obj.recon_loss.item())
                vae_kl_losses.append(vae_loss_obj.kl_loss.item())

                # ── WynerContextVAE losses ────────────────────────────────────
                #
                # Build a K=1 context buffer from episode_start_mu stored in buffer.
                # This gives the prior a real anchor (first observation of each episode)
                # without storing the full K-step buffer per transition.
                episode_start_mu = rollout_data.episode_start_mus.unsqueeze(1)  # (B, 1, mu_dim)

                wyner_out = self.policy.wyner_feature_extractor.forward(
                    vae_t.mu,               # mu_t
                    episode_start_mu,       # K=1 buffer — real signal, cheap to store
                    mu_next=vae_tp1.mu,     # mu_{t+1}
                )
                # Use target SCVAE mu's as stable reconstruction targets
                wyner_l = self.policy.wyner_feature_extractor.loss(
                    wyner_out,
                    recon_target=tgt_vae_t.mu.detach(),
                    recon_next_target=tgt_vae_tp1.mu.detach(),
                )

                # wyner_recon_coef scales both past and future reconstruction.
                # lambda_past / lambda_future (in WynerConfig) control their
                # relative weight internally; wyner_recon_coef is the global
                # scale relative to the PPO loss.
                wyner_loss = (
                    self.wyner_recon_coef * (
                        wyner_l.recon_past_loss + wyner_l.recon_future_loss
                    )
                    + effective_wyner_kl_coef * wyner_l.kl_loss
                )

                wyner_losses.append(wyner_loss.item())
                wyner_kl_losses.append(wyner_l.kl_loss.item())
                wyner_recon_past_losses.append(wyner_l.recon_past_loss.item())
                wyner_recon_future_losses.append(wyner_l.recon_future_loss.item())
                if wyner_l.recon_prior_loss is not None:
                    wyner_recon_prior_losses.append(wyner_l.recon_prior_loss.item())

                # ── Combined loss and update ─────────────────────────────────
                ppo_loss = (
                    policy_loss
                    + self.ent_coef   * entropy_loss
                    + self.vf_coef    * value_loss
                )
                aux_loss = vae_loss + wyner_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    all_approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch}: approx_kl={approx_kl_div:.2f}")
                    break

                # Parameter groups for gradient norm logging
                ppo_params   = (list(self.policy.mlp_extractor.parameters()) +
                                list(self.policy.value_net.parameters()) +
                                list(self.policy.action_net.parameters()))
                aux_params   = (list(self.policy.vae_feature_extractor.parameters()) +
                                list(self.policy.wyner_feature_extractor.parameters()))

                # PPO backward + step
                self.policy.optimizer.zero_grad()
                ppo_loss.backward(retain_graph=True)
                grad_norm = th.nn.utils.clip_grad_norm_(ppo_params, self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.policy.optimizer.step()

                # Aux backward + step (separate Adam state)
                self.aux_optimizer.zero_grad()
                aux_loss.backward()
                th.nn.utils.clip_grad_norm_(aux_params, self.aux_max_grad_norm)
                self.aux_optimizer.step()

                # Polyak-update target SCVAE for stable Wyner recon targets
                self._update_target_vae()

                # Log gradient norms (diagnostic)
                with th.no_grad():
                    for name, params in [("ppo", ppo_params), ("vae", list(self.policy.vae_feature_extractor.parameters())), ("wyner", list(self.policy.wyner_feature_extractor.parameters()))]:
                        total = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
                        self.logger.record(f"debug/{name}_grad_norm", total)

                loss = ppo_loss + aux_loss  # for logging only

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # ── Logging ─────────────────────────────────────────────────────────
        self.logger.record("rewards/intrinsic_mean", self.rollout_buffer.intrinsic_rewards.mean())
        self.logger.record("rewards/intrinsic_std",  self.rollout_buffer.intrinsic_rewards.std())
        self.logger.record("rewards/extrinsic_mean", self.rollout_buffer.rewards.mean())
        self.logger.record("rewards/extrinsic_std",  self.rollout_buffer.rewards.std())

        self.logger.record("train/entropy_loss",        np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss",           np.mean(value_losses))
        self.logger.record("train/approx_kl",            np.mean(all_approx_kl_divs))
        self.logger.record("train/clip_fraction",        np.mean(clip_fractions))
        self.logger.record("train/loss",                 loss.item())
        self.logger.record("train/explained_variance",   explained_var)
        self.logger.record("train/grad_norm",            np.mean(grad_norms))

        self.logger.record("vae/loss",       np.mean(vae_losses))
        self.logger.record("vae/recon_loss", np.mean(vae_recon_losses))
        self.logger.record("vae/kl_loss",    np.mean(vae_kl_losses))

        self.logger.record("wyner/loss",              np.mean(wyner_losses))
        self.logger.record("wyner/kl_loss",           np.mean(wyner_kl_losses))
        self.logger.record("wyner/recon_past_loss",   np.mean(wyner_recon_past_losses))
        self.logger.record("wyner/recon_future_loss", np.mean(wyner_recon_future_losses))
        if wyner_recon_prior_losses:
            self.logger.record("wyner/recon_prior_loss", np.mean(wyner_recon_prior_losses))

        self.logger.record("train/vae_kl_coef",   effective_vae_kl_coef)
        self.logger.record("train/wyner_kl_coef", effective_wyner_kl_coef)
        self.logger.record("train/n_updates",     self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range",    clip_range)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)