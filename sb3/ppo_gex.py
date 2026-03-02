from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from sc_vae_wrapper import SCVAEEncoderWrapper
from gex_rollout_buffer import GEXRolloutBuffer
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus


class PPOGEX(PPO):
    """
    PPO + GEX intrinsic reward + online SC-VAE with EMA target network.

    Two SC-VAE copies:
      sc_vae          — online encoder, trained every PPO iteration
      sc_vae_target   — EMA copy, used exclusively for μ in GEX bonuses

    EMA update after each SC-VAE gradient step:
      θ_target ← (1 - τ) · θ_target + τ · θ_online

    τ ≈ 0.005 gives ~200-step lag, keeping episodic memory and SimHash
    buckets consistent while still allowing the encoder to improve.
    """

    def __init__(
        self,
        *args,
        sc_vae=None,
        gex_modules=None,
        rms=None,
        eta: float = 1.0,
        tau: float = 0.005,     # EMA momentum.  0 = frozen target, 1 = no lag
        **kwargs,
    ):
        # Set before super().__init__() — SB3 calls _setup_model() inside it.
        self.sc_vae        = sc_vae
        self.gex_modules   = gex_modules
        self.rms           = rms
        self.eta           = float(eta)
        self.tau           = float(tau)

        # Target network: deep copy of online, frozen, EMA-updated.
        if sc_vae is not None:
            self.sc_vae_target = copy.deepcopy(sc_vae)
            for p in self.sc_vae_target.parameters():
                p.requires_grad_(False)
        else:
            self.sc_vae_target = None

        self._sc_vae_wrap: Optional[SCVAEEncoderWrapper] = None
        self.sc_vae_optimizer = None

        super().__init__(*args, **kwargs)

        if self.sc_vae is not None:
            self.sc_vae_optimizer = th.optim.Adam(
                self.sc_vae.parameters(),
                lr=self.sc_vae.cfg.lr,
            )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()

        from gymnasium import spaces
        assert isinstance(self.action_space, spaces.Discrete), \
            "PPOGEX currently supports Discrete action spaces only."

        self.rollout_buffer = GEXRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # GEX always encodes via the TARGET network, not the online one.
        if self.sc_vae_target is not None:
            self.sc_vae_target.to(self.device)
            self.sc_vae_target.eval()
            self._sc_vae_wrap = SCVAEEncoderWrapper(self.sc_vae_target, self.device)

        if self.sc_vae is not None:
            self.sc_vae.to(self.device)

        if self.gex_modules is not None:
            assert len(self.gex_modules) == self.n_envs, (
                f"Need one GEX module per env. "
                f"Got {len(self.gex_modules)} but n_envs={self.n_envs}"
            )

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(self) -> None:
        """Polyak-average online encoder into target encoder."""
        if self.sc_vae is None or self.sc_vae_target is None:
            return
        with th.no_grad():
            for p_online, p_target in zip(
                self.sc_vae.parameters(),
                self.sc_vae_target.parameters(),
            ):
                p_target.mul_(1.0 - self.tau).add_(self.tau * p_online)

    # ------------------------------------------------------------------
    # SC-VAE online update
    # ------------------------------------------------------------------

    def _update_sc_vae(self) -> None:
        """
        One pass over the rollout buffer to update the ONLINE encoder.
        Shape-agnostic: uses next_observations (never mutated by SB3)
        as the source of truth for (n_steps, n_envs, *obs_shape).
        """
        if self.sc_vae is None:
            return

        self.sc_vae.train()

        # next_observations is our custom field — always in original shape.
        next_obs  = self.rollout_buffer.next_observations          # (T, E, *obs)
        n_steps, n_envs = next_obs.shape[:2]
        n_total   = n_steps * n_envs
        obs_shape = next_obs.shape[2:]

        next_obs_flat = next_obs.reshape(n_total, *obs_shape)
        obs_flat      = self.rollout_buffer.observations.reshape(n_total, *obs_shape)
        actions_flat  = self.rollout_buffer.actions.reshape(n_total, -1).squeeze(-1)

        indices    = np.random.permutation(n_total)
        batch_size = 256

        for start in range(0, n_total, batch_size):
            idx = indices[start : start + batch_size]

            s_t = th.as_tensor(obs_flat[idx],      device=self.device)
            s_n = th.as_tensor(next_obs_flat[idx], device=self.device)
            a_t = th.as_tensor(actions_flat[idx],  device=self.device)

            out           = self.sc_vae(s_t, a_t, s_n)
            l_recon, l_kl = self.sc_vae.loss(out)
            loss          = l_recon + self.sc_vae.cfg.beta * l_kl

            self.sc_vae_optimizer.zero_grad()
            loss.backward()
            self.sc_vae_optimizer.step()

            # EMA update after every gradient step, not just once per epoch.
            # Finer-grained tracking = smoother target drift.
            self._update_ema()

        self.sc_vae.eval()

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: GEXRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None

        if self._sc_vae_wrap is None:
            return super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )

        self._rollout_r_int              = []
        self._rollout_r_ext              = []
        self._rollout_r_total            = []
        self._rollout_epi_size           = []
        self._rollout_lifetime_n_buckets = []

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        n_steps = 0
        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()
            new_obs, rewards_ext, dones, infos = env.step(actions_np)
            self._update_info_buffer(infos, dones)

            # Recover true terminal observations before SB3 overwrites them.
            real_next_obs = new_obs.copy()
            for i in range(env.num_envs):
                if dones[i] and "terminal_observation" in infos[i]:
                    real_next_obs[i] = infos[i]["terminal_observation"]

            # Encode via TARGET network — stable μ distribution.
            mu_batch = self._sc_vae_wrap.encode_mu(
                self._last_obs, actions_np, real_next_obs
            )
            mu_cpu = mu_batch.detach().cpu()

            r_int     = np.zeros_like(rewards_ext, dtype=np.float32)
            gex_infos = [None] * env.num_envs

            if self.gex_modules is not None:
                for i in range(env.num_envs):
                    # Score transition against this episode's memory first ...
                    ri, info     = self.gex_modules[i].step(mu_cpu[i])
                    r_int[i]     = float(ri)
                    gex_infos[i] = info

                    # ... then reset memory at episode boundary.
                    if dones[i]:
                        self.gex_modules[i].reset()

                        episode0_obs = new_obs[i]
                        no_op = self.sc_vae.cfg.no_op_action
                        mu0  = self._sc_vae_wrap.encode_mu(
                            episode0_obs[None],
                            np.array([no_op]),
                            episode0_obs[None],
                        )[0].detach().cpu()

                        self.gex_modules[i].episodic.query_and_add(mu0)
            
            # Truncation bootstrap — must happen before rollout_buffer.add()
            # so rewards_total gets the corrected value
            for i in range(env.num_envs):
                if (
                    dones[i]
                    and infos[i].get("terminal_observation") is not None
                    and infos[i].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = th.as_tensor(
                        infos[i]["terminal_observation"], device=self.device
                    ).unsqueeze(0)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards_ext[i] += self.gamma * terminal_value.item()

            r_int_norm = r_int.copy()
            if self.rms is not None:
                r_int_tensor = th.as_tensor(r_int, dtype=th.float32)
                self.rms.update(r_int_tensor)
                r_int_norm = self.rms.normalize(r_int_tensor).numpy()

            rewards_total = rewards_ext + self.eta * r_int_norm

            for i in range(env.num_envs):
                self._rollout_r_int.append(float(r_int[i]))
                self._rollout_r_ext.append(float(rewards_ext[i]))
                self._rollout_r_total.append(float(rewards_total[i]))

                if gex_infos[i] is not None:
                    if "episodic_size" in gex_infos[i]:
                        self._rollout_epi_size.append(
                            float(gex_infos[i]["episodic_size"])
                        )
                    if "lifetime_n_buckets" in gex_infos[i]:
                        self._rollout_lifetime_n_buckets.append(
                            float(gex_infos[i]["lifetime_n_buckets"])
                        )

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards_total,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=real_next_obs,
                intrinsic_reward=r_int,
                extrinsic_reward=rewards_ext,
            )

            self._last_obs            = new_obs
            self._last_episode_starts = dones

            n_steps += 1
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=self._last_episode_starts
        )
        callback.on_rollout_end()

        if self._rollout_r_int:
            self.logger.record("gex/r_int_mean",   np.mean(self._rollout_r_int))
            self.logger.record("gex/r_ext_mean",   np.mean(self._rollout_r_ext))
            self.logger.record("gex/r_total_mean", np.mean(self._rollout_r_total))
        if self._rollout_epi_size:
            self.logger.record(
                "gex/episodic_size_mean", np.mean(self._rollout_epi_size)
            )
        if self._rollout_lifetime_n_buckets:
            self.logger.record(
                "gex/lifetime_n_buckets_mean",
                np.mean(self._rollout_lifetime_n_buckets),
            )

        return True

    # ------------------------------------------------------------------
    # PPO train step + SC-VAE online update
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._update_sc_vae()   # read buffer before SB3 swap_and_flatten mutates it
        super().train()


