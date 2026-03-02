from typing import Optional
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from sc_vae_wrapper import SCVAEEncoderWrapper
from gex_rollout_buffer import GEXRolloutBuffer
from geodesic_bonus import GeodesicExplorationBonus


class PPOGEX(PPO):
    """
    PPO with GEX intrinsic reward and online SC-VAE training.

    SC-VAE is trained online: after each PPO update, _update_sc_vae()
    trains on the most recent rollout buffer.  The encoder improves as
    the agent explores, at the cost of slow μ-distribution drift.
    """

    def __init__(
        self,
        *args,
        sc_vae=None,
        gex_modules=None,
        rms=None,
        eta: float = 1.0,
        **kwargs,
    ):
        # Must be set before super().__init__() because SB3 calls
        # _setup_model() inside it.
        self.sc_vae = sc_vae
        self.gex_modules = gex_modules   # list, one GeodesicExplorationBonus per env
        self.rms = rms
        self.eta = float(eta)
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
            "PPOGEX + SCVAE currently supports Discrete action spaces only."

        # Replace SB3 rollout buffer with our extended version.
        self.rollout_buffer = GEXRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        if self.sc_vae is not None:
            self._sc_vae_wrap = SCVAEEncoderWrapper(self.sc_vae, self.device)

        if self.gex_modules is not None:
            assert len(self.gex_modules) == self.n_envs, (
                f"Need one GEX module per env. "
                f"Got {len(self.gex_modules)} but n_envs={self.n_envs}"
            )

    # ------------------------------------------------------------------
    # SC-VAE online update
    # ------------------------------------------------------------------

    def _update_sc_vae(self):
        """
        Train SC-VAE for one pass over the current rollout buffer.

        Shape-agnostic: next_observations is a custom field that SB3 never
        touches, so it is always (n_steps, n_envs, *obs_shape).  We use it
        as the source of truth for the total sample count and obs shape,
        then reshape obs and actions to match — regardless of whether
        super().train() has already called swap_and_flatten on them.
        """
        if self.sc_vae is None:
            return

        self.sc_vae.train()

        # next_observations: always (n_steps, n_envs, *obs_shape) — never mutated by SB3.
        next_obs = self.rollout_buffer.next_observations
        n_steps, n_envs = next_obs.shape[:2]
        n_total  = n_steps * n_envs
        obs_shape = next_obs.shape[2:]

        # Flatten to (n_total, *obs_shape).
        # If obs was already flattened by SB3's swap_and_flatten this is a no-op;
        # if it's still in (n_steps, n_envs, *obs_shape) form it reshapes correctly.
        next_obs_flat = next_obs.reshape(n_total, *obs_shape)
        obs_flat      = self.rollout_buffer.observations.reshape(n_total, *obs_shape)

        # actions: SB3 may have flattened to (n_total, 1) or left as (n_steps, n_envs).
        # Flatten to (n_total,) either way.
        actions_flat = self.rollout_buffer.actions.reshape(n_total, -1).squeeze(-1)

        batch_size = 256
        indices    = np.random.permutation(n_total)

        for start in range(0, n_total, batch_size):
            idx = indices[start : start + batch_size]

            s_t = th.as_tensor(obs_flat[idx],      device=self.device)
            s_n = th.as_tensor(next_obs_flat[idx], device=self.device)
            a_t = th.as_tensor(actions_flat[idx],  device=self.device)

            out            = self.sc_vae(s_t, a_t, s_n)
            l_recon, l_kl  = self.sc_vae.loss(out)
            loss           = l_recon + self.sc_vae.cfg.beta * l_kl

            self.sc_vae_optimizer.zero_grad()
            loss.backward()
            self.sc_vae_optimizer.step()

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

        # If sc_vae was not provided fall back to plain SB3 behaviour.
        if self._sc_vae_wrap is None:
            return super().collect_rollouts(
                env, callback, rollout_buffer, n_rollout_steps
            )

        # ----------------------------------------------------------
        # Tracking lists for logging (last rollout only)
        # ----------------------------------------------------------
        self._rollout_r_int              = []
        self._rollout_r_ext              = []
        self._rollout_r_total            = []
        self._rollout_epi_size           = []
        self._rollout_lifetime_n_buckets = []

        self.policy.set_training_mode(False)
        rollout_buffer.reset()

        # NOTE: Do NOT reset episodic memory here.
        # Episodic memory resets on episode boundaries (dones), not on
        # rollout-buffer boundaries.  Rollouts can cross episode borders.

        callback.on_rollout_start()

        n_steps = 0
        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()
            new_obs, rewards_ext, dones, infos = env.step(actions_np)

            # ----------------------------------------------------------
            # FIX (Bug 1): recover true terminal observations.
            # SB3 auto-resets done envs so new_obs[i] is already the
            # first obs of the next episode.  The real last obs before
            # reset lives in infos[i]["terminal_observation"].
            # ----------------------------------------------------------
            real_next_obs = new_obs.copy()
            for i in range(env.num_envs):
                if dones[i] and "terminal_observation" in infos[i]:
                    real_next_obs[i] = infos[i]["terminal_observation"]

            # ----------------------------------------------------------
            # Encode current transitions (uses pre-step obs).
            # ----------------------------------------------------------
            mu_batch = self._sc_vae_wrap.encode_mu(
                self._last_obs, actions_np, real_next_obs
            )  # (n_envs, d)
            mu_cpu = mu_batch.detach().cpu()

            # ----------------------------------------------------------
            # Per-env intrinsic reward computation.
            # Terminal transitions get zero reward; episodic memory is
            # reset at the episode boundary AFTER computing this step's
            # bonus so the terminal transition still gets scored against
            # its own episode's memory.
            # ----------------------------------------------------------
            r_int     = np.zeros_like(rewards_ext, dtype=np.float32)
            gex_infos = [None] * env.num_envs

            if self.gex_modules is not None:
                for i in range(env.num_envs):
                    ri, info      = self.gex_modules[i].step(mu_cpu[i])
                    r_int[i]      = float(ri)
                    gex_infos[i]  = info

                    # --------------------------------------------------
                    # FIX (Bug 2 + 3): reset episodic memory AFTER
                    # scoring this transition, not before.  Then seed
                    # the fresh memory with a dummy no-op transition for
                    # s0 so the first real transition of the new episode
                    # is scored against something.
                    # --------------------------------------------------
                    if dones[i]:
                        self.gex_modules[i].reset()

                        s0    = new_obs[i]          # first obs of new episode
                        no_op = self.sc_vae.cfg.no_op_action
                        mu0   = self._sc_vae_wrap.encode_mu(
                            s0[None],
                            np.array([no_op]),
                            s0[None],
                        )[0].detach().cpu()

                        # Seed memory without generating a reward.
                        self.gex_modules[i].episodic.query_and_add(mu0)

            # ----------------------------------------------------------
            # RMS normalisation.
            # ----------------------------------------------------------
            r_int_norm = r_int.copy()
            if self.rms is not None:
                r_int_tensor = th.as_tensor(r_int, dtype=th.float32)
                self.rms.update(r_int_tensor)
                r_int_norm = self.rms.normalize(r_int_tensor).numpy()

            rewards_total = rewards_ext + self.eta * r_int_norm

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            for i in range(env.num_envs):
                self._rollout_r_int.append(float(r_int[i]))
                self._rollout_r_ext.append(float(rewards_ext[i]))
                self._rollout_r_total.append(float(rewards_total[i]))

                if gex_infos[i] is not None:
                    if "episodic_size" in gex_infos[i]:
                        self._rollout_epi_size.append(
                            float(gex_infos[i]["episodic_size"])
                        )
                    # FIX (Bug 4): use the key that geodesic_bonus.py
                    # actually puts in info.
                    if "lifetime_n_buckets" in gex_infos[i]:
                        self._rollout_lifetime_n_buckets.append(
                            float(gex_infos[i]["lifetime_n_buckets"])
                        )

            # ----------------------------------------------------------
            # Buffer storage
            # ----------------------------------------------------------
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

            self._last_obs             = new_obs
            self._last_episode_starts  = dones

            n_steps += 1
            callback.update_locals(locals())
            if not callback.on_step():
                return False

        # ----------------------------------------------------------
        # GAE / returns
        # ----------------------------------------------------------
        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=self._last_episode_starts
        )

        callback.on_rollout_end()

        # ----------------------------------------------------------
        # Emit to SB3 logger
        # ----------------------------------------------------------
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

    def train(self):
        super().train()
        self._update_sc_vae()