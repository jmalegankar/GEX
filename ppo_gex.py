from typing import Optional
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from scvae_wrapper import SCVAEEncoderWrapper
from gex_rollout_buffer import GEXRolloutBuffer
from geodesic_bonus import GeodesicExplorationBonus

class PPOGEX(PPO):
    """
    PPO with GEX intrinsic reward injected during rollouts.

    Requirements:
      - self.gex_modules: list length n_envs, each has .reset() and .step(mu_i)->float
      - self.rms: has .update(x) and .normalize(x) (you said you already have reward_normalizer.py)
      - self.encoder: SCVAEEncoderWrapper (batch encode)
    """

    def __init__(
        self,
        *args,
        scvae=None,
        gex_modules=None,
        rms=None,
        eta: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scvae = scvae
        self.gex_modules = gex_modules  # list, one per env
        self.rms = rms
        self.eta = float(eta)

        self._scvae_wrap: Optional[SCVAEEncoderWrapper] = None
        self.scvae_optimizer = None
        if self.scvae is not None:
            self.scvae_optimizer = th.optim.Adam(
                self.scvae.parameters(),
                lr=self.scvae.cfg.lr,
            )

    def _setup_model(self) -> None:
        super()._setup_model()
        from gymnasium import spaces
        assert isinstance(self.action_space, spaces.Discrete), \
            "PPOGEX + SCVAE currently supports Discrete action spaces only."

        # replace rollout buffer
        self.rollout_buffer = GEXRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # SCVAE wrapper
        if self.scvae is not None:
            self._scvae_wrap = SCVAEEncoderWrapper(self.scvae, self.device)

        # sanity for env-count
        if self.gex_modules is not None:
            assert len(self.gex_modules) == self.n_envs, (
                f"Need one GEX module per env. Got len(gex_modules)={len(self.gex_modules)} "
                f"but n_envs={self.n_envs}"
            )

    def _update_scvae(self):
        if self.scvae is None:
            return

        self.scvae.train()

        obs = self.rollout_buffer.observations
        next_obs = self.rollout_buffer.next_observations
        actions = self.rollout_buffer.actions

        # flatten rollout dimension
        n_steps, n_envs = obs.shape[:2]

        obs = obs.reshape(n_steps * n_envs, *obs.shape[2:])
        next_obs = next_obs.reshape(n_steps * n_envs, *next_obs.shape[2:])
        actions = actions.reshape(n_steps * n_envs)
        if actions.ndim > 1:
            actions = actions.squeeze(-1)

        batch_size = 256
        n_samples = obs.shape[0]

        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            s_t = th.as_tensor(obs[idx], device=self.device)
            s_n = th.as_tensor(next_obs[idx], device=self.device)
            a_t = th.as_tensor(actions[idx], device=self.device)

            out = self.scvae(s_t, a_t, s_n)
            l_recon, l_kl = self.scvae.loss(out)

            loss = l_recon + self.scvae.cfg.beta * l_kl

            self.scvae_optimizer.zero_grad()
            loss.backward()
            self.scvae_optimizer.step()
        self.scvae.eval()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: GEXRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Copy of SB3 PPO.collect_rollouts with intrinsic reward injection.
        """
        assert self._last_obs is not None
        assert self._scvae_wrap is not None, "scvae wrapper not initialized"


        self._rollout_r_int = []
        self._rollout_r_ext = []
        self._rollout_r_total = []
        self._rollout_epi_size = []
        self._rollout_lifetime_buckets = []

        self.policy.set_training_mode(False)
        rollout_buffer.reset()


        # reset episodic modules at rollout start if needed
        # (SB3 rollouts can cross episode boundaries; we also reset per done below)
        if self.gex_modules is not None:
            for m in self.gex_modules:
                m.reset()

        callback.on_rollout_start()

        n_steps = 0
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()
            new_obs, rewards_ext, dones, infos = env.step(actions_np)

            # SB3: if done, env auto-resets; terminal_observation holds the real last obs
            real_next_obs = new_obs.copy()
            for i in range(env.num_envs):
                # If episode ended
                if dones[i]:

                    # Reset episodic memory
                    self.gex_modules[i].reset()

                    # SB3 auto-resets env; new_obs[i] is start of next episode
                    s0 = new_obs[i]

                    no_op = self.scvae.cfg.no_op_action

                    # Compute dummy transition (s0, no_op, s0)
                    mu0 = self._scvae_wrap.encode_mu(
                        s0[None],
                        np.array([no_op]),
                        s0[None],
                    )[0]

                    # Ensure CPU tensor for GEX
                    mu0 = mu0.detach().cpu()

                    # Insert into episodic memory WITHOUT reward
                    self.gex_modules[i].episodic.query_and_add(mu0)

            # --- compute intrinsic reward ---
            r_int = np.zeros_like(rewards_ext, dtype=np.float32)

            if self.gex_modules is not None:
                gex_infos = [None] * env.num_envs

                mu_batch = self._scvae_wrap.encode_mu(self._last_obs, actions_np, real_next_obs)  # (n_envs, d)

                # per-env intrinsic
                mu_cpu = mu_batch.detach().cpu()
                for i in range(env.num_envs):
                    ri = self.gex_modules[i].step(mu_cpu[i])
                    if isinstance(ri, tuple):
                        ri, info = ri
                    else:
                        info = {}
                        
                    r_int[i] = float(ri)
                    gex_infos[i] = info

            # --- RMS normalization (recommended to keep) ---
            r_int_norm = r_int.copy()
            if self.rms is not None:
                # update using mean over envs (or per-env update; choose one consistent rule)
                for i in range(env.num_envs):
                    self.rms.update(float(r_int[i]))
                    r_int_norm[i] = float(self.rms.normalize(float(r_int[i])))

            # total reward PPO uses for returns/advantages
            rewards_total = rewards_ext + self.eta * r_int_norm

            for i in range(env.num_envs):
                self._rollout_r_int.append(float(r_int[i]))
                self._rollout_r_ext.append(float(rewards_ext[i]))
                self._rollout_r_total.append(float(rewards_total[i]))

                if gex_infos[i] is not None:
                    if "episodic_size" in gex_infos[i]:
                        self._rollout_epi_size.append(float(gex_infos[i]["episodic_size"]))

                    if "lifetime_buckets" in gex_infos[i]:
                        self._rollout_lifetime_buckets.append(
                            float(gex_infos[i]["lifetime_buckets"])
                        )

            # episode_start flag for buffer (SB3 uses this instead of done directly)
            episode_start = self._last_episode_starts

            # store transition
            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards_total,
                episode_start,
                values,
                log_probs,
                next_obs=real_next_obs,
                intrinsic_reward=r_int,
                extrinsic_reward=rewards_ext,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

            # reset episodic gex module for envs that ended
            if self.gex_modules is not None:
                for i in range(env.num_envs):
                    if dones[i]:
                        self.gex_modules[i].reset()

            n_steps += 1
            callback.update_locals(locals())
            if not callback.on_step():
                return False

        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)

        callback.on_rollout_end()



        if len(self._rollout_r_int) > 0:
            self.logger.record("gex/r_int_mean", np.mean(self._rollout_r_int))
            self.logger.record("gex/r_ext_mean", np.mean(self._rollout_r_ext))
            self.logger.record("gex/r_total_mean", np.mean(self._rollout_r_total))

        if len(self._rollout_epi_size) > 0:
            self.logger.record("gex/episodic_size_mean", np.mean(self._rollout_epi_size))

        if len(self._rollout_lifetime_buckets) > 0:
            self.logger.record(
                "gex/lifetime_buckets_mean",
                np.mean(self._rollout_lifetime_buckets),
            )
        return True
    
    def train(self):
        super().train()
        self._update_scvae()