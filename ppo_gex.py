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
    PPO with GEX intrinsic reward injected during rollouts.

    Requirements:
      - self.gex_modules: list length n_envs, each has .reset() and .step(mu_i)->float
      - self.rms: has .update(x) and .normalize(x) (you said you already have reward_normalizer.py)
      - self.encoder: SCVAEEncoderWrapper (batch encode)
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
        # Must be set before super().__init__() because SB3 calls _setup_model() inside it.
        self.sc_vae = sc_vae
        self.gex_modules = gex_modules  # list, one per env
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
        if self.sc_vae is not None:
            self._sc_vae_wrap = SCVAEEncoderWrapper(self.sc_vae, self.device)

        # sanity for env-count
        if self.gex_modules is not None:
            assert len(self.gex_modules) == self.n_envs, (
                f"Need one GEX module per env. Got len(gex_modules)={len(self.gex_modules)} "
                f"but n_envs={self.n_envs}"
            )

    def _update_sc_vae(self):
        if self.sc_vae is None:
            return

        self.sc_vae.train()

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

            out = self.sc_vae(s_t, a_t, s_n)
            l_recon, l_kl = self.sc_vae.loss(out)

            loss = l_recon + self.sc_vae.cfg.beta * l_kl

            self.sc_vae_optimizer.zero_grad()
            loss.backward()
            self.sc_vae_optimizer.step()
        self.sc_vae.eval()

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
        assert self._sc_vae_wrap is not None, "sc_vae wrapper not initialized"


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
            # Bug fix: guard gex_modules usage; reset only once here (removed duplicate below)
            if self.gex_modules is not None:
                for i in range(env.num_envs):
                    if dones[i]:
                        # Reset episodic memory
                        self.gex_modules[i].reset()

                        # SB3 auto-resets env; new_obs[i] is start of next episode
                        s0 = new_obs[i]

                        no_op = self.sc_vae.cfg.no_op_action

                        # Compute dummy transition (s0, no_op, s0)
                        mu0 = self._sc_vae_wrap.encode_mu(
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
            gex_infos = [None] * env.num_envs  # always defined; populated below if gex active

            if self.gex_modules is not None:
                mu_batch = self._sc_vae_wrap.encode_mu(self._last_obs, actions_np, real_next_obs)  # (n_envs, d)

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

            # --- RMS normalization ---
            r_int_norm = r_int.copy()
            if self.rms is not None:
                r_int_tensor = th.as_tensor(r_int, dtype=th.float32)
                self.rms.update(r_int_tensor)
                r_int_norm = self.rms.normalize(r_int_tensor).numpy()

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
        self._update_sc_vae()