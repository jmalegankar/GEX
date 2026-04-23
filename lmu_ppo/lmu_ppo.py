"""
LMU-PPO: PPO with LMU recurrent policy — Phase 2 (E3B episodic bonus).

Changes from Phase 1:
─────────────────────
1. EllipticalEpisodicBonus integrated into collect_rollouts.
   y_t is extracted from (h_prev, m_new) inside the no_grad block, before
   callback.on_step(). beta_ep * normalize(b_t) is added to rewards.

2. RunningMeanStd normalises raw E3B bonuses. Raw b_t is heavy-tailed
   (early-episode steps can be 10–100× late-episode values). Without
   normalisation, beta_ep tuning is entangled with lambda_reg and episode
   length. With normalisation, beta_ep directly controls the contribution
   in std-normalised units — comparable to typical extrinsic reward scales.

3. beta renamed to beta_life (lifetime/step intrinsic reward from gated write).
   beta_ep is the new episodic E3B bonus weight.

4. No masking of b_t at episode boundaries. The SM matrix is reset BEFORE
   the first step of each new episode (see EllipticalEpisodicBonus.reset),
   so b_0 is the correct high-novelty value and should contribute to the
   reward. r_intr is still masked (LMU state is stale at episode transitions).

5. New tensorboard keys:
     debug/b_ep_mean, debug/b_ep_max, debug/b_ep_std   — raw bonus stats
     debug/b_ep_rms_std                                  — running normaliser
     debug/b_ep_norm_mean                                — normalised bonus mean

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
from .episodic_bonus import EllipticalEpisodicBonus, RunningMeanStd
import os


class LMUPPO(PPO):
    """
    PPO with LMU recurrent policy + E3B episodic bonus.

    Reward combination at each step t, env i:
        r_combined[i] = r_ext[i]
                      + beta_life * r_intr[i] * (1 - episode_start[i])
                      + beta_ep   * b_ep_norm[i]

    b_ep_norm[i] = b_ep_raw[i] / (running_std + 1e-6)
    b_ep_raw[i]  = φ_t[i]^T M_{t-1}[i] φ_t[i]  (E3B elliptical bonus)
    φ_t[i]       = L2-normalise(y_t[i])
    y_t[i]       = LMU dynamic readout at this step

    Set beta_ep=0 to disable E3B (Phase 1 / ablation).
    Set beta_life=0 to disable the gated-write intrinsic signal.
    """

    policy: LMUActorCriticPolicy
    rollout_buffer: LMURolloutBuffer

    def __init__(
        self,
        env:                GymEnv,
        policy=None,
        lr:                 Union[float, Schedule] = 3e-4,
        n_steps:            int   = 2048,
        batch_size:         int   = 256,
        n_epochs:           int   = 10,
        gamma:              float = 0.999,
        gae_lambda:         float = 0.95,
        clip_range:         Union[float, Schedule] = 0.2,
        clip_range_vf:      Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef:           float = 0.01,
        vf_coef:            float = 0.5,
        max_grad_norm:      float = 0.5,
        target_kl:          Optional[float] = None,
        encoder_dim:        int   = 64,
        hidden_size:        int   = 64,
        memory_size:        int   = 32,
        theta:              float = 50.0,
        chunk_len:          int   = 16,
        n_chunks_per_batch: int   = 16,
        # [OLD] beta: float = 0.001,
        beta_life:          float = 0.001,   # gated-write intrinsic (step-level)
        beta_ep:            float = 0.0,     # E3B episodic bonus weight
        lambda_reg:         float = 1.0,     # E3B ridge: M_0 = (1/λ)·I
        tensorboard_log:    Optional[str] = None,
        verbose:            int   = 1,
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
        self.beta_life          = beta_life
        self.beta_ep            = beta_ep
        self.lambda_reg         = lambda_reg

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

        # E3B: initialise only if beta_ep > 0, avoid 64×64×n_envs allocation
        # when running ablations with beta_ep=0.
        if self.beta_ep > 0:
            self.ep_bonus = EllipticalEpisodicBonus(
                n_envs=self.n_envs,
                dim=self.encoder_dim,
                lambda_reg=self.lambda_reg,
                device=str(self.device),
            )
        else:
            self.ep_bonus = None

        # Running std for raw b_t normalisation.  Always init — it's cheap
        # and allows logging even in beta_ep=0 ablation if bonus is computed
        # for diagnostics.
        self.b_rms = RunningMeanStd()

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

        # Reset episodic state for fresh training run.
        if self.ep_bonus is not None:
            self.ep_bonus.reset_all()
        self.b_rms.reset()

        return ret

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        rollout_buffer.reset()
        callback.on_rollout_start()

        r_intr_buf = []
        prod_buf   = []
        u_x_buf    = []
        b_ep_buf   = []   # [NEW] raw E3B bonuses per step
        beta_life  = self.beta_life
        beta_ep    = self.beta_ep

        n_steps = 0
        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)

                actions, values, log_probs, h_new, m_new, logits_t, r_intr, \
                    gate, innov, u_x = \
                    self.policy.forward(obs_t, self._lmu_h, self._lmu_m)

                # [NEW] E3B: extract y and compute bonus INSIDE no_grad block,
                # before callback.on_step() which may fire policy.forward on
                # the eval env (n_envs=1), clobbering local state.
                #
                # y_t = C_t @ m_new  where C_t = normalize(W_query(h_prev)).
                # h_prev = self._lmu_h (before this step's update).
                # m_new  = returned from policy.forward above.
                # This mirrors extract_y() in diagnose_e3b.py exactly.
                if self.ep_bonus is not None:
                    C_t  = F.normalize(
                        self.policy.lmu_cell.W_query(self._lmu_h), dim=-1
                    )                                                   # (n_envs, d)
                    y_t  = th.einsum('bd,bdc->bc', C_t, m_new)         # (n_envs, C)
                    phi  = F.normalize(y_t, dim=-1)                    # (n_envs, C)
                    b_raw = self.ep_bonus.bonus_and_update(phi)        # (n_envs,) tensor
                else:
                    b_raw = th.zeros(self.n_envs, device=self.device)

                # Capture diagnostic tensors before any callback can clobber them
                prod  = (gate * innov)   # (n_envs, C) detached
                # u_x already detached from lmu_cell.forward

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            # Accumulate diagnostics — all local variables, safe from callbacks
            r_intr_buf.append(r_intr.cpu())
            prod_buf.append(prod.cpu())
            u_x_buf.append(u_x.cpu())

            b_raw_np = b_raw.cpu().numpy()
            b_ep_buf.append(b_raw_np.copy())

            n_steps += 1

            # [NEW] Normalise b_t using running std.
            # Update RMS first, then divide — each step's contribution is tiny
            # so self-referential update is negligible in practice.
            self.b_rms.update(b_raw_np)
            b_norm = b_raw_np / (self.b_rms.std + 1e-6)

            # Combine rewards.
            # r_intr masked at episode boundaries (LMU state stale after done).
            # b_norm NOT masked — SM matrix is correctly reset at boundaries.
            r_intr_masked = r_intr.cpu().numpy() * (1.0 - self._last_episode_starts)
            rewards_combined = (
                rewards
                + beta_life * r_intr_masked
                + beta_ep   * b_norm
            )

            rollout_buffer.add(
                self._last_obs,
                actions_np.reshape(-1, 1),
                rewards_combined,
                self._last_episode_starts,
                values,
                log_probs,
                self._lmu_h,
                self._lmu_m,
            )

            self._lmu_h = h_new.clone()
            self._lmu_m = m_new.clone()

            # Reset LMU state and E3B matrix for done envs.
            done_envs = []
            for i, done in enumerate(dones):
                if done:
                    self._lmu_h[i].zero_()
                    self._lmu_m[i].zero_()
                    done_envs.append(i)
            # [NEW] Reset E3B inverse for done envs — must happen AFTER
            # bonus_and_update so the LAST step of the episode uses the
            # correct (non-reset) matrix, and the FIRST step of the next
            # episode uses the freshly reset (1/λ)·I.
            if done_envs and self.ep_bonus is not None:
                self.ep_bonus.reset(done_envs)

            self._last_obs            = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_t  = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_t, self._lmu_h, self._lmu_m)

        rollout_buffer.compute_returns_and_advantage(values, dones)

        # ── Diagnostics ───────────────────────────────────────────────────────
        r_intr_all = th.stack(r_intr_buf)          # (n_steps, n_envs)
        self.logger.record("debug/r_intr_mean", r_intr_all.mean().item())
        self.logger.record("debug/r_intr_max",  r_intr_all.max().item())

        e_x_norm = F.normalize(self.policy.lmu_cell.e_x, dim=0).norm().item()
        self.logger.record("debug/e_x_norm", e_x_norm)

        ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
        self.logger.record("debug/W_pre_ortho_error", ortho_err)
        if ortho_err > 1e-3:
            if self.verbose >= 1:
                print(f"  [warn] W_pre ortho error = {ortho_err:.2e} — SVD reset")
            self.policy.lmu_cell.W_pre.reorthogonalize()

        m_norm = self._lmu_m.norm(dim=(1, 2)).mean().item()
        self.logger.record("debug/m_norm", m_norm)

        self.logger.record("intrinsic/beta_life",         beta_life)
        self.logger.record("intrinsic/beta_ep",           beta_ep)
        self.logger.record("intrinsic/r_intr_contribution",
                           beta_life * r_intr_all.mean().item())

        prod_all = th.stack(prod_buf)
        u_x_all  = th.stack(u_x_buf)
        self.logger.record("debug/prod_positive_frac",
                           (prod_all > 0).float().mean().item())
        self.logger.record("debug/u_x_zero_frac",
                           (u_x_all.abs() < 1e-6).float().mean().item())
        self.logger.record("debug/gate_innov_per_channel",
                           prod_all.abs().mean(dim=(0, 1)).tolist())

        # [NEW] E3B bonus diagnostics
        if b_ep_buf:
            b_all = np.stack(b_ep_buf)              # (n_steps, n_envs)
            self.logger.record("debug/b_ep_mean",     float(b_all.mean()))
            self.logger.record("debug/b_ep_max",      float(b_all.max()))
            self.logger.record("debug/b_ep_std",      float(b_all.std()))
            self.logger.record("debug/b_ep_rms_std",  self.b_rms.std)
            b_norm_all = b_all / (self.b_rms.std + 1e-6)
            self.logger.record("debug/b_ep_norm_mean", float(b_norm_all.mean()))
            self.logger.record("intrinsic/b_ep_contribution",
                               beta_ep * float(b_norm_all.mean()))

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
            actions, _, _, h_new, m_new, logits_t, _, _, _, _ = \
                self.policy.forward(obs_tensor, h, m)

        actions = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions, (h_new, m_new)

    # ------------------------------------------------------------------
    # PPO update (unchanged from Phase 1)
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
        r_intrs_log = []

        continue_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.n_chunks_per_batch):

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

                if self.policy.lmu_cell.W_pre.weights.grad is not None:
                    wpre_gn = self.policy.lmu_cell.W_pre.weights.grad.norm().item()
                    self.logger.record("debug/W_pre_grad_norm", wpre_gn)
                self.policy.lmu_cell.W_pre.ortho_update(lr=1e-3)

                _comp_norms = []
                for _comp in [self.policy.encoder, self.policy.lmu_cell,
                               self.policy.actor, self.policy.critic]:
                    _comp_norms.append(
                        th.nn.utils.clip_grad_norm_(
                            _comp.parameters(), self.max_grad_norm
                        ).item()
                    )
                grad_norm = max(_comp_norms)
                self.policy.optimizer.step()

                if self._n_updates % 100 == 0:
                    ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
                    if ortho_err > 1e-3:
                        if self.verbose >= 1:
                            print(f"  [warn] W_pre ortho error={ortho_err:.2e} "
                                  f"at update {self._n_updates} — SVD reset")
                        self.policy.lmu_cell.W_pre.reorthogonalize()

                r_intrs_log.append(r_intrs.mean().item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                grad_norms.append(
                    grad_norm if isinstance(grad_norm, float) else grad_norm.item()
                )
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
        self.logger.record("debug/r_intrs_train_mean", np.mean(r_intrs_log))