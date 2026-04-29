"""
LMU-PPO: PPO with a swappable memory cell (gated_lmu / vanilla_lmu / gru / lstm)
and an optional E3B episodic bonus. POPGym-only.

Reward composition:
    r_t = r_ext_t
          + beta     * r_intr_t  * (1 - episode_start_t)   [lifelong]
          + beta_ep  * b_t_norm  * (1 - episode_start_t)   [episodic]
"""

from typing import Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)

from .buffer import LMURolloutBuffer
from .episodic_bonus import EllipticalEpisodicBonus, RunningStd
from .policies import LMUActorCriticPolicy
from .popgym_encoders import make_encoder


class LMUPPO(PPO):
    """
    PPO with a swappable recurrent cell and optional E3B episodic bonus.

    Set beta_ep=0.0 for lifelong-only.
    Set cell_type='vanilla_lmu' / 'gru' / 'lstm' for baseline comparisons.
    """

    policy: LMUActorCriticPolicy
    rollout_buffer: LMURolloutBuffer

    def __init__(
        self,
        env: GymEnv,
        policy=None,
        lr: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        encoder_dim: int = 64,
        hidden_size: int = 64,
        memory_size: int = 32,
        theta: float = 50.0,
        chunk_len: int = 16,
        n_chunks_per_batch: int = 16,
        beta: float = 0.001,
        beta_ep: float = 0.03,
        lambda_reg: float = 1.0,
        phi_source: str = 'y_readout',
        measure: str = 'LegT',
        gate_type: str = 'softsign_sum',
        residual_scale: float = 0.05,
        read_head: str = 'dynamic',
        postprocessor_dim: Optional[int] = None,
        head_layernorm: bool = False,
        cell_type: str = 'gated_lmu',
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.theta = theta
        self.chunk_len = chunk_len
        self.n_chunks_per_batch = n_chunks_per_batch
        self.beta = beta
        self.beta_ep = beta_ep
        self.lambda_reg = lambda_reg
        assert phi_source in (
            'y_readout', 'y_readout_unnorm',
            'random_encoder', 'encoder_detached',
            'innovation',
        ), f"phi_source invalid: {phi_source!r}"
        self.phi_source = phi_source
        self.measure = measure
        self.gate_type = gate_type
        self.residual_scale = residual_scale
        self.read_head = read_head
        self.postprocessor_dim = postprocessor_dim
        self.head_layernorm = head_layernorm
        self.cell_type = cell_type
        self.is_lmu = cell_type in ('gated_lmu', 'vanilla_lmu')

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
            measure=self.measure,
            gate_type=self.gate_type,
            residual_scale=self.residual_scale,
            read_head=self.read_head,
            cell_type=self.cell_type,
            postprocessor_dim=self.postprocessor_dim,
            head_layernorm=self.head_layernorm,
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

    # ------------------------------------------------------------------
    # Learn setup — E3B + phi_encoder + compatibility validation
    # ------------------------------------------------------------------

    def _setup_learn(self, total_timesteps, callback=None,
                     reset_num_timesteps=True, tb_log_name="lmu_ppo",
                     progress_bar=False):
        ret = super()._setup_learn(
            total_timesteps, callback, reset_num_timesteps,
            tb_log_name, progress_bar
        )
        self._lmu_h, self._lmu_m = self.policy.initial_state(
            self.n_envs, self.device
        )

        # LegS per-env step counter — only meaningful for LMU cells.
        if self.measure == 'LegS' and self.is_lmu:
            self._lmu_t = th.ones(self.n_envs, dtype=th.int32, device=self.device)
        else:
            self._lmu_t = None

        if self.verbose >= 1:
            print(f"  cell_type={self.cell_type}  measure={self.measure}  "
                  f"gate_type={self.gate_type}  read_head={self.read_head}  "
                  f"residual_scale={self.residual_scale}")

        # ── phi-source compatibility validation (only when beta_ep > 0) ──
        if self.beta_ep > 0:
            if self.phi_source in ('y_readout', 'y_readout_unnorm'):
                assert self.cell_type == 'gated_lmu' and self.read_head == 'dynamic', (
                    f"phi_source={self.phi_source!r} requires cell_type='gated_lmu' "
                    f"with read_head='dynamic' (needs W_query); got cell_type="
                    f"{self.cell_type!r}, read_head={self.read_head!r}."
                )
            if self.phi_source == 'innovation':
                assert self.is_lmu, (
                    f"phi_source='innovation' requires an LMU cell "
                    f"(needs E_h and e_m); got cell_type={self.cell_type!r}."
                )

        # ── E3B objects ──
        if self.beta_ep > 0:
            self._ep_bonus = EllipticalEpisodicBonus(
                n_envs=self.n_envs,
                dim=self.encoder_dim,
                lambda_reg=self.lambda_reg,
                device=self.device,
            )
            self._b_running_std = RunningStd(epsilon=1e-4)

            if self.phi_source == 'random_encoder':
                self._phi_encoder = make_encoder(
                    self.observation_space, out_dim=self.encoder_dim,
                ).to(self.device)
                for p in self._phi_encoder.parameters():
                    p.requires_grad = False
                self._phi_encoder.eval()
                if self.verbose >= 1:
                    n = sum(p.numel() for p in self._phi_encoder.parameters())
                    print(f"  phi_source=random_encoder ({n:,} frozen params)")
            else:
                self._phi_encoder = None
                if self.verbose >= 1:
                    print(f"  phi_source={self.phi_source}")
        else:
            self._ep_bonus = None
            self._b_running_std = None
            self._phi_encoder = None

        return ret

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        rollout_buffer.reset()
        callback.on_rollout_start()

        # Diagnostic buffers
        r_intr_buf = []
        prod_buf   = []
        u_x_buf    = []
        gate_buf   = []
        innov_buf  = []
        ep_start_buf = []
        b_buf = []

        beta    = self.beta
        beta_ep = self.beta_ep

        n_steps = 0
        while n_steps < n_rollout_steps:

            with th.no_grad():
                obs_t = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, h_new, m_new, logits_t, r_intr, \
                    gate, innov, u_x = \
                    self.policy.forward(
                        obs_t, self._lmu_h, self._lmu_m, t=self._lmu_t
                    )

                prod = gate * innov

                # ── E3B bonus ────────────────────────────────────────────
                if self._ep_bonus is not None:
                    if self.phi_source == 'y_readout':
                        C_t = F.normalize(
                            self.policy.lmu_cell.W_query(self._lmu_h), dim=-1
                        )
                        phi_t = th.einsum('bd,bdc->bc', C_t, m_new)
                    elif self.phi_source == 'y_readout_unnorm':
                        C_t = self.policy.lmu_cell.W_query(self._lmu_h)
                        phi_t = th.einsum('bd,bdc->bc', C_t, m_new)
                    elif self.phi_source == 'innovation':
                        lmu = self.policy.lmu_cell
                        u_h = lmu.E_h(self._lmu_h)
                        u_m = th.einsum('d,bdc->bc', lmu.e_m, self._lmu_m)
                        phi_t = (u_x - u_h - u_m).detach()
                    elif self.phi_source == 'random_encoder':
                        phi_t = self._phi_encoder(obs_t)
                    elif self.phi_source == 'encoder_detached':
                        phi_t = self.policy.encoder(obs_t).detach()
                    else:
                        raise ValueError(
                            f"Unknown phi_source: {self.phi_source}"
                        )

                    b_raw = self._ep_bonus.bonus_and_update(phi_t)

                    self._b_running_std.update(b_raw.cpu().numpy())
                    b_norm = (b_raw.cpu().numpy()
                              / (self._b_running_std.std + 1e-6))

                    b_buf.append(b_norm)

            ep_start_buf.append(self._last_episode_starts.copy())

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            r_intr_buf.append(r_intr)
            prod_buf.append(prod)
            u_x_buf.append(u_x)
            gate_buf.append(gate)
            innov_buf.append(innov)

            n_steps += 1

            # ── Reward combination ────────────────────────────────────
            mask = (1.0 - self._last_episode_starts)
            r_intr_masked = r_intr.cpu().numpy() * mask
            rewards_combined = rewards + beta * r_intr_masked

            if self._ep_bonus is not None:
                b_masked = b_norm * mask
                rewards_combined = rewards_combined + beta_ep * b_masked

            rollout_buffer.add(
                self._last_obs,
                actions_np.reshape(-1, 1),
                rewards_combined,
                self._last_episode_starts,
                values,
                log_probs,
                self._lmu_h,
                self._lmu_m,
                lmu_t=self._lmu_t,
            )

            self._lmu_h = h_new.clone()
            self._lmu_m = m_new.clone()

            if self._lmu_t is not None:
                self._lmu_t = self._lmu_t + 1

            # Reset cell state and E3B buffer at episode boundaries.
            done_envs = np.where(dones)[0].tolist()
            for i in done_envs:
                self._lmu_h[i].zero_()
                self._lmu_m[i].zero_()
                if self._lmu_t is not None:
                    self._lmu_t[i] = 1
            if self._ep_bonus is not None and done_envs:
                self._ep_bonus.reset(done_envs)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_t = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(
                obs_t, self._lmu_h, self._lmu_m, t=self._lmu_t
            )

        rollout_buffer.compute_returns_and_advantage(values, dones)

        # ────────── Per-rollout diagnostics ──────────────────────────
        r_intr_all = th.stack(r_intr_buf).cpu()
        prod_all   = th.stack(prod_buf).cpu()
        u_x_all    = th.stack(u_x_buf).cpu()
        gate_all   = th.stack(gate_buf).cpu()
        innov_all  = th.stack(innov_buf).cpu()
        ep_start_all = np.stack(ep_start_buf)

        # ── r_intr (zero for GRU/LSTM, kept for uniform logging) ─────
        self.logger.record("debug/r_intr_mean", r_intr_all.mean().item())
        self.logger.record("debug/r_intr_max",  r_intr_all.max().item())

        # ── LMU-specific diagnostics — gate on is_lmu ────────────────
        if self.is_lmu:
            e_x_norm = F.normalize(self.policy.lmu_cell.e_x, dim=0).norm().item()
            self.logger.record("debug/e_x_norm", e_x_norm)

            ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
            self.logger.record("debug/W_pre_ortho_error", ortho_err)
            if ortho_err > 1e-3:
                if self.verbose >= 1:
                    print(f"  [warn] W_pre ortho error = {ortho_err:.2e} — SVD reset")
                self.policy.lmu_cell.W_pre.reorthogonalize()

        # ── Memory magnitude (zero for GRU; LSTM packs c-state here) ─
        m_norm = self._lmu_m.norm(dim=(1, 2)).mean().item()
        self.logger.record("debug/m_norm", m_norm)
        self.logger.record("intrinsic/beta",    beta)
        self.logger.record("intrinsic/beta_ep", beta_ep)
        self.logger.record(
            "intrinsic/phi_source",
            {'y_readout': 0, 'random_encoder': 1, 'encoder_detached': 2,
             'y_readout_unnorm': 3, 'innovation': 4}.get(self.phi_source, -1),
        )
        self.logger.record("intrinsic/r_intr_contribution",
                           beta * r_intr_all.mean().item())
        self.logger.record(
            "intrinsic/cell_type",
            {'gated_lmu': 0, 'vanilla_lmu': 1, 'gru': 2, 'lstm': 3}.get(
                self.cell_type, -1
            ),
        )

        # ── Cold-start sign bias / per-channel novelty — LMU only ────
        if self.is_lmu:
            ep_start_t = th.from_numpy(ep_start_all)
            prod_cold = prod_all[ep_start_t]
            prod_warm = prod_all[~ep_start_t]

            if prod_cold.numel() > 0:
                self.logger.record("debug/prod_positive_frac_cold_start",
                                   (prod_cold > 0).float().mean().item())
                gate_cold  = gate_all[ep_start_t]
                innov_cold = innov_all[ep_start_t]
                self.logger.record("debug/gate_positive_frac_cold",
                                   (gate_cold > 0).float().mean().item())
                self.logger.record("debug/innov_positive_frac_cold",
                                   (innov_cold > 0).float().mean().item())
            if prod_warm.numel() > 0:
                self.logger.record("debug/prod_positive_frac_warm",
                                   (prod_warm > 0).float().mean().item())
            self.logger.record("debug/prod_positive_frac_all",
                               (prod_all > 0).float().mean().item())
            self.logger.record("debug/gate_abs_mean",  gate_all.abs().mean().item())
            self.logger.record("debug/innov_abs_mean", innov_all.abs().mean().item())

            self.logger.record("debug/u_x_zero_frac",
                               (u_x_all.abs() < 1e-6).float().mean().item())

            per_chan = prod_all.abs().mean(dim=(0, 1))
            self.logger.record("debug/per_channel_mean", per_chan.mean().item())
            self.logger.record("debug/per_channel_std",  per_chan.std().item())
            self.logger.record("debug/per_channel_max",  per_chan.max().item())
            active = (per_chan > 0.1 * per_chan.max()).float().sum().item()
            self.logger.record("debug/active_channel_count", active)

        # ── E3B diagnostics ───────────────────────────────────────────
        if self._ep_bonus is not None and b_buf:
            b_all = np.stack(b_buf)
            self.logger.record("episodic/b_mean",    float(b_all.mean()))
            self.logger.record("episodic/b_max",     float(b_all.max()))
            self.logger.record("episodic/b_std",     float(b_all.std()))
            self.logger.record("episodic/b_running_std",
                               self._b_running_std.std)
            self.logger.record("episodic/b_contribution",
                               beta_ep * float(b_all.mean()))

            safety = self._ep_bonus.get_diagnostics()
            self.logger.record("episodic/M_min_eigval",
                               self._ep_bonus.min_eigenvalue())
            self.logger.record("episodic/n_skipped",  safety['n_skipped'])
            self.logger.record("episodic/n_reset",    safety['n_reset'])
            self.logger.record("episodic/b_max",      safety['b_max'])
            self.logger.record("episodic/skip_frac",  safety['skip_frac'])
            self.logger.record("episodic/reset_frac", safety['reset_frac'])

            if self.verbose >= 1:
                if safety['n_reset'] > 0:
                    print(f"  [warn] M was reset {safety['n_reset']} times "
                          f"(b_true went negative — M corruption recovery)")
                if safety['skip_frac'] > 0.01:
                    print(f"  [warn] {safety['skip_frac']:.2%} of M updates skipped "
                          f"(b_true non-finite or > {self._ep_bonus.reject_threshold:.0e}); "
                          f"b_max = {safety['b_max']:.3g}")
                e_min = self._ep_bonus.min_eigenvalue()
                if e_min < 0:
                    print(f"  [ERROR] M_min_eigval = {e_min:.3e} — math is broken, "
                          f"investigate immediately")

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

        is_legs = (self.measure == 'LegS') and self.is_lmu

        if state is None:
            h, m = self.policy.initial_state(n, self.device)
            t = th.ones(n, dtype=th.int32, device=self.device) if is_legs else None
        else:
            if is_legs and len(state) == 3:
                h, m, t = state
            else:
                h, m = state[0], state[1]
                t = th.ones(n, dtype=th.int32, device=self.device) if is_legs else None

        if episode_start is not None:
            dones = th.as_tensor(episode_start, dtype=th.bool, device=self.device)
            h = h.clone(); m = m.clone()
            h[dones] = 0.0; m[dones] = 0.0
            if t is not None:
                t = t.clone()
                t[dones] = 1

        with th.no_grad():
            actions, _, _, h_new, m_new, _, _, _, _, _ = \
                self.policy.forward(obs_tensor, h, m, t=t)

        actions = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        if t is not None:
            return actions, (h_new, m_new, t + 1)
        return actions, (h_new, m_new)

    # ------------------------------------------------------------------
    # PPO update
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
        wpre_grad_norms = []
        comp_grad_norms = {
            "encoder": [], "lmu_cell": [], "actor": [], "critic": []
        }

        has_W_pre = hasattr(self.policy.lmu_cell, 'W_pre')

        continue_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.n_chunks_per_batch):

                values, log_prob, entropy, r_intrs = self.policy.evaluate_actions(
                    obs_seq=batch.observations,
                    lmu_h=batch.lmu_h,
                    lmu_m=batch.lmu_m,
                    episode_starts=batch.episode_starts,
                    actions_seq=batch.actions,
                    lmu_t=batch.lmu_t if (self.measure == 'LegS' and self.is_lmu) else None,
                )

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / \
                                 (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - batch.old_log_prob)
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

                loss = (policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss)

                with th.no_grad():
                    log_ratio = log_prob - batch.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).item()
                    approx_kl_divs.append(approx_kl_div)

                if (self.target_kl is not None
                        and approx_kl_div > 1.5 * self.target_kl):
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"  Early stopping epoch {epoch}, "
                              f"KL={approx_kl_div:.3f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # W_pre Riemannian update — only for cells that have W_pre.
                if has_W_pre:
                    if self.policy.lmu_cell.W_pre.weights.grad is not None:
                        wpre_grad_norms.append(
                            self.policy.lmu_cell.W_pre.weights.grad.norm().item()
                        )
                    self.policy.lmu_cell.W_pre.ortho_update(lr=1e-3)

                _per_comp = []
                for _name, _mod in [
                    ("encoder",  self.policy.encoder),
                    ("lmu_cell", self.policy.lmu_cell),
                    ("actor",    self.policy.actor),
                    ("critic",   self.policy.critic),
                ]:
                    pre_clip = th.nn.utils.clip_grad_norm_(
                        _mod.parameters(), self.max_grad_norm
                    ).item()
                    comp_grad_norms[_name].append(pre_clip)
                    _per_comp.append(pre_clip)

                grad_norm = max(_per_comp)
                self.policy.optimizer.step()

                if has_W_pre and self._n_updates % 100 == 0:
                    ortho_err = self.policy.lmu_cell.W_pre.orthogonality_error()
                    if ortho_err > 1e-3:
                        if self.verbose >= 1:
                            print(f"  [warn] W_pre ortho={ortho_err:.2e} "
                                  f"at update {self._n_updates} — SVD reset")
                        self.policy.lmu_cell.W_pre.reorthogonalize()

                r_intrs_log.append(r_intrs.mean().item())
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )
                grad_norms.append(grad_norm)
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
        self.logger.record("train/n_updates",
                           self._n_updates, exclude="tensorboard")
        self.logger.record("train/learning_rate",      lr)
        self.logger.record("train/clip_range",         clip_range)
        self.logger.record("debug/r_intrs_train_mean", np.mean(r_intrs_log))

        for name, values_list in comp_grad_norms.items():
            if values_list:
                self.logger.record(f"grad/{name}_norm",
                                   float(np.mean(values_list)))
        if wpre_grad_norms:
            self.logger.record("debug/W_pre_grad_norm",
                               float(np.mean(wpre_grad_norms)))
