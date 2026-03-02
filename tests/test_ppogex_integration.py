import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv

from sb3.ppo_gex import PPOGEX
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus
from intrinsic_reward.reward_normalizer import RunningMeanStd
from models.sc_vae import TransitionSCVAE
from models.config import SCVAEConfig
from models.obs_embeddings import ObservationEmbedding


def test_ppogex_runs_one_iteration():

    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # ------------------------------------------------------------------
    # Minimal embedding: (B, D) vector obs -> (B, 1, D, 1) feature map.
    # out_channels = 1 so the conv encoder sees a (1, D, 1) "image".
    # ------------------------------------------------------------------
    class IdentityEmbedding(ObservationEmbedding):
        def __init__(self, obs_dim):
            super().__init__()
            self.obs_dim = obs_dim

        @property
        def out_channels(self):
            return 1

        def forward(self, obs):
            return obs.float().unsqueeze(1).unsqueeze(-1)   # (B,1,D,1)

    obs_dim   = env.observation_space.shape[0]
    embedding = IdentityEmbedding(obs_dim)
    cfg       = SCVAEConfig(n_actions=env.action_space.n)

    scvae = TransitionSCVAE(
        embedding=embedding,
        cfg=cfg,
        sample_input_shape=(obs_dim,),
    )

    gex_modules = [GeodesicExplorationBonus(mu_dim=cfg.latent_dim)]
    rms         = RunningMeanStd()

    model = PPOGEX(
        "MlpPolicy",
        env,
        sc_vae=scvae,
        gex_modules=gex_modules,
        rms=rms,
        n_steps=32,
        batch_size=32,
        learning_rate=3e-4,
        verbose=0,
    )

    # Snapshot parameters before training.
    params_before = {
        n: p.clone().detach() for n, p in scvae.named_parameters()
    }

    model.learn(total_timesteps=64)

    # ------------------------------------------------------------------
    # 1. SC-VAE parameters must have changed (online training happened).
    # ------------------------------------------------------------------
    params_changed = any(
        not th.allclose(params_before[n], p.detach())
        for n, p in scvae.named_parameters()
    )
    print("SC-VAE parameters changed:", params_changed)    
    assert params_changed, "SC-VAE parameters did not change — online training failed"

    # ------------------------------------------------------------------
    # 2. Intrinsic reward was computed and is non-negative.
    # ------------------------------------------------------------------
    assert len(model._rollout_r_int) > 0, "No intrinsic reward logged"
    assert np.mean(model._rollout_r_int) >= 0.0, "Negative intrinsic reward"

    # ------------------------------------------------------------------
    # 3. Episodic size grows during a rollout (memory is accumulating).
    # ------------------------------------------------------------------
    assert len(model._rollout_epi_size) > 0, "Episodic size not logged"
    assert max(model._rollout_epi_size) > 0, "Episodic memory never grew"

    print("All assertions passed.")


test_ppogex_runs_one_iteration()