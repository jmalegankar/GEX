import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.vec_env import DummyVecEnv

from ppo_gex import PPOGEX
from geodesic_bonus import GeodesicExplorationBonus
from reward_normalizer import RunningMeanStd
from scvae_wrapper import SCVAEEncoderWrapper
from sc_vae import TransitionSCVAE
from config import SCVAEConfig
from obs_embeddings import ObservationEmbedding


def test_ppogex_runs_one_iteration():

    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    # ---- Fake minimal embedding for test ----
    class IdentityEmbedding(ObservationEmbedding):
        def __init__(self, obs_dim):
            self.out_channels = 1
            self.obs_dim = obs_dim

        def __call__(self, obs):
            obs = obs.float()
            return obs.unsqueeze(1)  # (B,1,D)
        

    obs_dim = env.observation_space.shape[0]

    embedding = IdentityEmbedding(obs_dim)
    cfg = SCVAEConfig(n_actions=env.action_space.n)

    scvae = TransitionSCVAE(
        embedding=embedding,
        cfg=cfg,
        sample_input_shape=(obs_dim,),
    )

    gex_modules = [
        GeodesicExplorationBonus(mu_dim=cfg.latent_dim)
    ]

    rms = RunningMeanStd()

    model = PPOGEX(
        "MlpPolicy",
        env,
        scvae=scvae,
        gex_modules=gex_modules,
        rms=rms,
        n_steps=32,
        batch_size=32,
        learning_rate=3e-4,
        verbose=0,
    )

    model.learn(total_timesteps=64)

    # Assert SCVAE was updated
    params = list(scvae.parameters())
    grads_exist = any(p.grad is not None for p in params)
    assert grads_exist

    # Assert intrinsic reward was nonzero at some point
    assert len(model._rollout_r_int) > 0
    assert np.mean(model._rollout_r_int) >= 0.0

test_ppogex_runs_one_iteration()