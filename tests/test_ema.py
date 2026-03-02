import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3.ppo_gex import PPOGEX
from models.sc_vae import TransitionSCVAE
from models.config import SCVAEConfig
from intrinsic_reward.geodesic_bonus import GeodesicExplorationBonus

# Minimal embedding for vector obs
from models.obs_embeddings import ObservationEmbedding

class IdentityEmbedding(ObservationEmbedding):
    def __init__(self, obs_dim: int):
        super().__init__()
        self._obs_dim = obs_dim

    @property
    def out_channels(self) -> int:
        return 1

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # (B,D) -> (B,1,D,1)
        return obs.float().unsqueeze(1).unsqueeze(-1)


def test_ema_target_is_frozen_and_updates():
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    obs_dim = env.observation_space.shape[0]

    cfg = SCVAEConfig(n_actions=env.action_space.n)
    scvae = TransitionSCVAE(
        embedding=IdentityEmbedding(obs_dim),
        cfg=cfg,
        sample_input_shape=(obs_dim,),
    )

    model = PPOGEX(
        "MlpPolicy",
        env,
        sc_vae=scvae,
        gex_modules=[GeodesicExplorationBonus(mu_dim=cfg.latent_dim)],
        rms=None,
        n_steps=32,
        batch_size=32,
        learning_rate=3e-4,
        tau=0.1,   # big tau so change is obvious
        verbose=0,
    )

    # target params require_grad False
    assert model.sc_vae_target is not None
    assert all(not p.requires_grad for p in model.sc_vae_target.parameters())
    

    # snapshot target params
    before = [p.detach().clone() for p in model.sc_vae_target.parameters()]

    model.learn(total_timesteps=64)

    after = [p.detach().clone() for p in model.sc_vae_target.parameters()]
    changed = any(not th.allclose(b, a) for b, a in zip(before, after))
    assert changed, "EMA target did not change after learning"

test_ema_target_is_frozen_and_updates()