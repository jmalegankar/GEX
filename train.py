import argparse
import numpy as np
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from envs.wrappers import DoorButtonTrainingWrapper, MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig
from models.episodic_memory import BatchedNoveltyMemory
from models.vae import TransitionSCVAE
from models.wyner import WynerVAE, WynerConfig
from hswvime_ppo.hswvime_ppo import HSWVimePPO
from hswvime_ppo.policies import HSWVIMEActorCriticPolicy, HSWVIMEFeaturesExtractor

import minigrid


N_OBJECT_TYPES = 12
N_COLORS       = 6
N_STATES       = 3
N_DIRS         = 4


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class RenderCallback(BaseCallback):
    def __init__(self, render_env, render_freq: int):
        super().__init__()
        self.render_env  = render_env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            obs, _ = self.render_env.reset()
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.render_env.step(action)
                self.render_env.render()
                done = terminated or truncated
        return True

    def _on_training_end(self):
        self.render_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(env_name: str, view_size: int, env_kwargs: dict):
    if env_name == "door_button":
        def _fn():
            kwargs = dict(
                size=env_kwargs.get("size", 10),
                view_size=view_size,
                max_steps=env_kwargs.get("max_steps", 200),
            )
            if env_kwargs.get("render_mode") is not None:
                kwargs["render_mode"] = env_kwargs["render_mode"]
            return DoorButtonTrainingWrapper(**kwargs)
    else:
        def _fn():
            kw = dict(env_kwargs)
            kw["agent_view_size"] = view_size
            return MiniGridTrainingWrapper(gymnasium.make(env_name, **kw))
    return _fn


def build_embedding(obs_h, obs_w, embed_per_channel, dir_embed_dim):
    return CategoricalGridWithDirEmbedding(
        n_object_types=N_OBJECT_TYPES,
        n_colors=N_COLORS,
        n_states=N_STATES,
        obs_h=obs_h,
        obs_w=obs_w,
        embed_per_channel=embed_per_channel,
        n_dirs=N_DIRS,
        dir_embed_dim=dir_embed_dim,
    )


def build_scvae_cfg(act_dim, action_embed_dim, conv_channels, hidden_dim, latent_dim):
    return SCVAEConfig(
        act_dim=act_dim,
        action_embed_dim=action_embed_dim,
        conv_channels=conv_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )


def build_wyner_cfg(
    mu_dim: int,
    wyner_dim: int,
    context_dim: int,
    hidden_dim: int,
    lambda_past: float,
    lambda_future: float,
    alpha_intrinsic: float,
) -> WynerConfig:
    return WynerConfig(
        mu_dim=mu_dim,
        wyner_dim=wyner_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        lambda_past=lambda_past,
        lambda_future=lambda_future,
        alpha_intrinsic=alpha_intrinsic,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--env",        type=str,   default="MiniGrid-MemoryS7-v0")
    p.add_argument("--view_size",  type=int,   default=5)
    p.add_argument("--env_size",   type=int,   default=10)
    p.add_argument("--max_steps",  type=int,   default=200)
    p.add_argument("--render",     action="store_true")
    p.add_argument("--render_freq",type=int,   default=0)

    # Training
    p.add_argument("--total_timesteps", type=int,   default=500_000)
    p.add_argument("--n_envs",          type=int,   default=8)
    p.add_argument("--n_steps",         type=int,   default=1024)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--n_epochs",        type=int,   default=2)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--ent_coef",        type=float, default=0.02)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--device",          type=str,   default="auto")

    # Loss coefficients
    p.add_argument("--vae_recon_coef",   type=float, default=1.0)
    p.add_argument("--vae_kl_coef",      type=float, default=0.1)
    p.add_argument("--wyner_recon_coef", type=float, default=1.0)
    p.add_argument("--wyner_kl_coef",    type=float, default=0.1)
    p.add_argument("--intrinsic_scale",  type=float, default=1.0)

    # SCVAE architecture
    p.add_argument("--embed_per_channel", type=int, default=4)
    p.add_argument("--dir_embed_dim",     type=int, default=4)
    p.add_argument("--vae_latent_dim",    type=int, default=32)
    p.add_argument("--vae_hidden_dim",    type=int, default=256)
    p.add_argument("--vae_action_embed",  type=int, default=32)

    # WynerVAE architecture
    # wyner_dim    : dimension of the Wyner latent z  (common cause bottleneck)
    # context_dim  : GRU hidden state dimension (h_slow)
    # wyner_hidden : MLP hidden dim for Prior / Posterior / Decoders
    # lambda_past / lambda_future : reconstruction loss weights
    p.add_argument("--wyner_dim",         type=int,   default=64)
    p.add_argument("--context_dim",       type=int,   default=256)
    p.add_argument("--wyner_hidden",      type=int,   default=256)
    p.add_argument("--lambda_past",       type=float, default=1.0)
    p.add_argument("--lambda_future",     type=float, default=2.0)
    p.add_argument("--alpha_intrinsic",   type=float, default=1.0)

    # Logging
    p.add_argument("--tensorboard_log", type=str, default=None)
    p.add_argument("--verbose",         type=int, default=1)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Env ────────────────────────────────────────────────────────────────
    env_kwargs = {
        "max_steps":   args.max_steps,
        "render_mode": "human" if args.render else None,
    }
    if args.env == "door_button":
        env_kwargs["size"] = args.env_size

    env_fn  = make_env_fn(args.env, args.view_size, env_kwargs)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)

    render_callback = None
    if args.render_freq > 0:
        render_env      = make_env_fn(args.env, args.view_size, env_kwargs)()
        render_callback = RenderCallback(render_env, args.render_freq)

    # Probe dims
    _probe = env_fn()
    obs_h, obs_w, _ = _probe.observation_space.shape   # (H, W, 4)
    act_dim          = 1                                # Discrete → stored as (B, 1)
    _probe.close()

    # ── 2. Configs ────────────────────────────────────────────────────────────
    conv_channels = [32, 64, 128]

    scvae_cfg = build_scvae_cfg(
        act_dim          = act_dim,
        action_embed_dim = args.vae_action_embed,
        conv_channels    = conv_channels,
        hidden_dim       = args.vae_hidden_dim,
        latent_dim       = args.vae_latent_dim,
    )

    # WynerConfig is the single source of truth for all WynerVAE dimensions.
    # mu_dim must equal scvae_cfg.latent_dim — WynerVAE reads SCVAE's output.
    wyner_cfg = build_wyner_cfg(
        mu_dim          = args.vae_latent_dim,   # must match SCVAE latent_dim
        wyner_dim       = args.wyner_dim,
        context_dim     = args.context_dim,
        hidden_dim      = args.wyner_hidden,
        lambda_past     = args.lambda_past,
        lambda_future   = args.lambda_future,
        alpha_intrinsic = args.alpha_intrinsic,
    )

    # memory_shape = (context_dim,) — one h_slow vector per env per step.
    # This is stored in the rollout buffer and passed as h_prev to forward_train.
    memory_shape = (wyner_cfg.context_dim,)

    # ── 3. Policy kwargs ──────────────────────────────────────────────────────
    # features_extractor_kwargs controls HSWVIMEFeaturesExtractor:
    #   features = concat(sg(z_t), sg(mu_t))  →  dim = wyner_dim + mu_dim
    # net_arch must account for this combined dim being the MLP input.
    policy_kwargs = {
        "features_extractor_class":  HSWVIMEFeaturesExtractor,
        "features_extractor_kwargs": {
            "wyner_dim": wyner_cfg.wyner_dim,
            "mu_dim":    wyner_cfg.mu_dim,
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    }

    null_action = np.zeros(1, dtype=np.float32)

    # ── 4. Agent ──────────────────────────────────────────────────────────────
    embedding = build_embedding(obs_h, obs_w, args.embed_per_channel, args.dir_embed_dim)

    model = HSWVimePPO(
        policy      = HSWVIMEActorCriticPolicy,
        env         = vec_env,
        null_action = null_action,

        # PPO
        learning_rate      = args.lr,
        n_steps            = args.n_steps,
        batch_size         = args.batch_size,
        n_epochs           = args.n_epochs,
        gamma              = args.gamma,
        gae_lambda         = args.gae_lambda,
        ent_coef           = args.ent_coef,

        # Loss coefficients
        vae_recon_coef   = args.vae_recon_coef,
        vae_kl_coef      = args.vae_kl_coef,
        wyner_recon_coef = args.wyner_recon_coef,
        wyner_kl_coef    = args.wyner_kl_coef,
        intrinsic_scale  = args.intrinsic_scale,

        # Memory: shape must match wyner_cfg.context_dim
        memory_shape = memory_shape,

        # Policy
        policy_kwargs = policy_kwargs,

        # SCVAE: TransitionSCVAE takes (embedding, cfg) as kwargs
        vae_features_extractor_class  = TransitionSCVAE,
        vae_features_extractor_kwargs = {
            "embedding": embedding,
            "cfg":       scvae_cfg,
        },

        # WynerVAE: takes a single WynerConfig
        wyner_features_extractor_class  = WynerVAE,
        wyner_features_extractor_kwargs = {"cfg": wyner_cfg},

        # Episodic memory: SimHash over mu_t for novelty bonus
        episodic_memory_class  = BatchedNoveltyMemory,
        episodic_memory_kwargs = {
            "input_dim": wyner_cfg.mu_dim,
            "hash_dim":  8,
        },

        tensorboard_log = args.tensorboard_log,
        verbose         = args.verbose,
        seed            = args.seed,
        device          = args.device,
    )

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(
        f"\nTraining on '{args.env}' for {args.total_timesteps:,} steps "
        f"across {args.n_envs} envs on '{args.device}'\n"
        f"  SCVAE:  latent={args.vae_latent_dim}  hidden={args.vae_hidden_dim}\n"
        f"  Wyner:  z_dim={args.wyner_dim}  context={args.context_dim}"
        f"  hidden={args.wyner_hidden}\n"
        f"          λ_past={args.lambda_past}  λ_future={args.lambda_future}\n"
        f"  Coeffs: vae_recon={args.vae_recon_coef}  vae_kl={args.vae_kl_coef}"
        f"  wyner_recon={args.wyner_recon_coef}  wyner_kl={args.wyner_kl_coef}"
        f"  intrinsic={args.intrinsic_scale}\n"
    )

    model.learn(
        total_timesteps = args.total_timesteps,
        progress_bar    = True,
        callback        = render_callback,
    )
    print("Done.")


if __name__ == "__main__":
    main()