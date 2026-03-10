import argparse
import numpy as np
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from envs.wrappers import DoorButtonTrainingWrapper, MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig
from models.vae import TransitionSCVAE
from models.wyner import WynerVAE, WynerIndependentVAE
from hswvime_ppo.hswvime_ppo import HSWVimePPO
from hswvime_ppo.policies import HSWVIMEActorCriticPolicy, HSWVIMEFeaturesExtractor

import minigrid


class RenderCallback(BaseCallback):
    """Runs one greedy episode in a separate render env every `render_freq` steps."""

    def __init__(self, render_env, render_freq: int):
        super().__init__()
        self.render_env = render_env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            obs, _ = self.render_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.render_env.step(action)
                self.render_env.render()
                done = terminated or truncated
        return True

    def _on_training_end(self):
        self.render_env.close()

#Vocabulary sizes shared by both multigrid and minigrid
N_OBJECT_TYPES = 12  
N_COLORS       = 6   
N_STATES       = 3   
N_DIRS         = 4   


def make_env_fn(env_name: str, view_size: int, env_kwargs: dict, render_mode: str = None):
    """Returns a callable that constructs one instance of the requested env."""
    if env_name == "door_button":
        def _fn():
            kwargs = dict(size=env_kwargs.get("size", 10), view_size=view_size, max_steps=env_kwargs.get("max_steps", 200))
            if render_mode is not None:
                kwargs["render_mode"] = render_mode
            return DoorButtonTrainingWrapper(**kwargs)
    else:
        def _fn():
            return MiniGridTrainingWrapper(gymnasium.make(env_name, render_mode=render_mode, **env_kwargs))
    return _fn


def build_embedding(obs_h: int, obs_w: int, embed_per_channel: int, dir_embed_dim: int):
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


def build_vae(embedding, act_dim: int, latent_dim: int, conv_channels, hidden_dim: int, action_embed_dim: int):
    cfg = SCVAEConfig(
        act_dim=act_dim,
        action_embed_dim=action_embed_dim,
        conv_channels=conv_channels,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    return TransitionSCVAE(embedding, cfg)


def build_wyner(recon_dim: int, mu_dim: int, latent_dim: int, decode_hidden: int):
    return WynerVAE(
        recon_dim=recon_dim,
        mu_dim=mu_dim,
        latent_dim=latent_dim,
        latent_tokens=1,
        decode_hidden=decode_hidden,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train HSWVimePPO on a grid environment.")

    # Environment
    p.add_argument("--env",       type=str, default="door_button",
                   help='Environment name: "door_button" or a MiniGrid gym id.')
    p.add_argument("--view_size", type=int, default=5,
                   help="Agent view size (only used for door_button).")
    p.add_argument("--env_size",  type=int, default=10,
                   help="Grid size (only used for door_button).")
    p.add_argument("--max_steps", type=int, default=200,
                   help="Max steps per episode.")
    
    p.add_argument("--render", action="store_true", help="Render the environment during training.")

    # Training 
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--n_envs",          type=int, default=4)
    p.add_argument("--n_steps",         type=int, default=512,
                   help="Steps per env per rollout (total = n_steps * n_envs).")
    p.add_argument("--batch_size",      type=int, default=256)
    p.add_argument("--n_epochs",        type=int, default=4)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--ent_coef",        type=float, default=0.01)
    p.add_argument("--seed",            type=int,   default=0)
    p.add_argument("--device",          type=str,   default="auto")

    # Loss coefficients 
    p.add_argument("--vae_recon_coef",  type=float, default=1.0)
    p.add_argument("--vae_kl_coef",     type=float, default=0.01)
    p.add_argument("--wyner_recon_coef",type=float, default=1.0)
    p.add_argument("--wyner_kl_coef",   type=float, default=0.01)
    p.add_argument("--intrinsic_scale", type=float, default=1.0)

    # Model architecture 
    p.add_argument("--embed_per_channel", type=int, default=4)
    p.add_argument("--dir_embed_dim",     type=int, default=4)
    p.add_argument("--vae_latent_dim",    type=int, default=32)
    p.add_argument("--vae_hidden_dim",    type=int, default=256)
    p.add_argument("--vae_action_embed",  type=int, default=32)
    p.add_argument("--wyner_latent_dim",  type=int, default=64)
    p.add_argument("--wyner_decode_hidden", type=int, default=128)

    # Logging
    p.add_argument("--tensorboard_log", type=str, default=None)
    p.add_argument("--verbose",         type=int, default=1)

    # Rendering
    p.add_argument("--render_freq", type=int, default=0,
                   help="Run a rendered eval episode every N training steps (0 = disabled).")

    return p.parse_args()


def main():
    args = parse_args()

    # 1. Build env factory
    env_kwargs = {"max_steps": args.max_steps, "render_mode": "human" if args.render else None}
    if args.env == "door_button":
        env_kwargs["size"] = args.env_size

    env_fn = make_env_fn(args.env, args.view_size, env_kwargs)

    # Optional: separate env for rendering
    render_callback = None
    if args.render_freq > 0:
        render_env = make_env_fn(args.env, args.view_size, env_kwargs, render_mode="human")()
        render_callback = RenderCallback(render_env, args.render_freq)

    # Probe one env to get obs/action dims
    _probe = env_fn()
    obs_h, obs_w, _ = _probe.observation_space.shape   # (H, W, 4)
    # Discrete actions are stored as raw indices (B, 1), not one-hot.
    act_dim = 1
    _probe.close()

    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)

    # ── 2. Build models ───────────────────────────────────────────────────────
    embedding = build_embedding(obs_h, obs_w, args.embed_per_channel, args.dir_embed_dim)

    conv_channels = [32, 64, 128]
    vae = build_vae(
        embedding,
        act_dim=act_dim,
        latent_dim=args.vae_latent_dim,
        conv_channels=conv_channels,
        hidden_dim=args.vae_hidden_dim,
        action_embed_dim=args.vae_action_embed,
    )

    # recon_dim = 2 * conv_out + action_embed
    recon_dim = 2 * conv_channels[-1] + args.vae_action_embed

    wyner = build_wyner(
        recon_dim=recon_dim,
        mu_dim=args.vae_latent_dim,
        latent_dim=args.wyner_latent_dim,
        decode_hidden=args.wyner_decode_hidden,
    )

    memory_shape = (1, args.wyner_latent_dim)

    # 3. Policy kwargs
    policy_kwargs = {
        "features_extractor_class":  HSWVIMEFeaturesExtractor,
        "features_extractor_kwargs": {
            "wyner_dim": args.wyner_latent_dim,
            "mu_dim":    args.vae_latent_dim,
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    }

    # null_action is passed to the GRU on the very first step (before any real action)
    null_action = np.zeros(1, dtype=np.float32)

    # 4. Instantiate agent
    model = HSWVimePPO(
        policy=HSWVIMEActorCriticPolicy,
        env=vec_env,
        null_action=null_action,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vae_recon_coef=args.vae_recon_coef,
        vae_kl_coef=args.vae_kl_coef,
        wyner_recon_coef=args.wyner_recon_coef,
        wyner_kl_coef=args.wyner_kl_coef,
        intrinsic_scale=args.intrinsic_scale,
        memory_shape=memory_shape,
        policy_kwargs=policy_kwargs,
        vae_features_extractor_class=TransitionSCVAE,
        vae_features_extractor_kwargs={
            "embedding": embedding,
            "cfg": SCVAEConfig(
                act_dim=act_dim,
                action_embed_dim=args.vae_action_embed,
                conv_channels=conv_channels,
                hidden_dim=args.vae_hidden_dim,
                latent_dim=args.vae_latent_dim,
            ),
        },
        wyner_features_extractor_class=WynerVAE,
        wyner_features_extractor_kwargs={
            "recon_dim":      recon_dim,
            "mu_dim":         args.vae_latent_dim,
            "latent_dim":     args.wyner_latent_dim,
            "latent_tokens":  1,
            "decode_hidden":  args.wyner_decode_hidden,
        },
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
    )

    # 5. Train
    print(
        f"Training on '{args.env}' for {args.total_timesteps:,} timesteps "
        f"with {args.n_envs} envs on device '{args.device}'.\n"
        f"  env_size={args.env_size}  view_size={args.view_size}  max_steps={args.max_steps}\n"
        f"  lr={args.lr}  n_steps={args.n_steps}  batch={args.batch_size}  epochs={args.n_epochs}\n"
        f"  gamma={args.gamma}  gae={args.gae_lambda}  ent={args.ent_coef}  seed={args.seed}\n"
        f"  vae_latent={args.vae_latent_dim}  wyner_latent={args.wyner_latent_dim}\n"
        f"  vae_recon={args.vae_recon_coef}  vae_kl={args.vae_kl_coef}"
        f"  wyner_recon={args.wyner_recon_coef}  wyner_kl={args.wyner_kl_coef}"
        f"  intrinsic={args.intrinsic_scale}"
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=render_callback)

    print("Done.")


if __name__ == "__main__":
    main()
