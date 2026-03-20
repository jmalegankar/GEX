import argparse
import os
import numpy as np
import gymnasium
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from envs.wrappers import DoorButtonTrainingWrapper, MiniGridTrainingWrapper
from models.embeddings import CategoricalGridWithDirEmbedding
from models.config import SCVAEConfig, GaussianVAEConfig
from models.vae import TransitionSCVAE, TransitionGaussianVAE
from hswvime_ppo.hswvime_ppo import HSWVimePPO
from hswvime_ppo.policies import HSWVIMEActorCriticPolicy, SimpleVAEFeaturesExtractor

import minigrid


# ── Callbacks ──────────────────────────────────────────────────────────

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


class EvalCallback(BaseCallback):
    """Runs deterministic eval episodes using the custom policy.predict(s_tm1, a_tm1, s_t)."""

    def __init__(self, eval_env_fn, null_action: np.ndarray, n_eval_episodes: int = 10, eval_freq: int = 10_000):
        super().__init__()
        self.eval_env_fn = eval_env_fn
        self.null_action = null_action
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self._eval_env = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        if self._eval_env is None:
            self._eval_env = self.eval_env_fn()

        policy = self.model.policy

        rewards, successes, lengths = [], [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self._eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0
            prev_obs = obs.copy()
            prev_action = self.null_action.copy()
            while not done:
                action = policy.predict(prev_obs, prev_action, obs, deterministic=True)
                new_obs, reward, terminated, truncated, info = self._eval_env.step(action)
                ep_reward += reward
                ep_len += 1
                prev_obs = obs
                prev_action = np.array([action], dtype=np.float32).reshape(self.null_action.shape)
                obs = new_obs
                done = terminated or truncated
            rewards.append(ep_reward)
            lengths.append(ep_len)
            successes.append(float(info.get("success", ep_reward > 0)))

        self.logger.record("eval/mean_reward", np.mean(rewards))
        self.logger.record("eval/mean_ep_length", np.mean(lengths))
        self.logger.record("eval/success_rate", np.mean(successes))
        return True

    def _on_training_end(self):
        if self._eval_env is not None:
            self._eval_env.close()


# ── Environment helpers ───────────────────────────────────────────────

# Vocabulary sizes shared by both multigrid and minigrid
N_OBJECT_TYPES = 12
N_COLORS       = 6
N_STATES       = 3
N_DIRS         = 4

# Predefined environment suite for experiments
ENV_SUITE = {
    "door_button":    {"desc": "DoorButton 10x10 (custom)", "max_steps": 200},
    "keycorridor":    {"id": "MiniGrid-KeyCorridorS3R3-v0", "max_steps": 400},
    "obstructed":     {"id": "MiniGrid-ObstructedMaze-1Dl-v0", "max_steps": 400},
    "multiroom":      {"id": "MiniGrid-MultiRoom-N4-S5-v0", "max_steps": 400},
    "doorkey":        {"id": "MiniGrid-DoorKey-8x8-v0", "max_steps": 300},
    "multiroom_hard": {"id": "MiniGrid-MultiRoom-N6-v0", "max_steps": 600},
}


def resolve_env_name(env_name: str) -> str:
    """Map short alias to MiniGrid gym id, or pass through as-is."""
    if env_name in ENV_SUITE and "id" in ENV_SUITE[env_name]:
        return ENV_SUITE[env_name]["id"]
    return env_name


def make_env_fn(env_name: str, view_size: int, env_kwargs: dict):
    """Returns a callable that constructs one instance of the requested env."""
    if env_name == "door_button":
        def _fn():
            kwargs = dict(size=env_kwargs.get("size", 10), view_size=view_size, max_steps=env_kwargs.get("max_steps", 200))
            render_mode = env_kwargs.get("render_mode", None)
            if render_mode is not None:
                kwargs["render_mode"] = render_mode
            return DoorButtonTrainingWrapper(**kwargs)
    else:
        gym_id = resolve_env_name(env_name)
        def _fn():
            kw = dict(env_kwargs)
            kw['agent_view_size'] = view_size
            return MiniGridTrainingWrapper(gymnasium.make(gym_id, **kw))
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


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train spCauchy PPO on a grid environment.")

    # Environment
    env_choices_help = ", ".join(f'"{k}"' for k in ENV_SUITE)
    p.add_argument("--env",       type=str, default="door_button",
                   help=f'Environment: {env_choices_help}, or any MiniGrid gym id.')
    p.add_argument("--view_size", type=int, default=5,
                   help="Agent view size.")
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
    p.add_argument("--vae_fwd_coef",    type=float, default=1.0)
    p.add_argument("--kl_use_schedule", action="store_true",
                   help="Enable KL coefficient annealing from 0 to target over kl_anneal_steps.")
    p.add_argument("--kl_anneal_steps", type=int, default=50_000,
                   help="Number of timesteps to anneal KL coefficient from 0 to target.")

    # Model architecture
    p.add_argument("--latent_type",       type=str, default="spcauchy",
                   choices=["spcauchy", "gaussian"],
                   help="Latent space type: spcauchy (Spherical Cauchy) or gaussian.")
    p.add_argument("--embed_per_channel", type=int, default=4)
    p.add_argument("--dir_embed_dim",     type=int, default=4)
    p.add_argument("--vae_latent_dim",    type=int, default=32)
    p.add_argument("--vae_hidden_dim",    type=int, default=256)
    p.add_argument("--vae_action_embed",  type=int, default=32)
    p.add_argument("--gru_hidden_dim",    type=int, default=0,
                   help="GRU memory hidden dim. 0 = no GRU (features = mu only).")
    p.add_argument("--vae_lr",            type=float, default=1e-3,
                   help="Learning rate for VAE + forward predictor optimizer.")
    p.add_argument("--normalize_intrinsic", action="store_true",
                   help="Normalize intrinsic rewards with running mean/std.")

    # Checkpointing
    p.add_argument("--checkpoint_freq", type=int, default=50_000,
                   help="Save model checkpoint every N timesteps (0 = disabled).")
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints",
                   help="Directory for model checkpoints.")
    p.add_argument("--no_save_final",    action="store_true",
                   help="Disable saving final model after training.")

    # Evaluation
    p.add_argument("--eval_freq",       type=int, default=10_000,
                   help="Run deterministic eval every N timesteps (0 = disabled).")
    p.add_argument("--n_eval_episodes", type=int, default=10,
                   help="Number of eval episodes per evaluation.")

    # Logging
    p.add_argument("--tensorboard_log", type=str, default=None)
    p.add_argument("--wandb",           action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project",   type=str, default="spcauchy-exploration",
                   help="WandB project name.")
    p.add_argument("--wandb_entity",    type=str, default=None,
                   help="WandB entity (team/user).")
    p.add_argument("--run_name",        type=str, default=None,
                   help="Run name for WandB and checkpoint dirs.")
    p.add_argument("--verbose",         type=int, default=1)

    # Rendering
    p.add_argument("--render_freq", type=int, default=0,
                   help="Run a rendered eval episode every N training steps (0 = disabled).")

    return p.parse_args()


def _make_run_name(args) -> str:
    """Generate a descriptive run name if not specified."""
    if args.run_name:
        return args.run_name
    parts = [args.env, args.latent_type, f"z{args.vae_latent_dim}"]
    if args.gru_hidden_dim > 0:
        parts.append(f"gru{args.gru_hidden_dim}")
    parts.append(f"s{args.seed}")
    return "_".join(parts)


def main():
    args = parse_args()
    run_name = _make_run_name(args)

    # Auto-set max_steps from suite if not explicitly provided and env is in suite
    if args.env in ENV_SUITE and args.max_steps == 200:
        suite_max = ENV_SUITE[args.env].get("max_steps", 200)
        if suite_max != 200:
            args.max_steps = suite_max

    # ── WandB setup ──
    if args.wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                sync_tensorboard=True,
            )
        except ImportError:
            print("WARNING: wandb not installed. Install with: pip install wandb")
            args.wandb = False

    # ── Tensorboard log dir ──
    if args.tensorboard_log is None:
        args.tensorboard_log = os.path.join("runs", run_name)

    # 1. Build env factory
    env_kwargs = {"max_steps": args.max_steps, "render_mode": "human" if args.render else None}
    if args.env == "door_button":
        env_kwargs["size"] = args.env_size

    env_fn = make_env_fn(args.env, args.view_size, env_kwargs)

    # Probe one env to get obs/action dims
    _probe = env_fn()
    obs_h, obs_w, _ = _probe.observation_space.shape   # (H, W, 4)
    act_dim = 1  # Discrete actions stored as raw indices (B, 1)
    _probe.close()

    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed)

    # 2. Build models
    embedding = build_embedding(obs_h, obs_w, args.embed_per_channel, args.dir_embed_dim)
    conv_channels = [32, 64, 128]

    # 3. Policy kwargs
    policy_kwargs = {
        "features_extractor_class":  SimpleVAEFeaturesExtractor,
        "features_extractor_kwargs": {
            "mu_dim": args.vae_latent_dim,
            "gru_hidden_dim": args.gru_hidden_dim,
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
    }

    null_action = np.zeros(1, dtype=np.float32)

    # 4. Select VAE class and config based on latent type
    if args.latent_type == "gaussian":
        vae_class = TransitionGaussianVAE
        vae_cfg = GaussianVAEConfig(
            act_dim=act_dim,
            action_embed_dim=args.vae_action_embed,
            conv_channels=conv_channels,
            hidden_dim=args.vae_hidden_dim,
            latent_dim=args.vae_latent_dim,
        )
    else:
        vae_class = TransitionSCVAE
        vae_cfg = SCVAEConfig(
            act_dim=act_dim,
            action_embed_dim=args.vae_action_embed,
            conv_channels=conv_channels,
            hidden_dim=args.vae_hidden_dim,
            latent_dim=args.vae_latent_dim,
        )

    # 5. Instantiate agent
    model = HSWVimePPO(
        policy=HSWVIMEActorCriticPolicy,
        env=vec_env,
        null_action=null_action,
        learning_rate=args.lr,
        vae_lr=args.vae_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vae_recon_coef=args.vae_recon_coef,
        vae_kl_coef=args.vae_kl_coef,
        vae_fwd_coef=args.vae_fwd_coef,
        kl_use_schedule=args.kl_use_schedule,
        kl_anneal_steps=args.kl_anneal_steps,
        normalize_intrinsic=args.normalize_intrinsic,
        gru_hidden_dim=args.gru_hidden_dim,
        policy_kwargs=policy_kwargs,
        vae_features_extractor_class=vae_class,
        vae_features_extractor_kwargs={
            "embedding": embedding,
            "cfg": vae_cfg,
        },
        tensorboard_log=args.tensorboard_log,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
    )

    # 6. Build callbacks
    callbacks = []

    # Checkpointing
    if args.checkpoint_freq > 0:
        ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=ckpt_dir,
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ))

    # Evaluation
    if args.eval_freq > 0:
        eval_env_kwargs = {"max_steps": args.max_steps}
        if args.env == "door_button":
            eval_env_kwargs["size"] = args.env_size
        eval_env_fn = make_env_fn(args.env, args.view_size, eval_env_kwargs)
        callbacks.append(EvalCallback(
            eval_env_fn=eval_env_fn,
            null_action=null_action,
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=args.eval_freq,
        ))

    # Rendering
    if args.render_freq > 0:
        render_kwargs = dict(env_kwargs)
        render_kwargs["render_mode"] = "human"
        render_env = make_env_fn(args.env, args.view_size, render_kwargs)()
        callbacks.append(RenderCallback(render_env, args.render_freq))

    # WandB
    if args.wandb:
        callbacks.append(WandbCallback(
            verbose=args.verbose,
        ))

    callback = CallbackList(callbacks) if callbacks else None

    # 7. Train
    print(
        f"Training on '{args.env}' for {args.total_timesteps:,} timesteps "
        f"with {args.n_envs} envs on device '{args.device}'.\n"
        f"  env_size={args.env_size}  view_size={args.view_size}  max_steps={args.max_steps}\n"
        f"  lr={args.lr}  n_steps={args.n_steps}  batch={args.batch_size}  epochs={args.n_epochs}\n"
        f"  gamma={args.gamma}  gae={args.gae_lambda}  ent={args.ent_coef}  seed={args.seed}\n"
        f"  latent_type={args.latent_type}  vae_latent={args.vae_latent_dim}\n"
        f"  vae_recon={args.vae_recon_coef}  vae_kl={args.vae_kl_coef}  vae_fwd={args.vae_fwd_coef}\n"
        f"  vae_lr={args.vae_lr}  gru_hidden_dim={args.gru_hidden_dim}  normalize_intrinsic={args.normalize_intrinsic}\n"
        f"  run_name={run_name}"
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback)

    # 8. Save final model
    if not args.no_save_final:
        final_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "final_model")
        model.save(final_path)
        print(f"Final model saved to {final_path}")

    if args.wandb:
        import wandb
        wandb.finish()

    print("Done.")


if __name__ == "__main__":
    main()
