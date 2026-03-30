"""
Baseline: RecurrentPPO (LSTM) on MiniGrid-MemoryS7-v0.

Determines whether vanilla recurrent PPO can break the 0.5 reward ceiling.
If it can't, the problem is credit assignment, not our architecture.
"""
import argparse
import gymnasium
import minigrid
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

from envs.wrappers import MiniGridTrainingWrapper, MemorySignalVisibleWrapper


def make_env_fn(view_size: int = 5, max_steps: int = 200):
    def _fn():
        env = MiniGridTrainingWrapper(
            gymnasium.make("MiniGrid-MemoryS7-v0", agent_view_size=view_size, max_steps=max_steps)
        )
        env = MemorySignalVisibleWrapper(env)
        env = FlattenObservation(env)  # (5,5,4) -> (100,)
        return env
    return _fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=500_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tensorboard_log", type=str, default="runs/baseline_lstm")
    args = p.parse_args()

    vec_env = make_vec_env(make_env_fn(), n_envs=args.n_envs, seed=args.seed)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=f"{args.tensorboard_log}/checkpoints",
        name_prefix="model",
    )

    print(
        f"Baseline LSTM-PPO on MiniGrid-MemoryS7-v0\n"
        f"  {args.total_timesteps:,} steps, {args.n_envs} envs\n"
        f"  lr={args.lr} batch={args.batch_size} epochs={args.n_epochs}\n"
        f"  gamma={args.gamma} gae={args.gae_lambda} ent={args.ent_coef}\n"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=checkpoint_cb,
    )
    print("Done.")


if __name__ == "__main__":
    main()
