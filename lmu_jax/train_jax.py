"""
JAX/Flax LMU-PPO training on MiniGrid-Memory*.
Phase 0a — first milestone: reproduce ≥ 88% on MemoryS7.

Run:
    python train_jax.py --env MemoryS7 --seed 0
    python train_jax.py --env MemoryS11 --seed 0 --n_steps 1024 --n_envs 16

Differences vs train.py (PyTorch):
  - No VecTransposeImage — encoder handles NHWC image directly.
  - SubprocVecEnv still used for env parallelism (envs are gymnasium, not JAX).
  - Policy forward + PPO update are jax.jit-compiled.
  - GAE is a plain numpy loop (tiny relative to model compute).
  - LMU state (h, m) are jnp arrays zeroed on episode boundaries via jnp.where.
"""

import argparse
import os
import time
from typing import Dict, NamedTuple, Tuple

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401
from gymnasium.wrappers import FilterObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter   # reuse existing TB install

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from policies_jax import LMUActorCriticPolicy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_IDS = {
    "MemoryS7":  "MiniGrid-MemoryS7-v0",
    "MemoryS9":  "MiniGrid-MemoryS9-v0",
    "MemoryS11": "MiniGrid-MemoryS11-v0",
    "MemoryS13": "MiniGrid-MemoryS13-v0",
}

# Approx episode length → LMU theta (window covers full memory horizon)
THETA = {
    "MemoryS7":  50.0,
    "MemoryS9":  75.0,
    "MemoryS11": 100.0,
    "MemoryS13": 150.0,
}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int, rank: int = 0):
    """No VecTransposeImage — image arrives as NHWC (B, H, W, 3)."""
    def _init():
        env = gym.make(env_id)
        env = FilterObservation(env, filter_keys=["image", "direction"])
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def np_to_jax(obs: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.array(v) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation  (numpy, runs between rollout and update)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards:    np.ndarray,   # (T, n_envs)
    values:     np.ndarray,   # (T, n_envs)
    dones:      np.ndarray,   # (T, n_envs) float32  — 1.0 if episode ended
    last_value: np.ndarray,   # (n_envs,)   — V(s_{T}) bootstrap
    gamma:      float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_val    = values[t + 1] if t < T - 1 else last_value
        nonterminal = 1.0 - dones[t]
        delta       = rewards[t] + gamma * next_val * nonterminal - values[t]
        gae         = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values   # (advantages, returns)


# ---------------------------------------------------------------------------
# Minibatch container
# ---------------------------------------------------------------------------

class Batch(NamedTuple):
    obs_image:    jnp.ndarray   # (B, H, W, 3)  uint8 cast to int32 in encoder
    obs_dir:      jnp.ndarray   # (B,)
    actions:      jnp.ndarray   # (B,) int32
    old_log_prob: jnp.ndarray   # (B,)
    old_values:   jnp.ndarray   # (B,)
    advantages:   jnp.ndarray   # (B,)
    returns:      jnp.ndarray   # (B,)
    lmu_h:        jnp.ndarray   # (B, hidden_size)
    lmu_m:        jnp.ndarray   # (B, memory_size)


# ---------------------------------------------------------------------------
# JIT-compiled functions (built once inside train(), closed over `policy`)
# ---------------------------------------------------------------------------

def build_jit_fns(policy, vf_coef: float, ent_coef: float):

    @jax.jit
    def act_fn(params, obs_jax, h, m):
        return policy.apply(params, obs_jax, h, m, method=policy.act)

    @jax.jit
    def update_fn(train_state, batch, clip_range: float):
        def loss_fn(params):
            obs = {"image": batch.obs_image, "direction": batch.obs_dir}
            value, log_prob, entropy = policy.apply(
                params, obs, batch.lmu_h, batch.lmu_m, batch.actions
            )

            # Use pre-normalized advantages (DO NOT renormalize here)
            adv = batch.advantages

            # PPO clipped surrogate
            ratio = jnp.exp(log_prob - batch.old_log_prob)
            pg_loss = -jnp.minimum(
                adv * ratio,
                adv * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range),
            ).mean()

            # ---------------- VALUE CLIPPING FIX ----------------
            value_pred_clipped = batch.old_values + jnp.clip(
                value - batch.old_values,
                -clip_range,
                clip_range
            )

            v_loss_unclipped = (value - batch.returns) ** 2
            v_loss_clipped   = (value_pred_clipped - batch.returns) ** 2

            v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

            # Entropy bonus
            ent_loss = -entropy.mean()

            total = pg_loss + vf_coef * v_loss + ent_coef * ent_loss

            # ---------------- KL FIX ----------------
            approx_kl = jnp.mean(batch.old_log_prob - log_prob)

            clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_range).astype(jnp.float32))

            expl_var = 1.0 - jnp.var(batch.returns - value) / (
                jnp.var(batch.returns) + 1e-8
            )

            return total, (pg_loss, v_loss, ent_loss, approx_kl,
                           clip_frac, expl_var, entropy.mean())

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(train_state.params)

        grad_norm = optax.global_norm(grads)
        # W_m gradient: tells us if TBPTT-1 is reaching the memory readout layer
        wm_grad_norm = optax.global_norm(
            grads["params"]["lmu_cell"]["W_m"]["kernel"]
        )
        train_state = train_state.apply_gradients(grads=grads)

        pg_loss, v_loss, ent_loss, approx_kl, clip_frac, expl_var, entropy = aux

        metrics = {
            "loss/total": loss,
            "loss/policy": pg_loss,
            "loss/value": v_loss,
            "loss/entropy": ent_loss,
            "train/approx_kl": approx_kl,
            "train/clip_frac": clip_frac,
            "train/expl_var": expl_var,
            "train/entropy": entropy,
            "train/grad_norm": grad_norm,
            "debug/wm_grad":   wm_grad_norm,  # must be > 0 for memory readout to learn
        }

        return train_state, metrics
    return act_fn, update_fn


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(
    env,
    params,
    act_fn,
    h:          jnp.ndarray,
    m:          jnp.ndarray,
    n_steps:    int,
    n_envs:     int,
    key:        jax.Array,
    last_obs:   Dict[str, np.ndarray],
    last_dones: np.ndarray,
    gamma:      float,
):
    """
    Collect n_steps of experience.  Returns numpy rollout arrays + updated
    (h, m, key, obs, dones) for the next call.
    """
    H, W, C = last_obs["image"].shape[1:]

    buf_image   = np.empty((n_steps, n_envs, H, W, C), dtype=np.uint8)
    buf_dir     = np.empty((n_steps, n_envs),           dtype=np.int64)
    buf_actions = np.empty((n_steps, n_envs),           dtype=np.int64)
    buf_rewards = np.empty((n_steps, n_envs),           dtype=np.float32)
    buf_dones   = np.empty((n_steps, n_envs),           dtype=np.float32)
    buf_values  = np.empty((n_steps, n_envs),           dtype=np.float32)
    buf_logprob = np.empty((n_steps, n_envs),           dtype=np.float32)
    buf_h       = np.empty((n_steps, n_envs, h.shape[-1]), dtype=np.float32)
    buf_m       = np.empty((n_steps, n_envs, m.shape[-1]), dtype=np.float32)

    obs   = last_obs
    dones = last_dones

    for t in range(n_steps):
        obs_jax = np_to_jax(obs)

        # Store states BEFORE this step (needed for gradient re-run at update time)
        buf_h[t] = np.array(h)
        buf_m[t] = np.array(m)

        logits, value, h_new, m_new = act_fn(params, obs_jax, h, m)

        # Sample action
        key, subkey = jax.random.split(key)
        action   = jax.random.categorical(subkey, logits)           # (n_envs,)
        log_prob = jax.nn.log_softmax(logits)[jnp.arange(n_envs), action]

        action_np = np.array(action)
        new_obs, reward, done, infos = env.step(action_np)

        # Timeout bootstrapping: if episode truncated (not terminated),
        # add gamma * V(terminal_obs) so the agent isn't penalised for the cutoff.
        for idx, (d, info) in enumerate(zip(done, infos)):
            if d and info.get("TimeLimit.truncated", False):
                term = info["terminal_observation"]
                term_jax = {k: jnp.array(np.array(v)[None]) for k, v in term.items()}
                _, term_val, _, _ = act_fn(params, term_jax, h_new[idx:idx+1], m_new[idx:idx+1])
                reward[idx] += gamma * float(term_val[0])

        buf_image[t]   = obs["image"]
        buf_dir[t]     = obs["direction"]
        buf_actions[t] = action_np
        buf_values[t]  = np.array(value)
        buf_logprob[t] = np.array(log_prob)
        buf_rewards[t] = reward
        buf_dones[t]   = done.astype(np.float32)

        # Advance state; zero out finished episodes
        dones_mask = jnp.array(done, dtype=jnp.bool_)
        h = jnp.where(dones_mask[:, None], 0.0, h_new)
        m = jnp.where(dones_mask[:, None], 0.0, m_new)

        obs   = new_obs
        dones = done

    return (
        dict(image=buf_image, dir=buf_dir, actions=buf_actions,
             rewards=buf_rewards, dones=buf_dones, values=buf_values,
             logprob=buf_logprob, h=buf_h, m=buf_m),
        h, m, key, obs, dones,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(eval_env, params, act_fn, hidden_size, memory_size,
             n_episodes=20) -> float:
    """Run n_episodes deterministically; return mean reward."""
    rewards = []
    for _ in range(n_episodes):
        obs   = eval_env.reset()
        done  = np.array([False])
        h     = jnp.zeros((1, hidden_size))
        m     = jnp.zeros((1, memory_size))
        total = 0.0
        key   = jax.random.PRNGKey(0)
        while not done[0]:
            obs_jax = np_to_jax(obs)
            logits, _, h_new, m_new = act_fn(params, obs_jax, h, m)
            action = jnp.argmax(logits, axis=-1)   # deterministic
            obs, r, done, _ = eval_env.step(np.array(action))
            total += float(r[0])
            dones_mask = jnp.array(done, dtype=jnp.bool_)
            h = jnp.where(dones_mask[:, None], 0.0, h_new)
            m = jnp.where(dones_mask[:, None], 0.0, m_new)
        rewards.append(total)
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    env_id = ENV_IDS[args.env]
    theta  = THETA[args.env]
    n_envs = args.n_envs

    # ---- Environments -------------------------------------------------------
    train_env = SubprocVecEnv([make_env(env_id, args.seed, i) for i in range(n_envs)])
    eval_env  = DummyVecEnv([make_env(env_id, args.seed + 1000)])

    # Infer image shape from env (works for any MemoryS* size)
    img_shape = train_env.observation_space["image"].shape  # (H, W, 3)

    # ---- Policy & params ----------------------------------------------------
    policy = LMUActorCriticPolicy(
        n_actions   = train_env.action_space.n,
        encoder_dim = args.encoder_dim,
        hidden_size = args.hidden_size,
        memory_size = args.memory_size,
        theta       = theta,
    )

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    H, W, C  = img_shape
    dummy_obs = {
        "image":     jnp.zeros((1, H, W, C), dtype=jnp.int32),
        "direction": jnp.zeros((1,), dtype=jnp.int32),
    }
    dummy_h   = jnp.zeros((1, args.hidden_size))
    dummy_m   = jnp.zeros((1, args.memory_size))
    dummy_act = jnp.zeros((1,), dtype=jnp.int32)

    # init via __call__ (= evaluate_actions) so all submodules are initialised
    params = policy.init({"params": init_key}, dummy_obs, dummy_h, dummy_m, dummy_act)

    # ---- Optimiser ----------------------------------------------------------
    # Gradient clip then Adam — clip is baked into the optax chain so
    # apply_gradients handles it; we log the pre-clip norm separately.
    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=policy.apply, params=params, tx=tx)

    # ---- JIT functions ------------------------------------------------------
    act_fn, update_fn = build_jit_fns(policy, args.vf_coef, args.ent_coef)

    # Warm up JIT (avoids counting compile time in the first update)
    dummy_batch = Batch(
        obs_image    = jnp.zeros((args.batch_size, H, W, C), dtype=jnp.int32),
        obs_dir      = jnp.zeros((args.batch_size,), dtype=jnp.int32),
        actions      = jnp.zeros((args.batch_size,), dtype=jnp.int32),
        old_log_prob = jnp.zeros((args.batch_size,)),
        old_values   = jnp.zeros((args.batch_size,)),
        advantages   = jnp.zeros((args.batch_size,)),
        returns      = jnp.zeros((args.batch_size,)),
        lmu_h        = jnp.zeros((args.batch_size, args.hidden_size)),
        lmu_m        = jnp.zeros((args.batch_size, args.memory_size)),
    )
    _ = act_fn(train_state.params, dummy_obs, dummy_h, dummy_m)
    _ = update_fn(train_state, dummy_batch, args.clip_range)
    print("JIT compilation done.\n")

    # ---- Logging ------------------------------------------------------------
    writer = SummaryWriter(log_dir=f"{args.tb_log}/{args.env}_s{args.seed}")

    # ---- Initial env state --------------------------------------------------
    h = jnp.zeros((n_envs, args.hidden_size))
    m = jnp.zeros((n_envs, args.memory_size))
    last_obs   = train_env.reset()
    last_dones = np.zeros(n_envs, dtype=bool)

    total_steps  = 0
    n_updates    = 0
    steps_per_update = n_envs * args.n_steps
    total_updates    = args.total_steps // steps_per_update
    eval_every        = max(args.eval_freq // steps_per_update, 1)

    print(f"Training LMU-PPO (JAX) on {env_id}  |  seed={args.seed}")
    print(f"  hidden={args.hidden_size}  memory={args.memory_size}  theta={theta}")
    print(f"  gamma={args.gamma}  n_envs={n_envs}  total_steps={args.total_steps:,}")
    print(f"  {total_updates} updates × {steps_per_update:,} steps each\n")

    best_mean_reward = -np.inf
    t_start = time.time()

    for update in range(1, total_updates + 1):

        # ---- Collect rollout ------------------------------------------------
        rollout, h, m, key, last_obs, last_dones = collect_rollouts(
            train_env, train_state.params, act_fn,
            h, m, args.n_steps, n_envs, key, last_obs, last_dones, args.gamma,
        )
        total_steps += steps_per_update

        # ---- GAE bootstrap --------------------------------------------------
        obs_jax = np_to_jax(last_obs)
        _, last_val, _, _ = act_fn(train_state.params, obs_jax, h, m)
        last_val_np = np.array(last_val)

        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            last_val_np, args.gamma, args.gae_lambda,
        )

        # ---- Flatten (T, n_envs, ...) → (T*n_envs, ...) --------------------
        def flat(x):
            return x.reshape(-1, *x.shape[2:])

        def flat_adv_normal(flat_adv):
            return (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        N = steps_per_update
        flat_image   = flat(rollout["image"])    # (N, H, W, 3)
        flat_dir     = flat(rollout["dir"])      # (N,)
        flat_actions = flat(rollout["actions"])  # (N,)
        flat_logprob = flat(rollout["logprob"])  # (N,)
        flat_values  = flat(rollout["values"])   # (N,)
        flat_adv     = flat_adv_normal(flat(advantages))          # (N,)
        flat_ret     = flat(returns)             # (N,)
        flat_h       = flat(rollout["h"])        # (N, hidden_size)
        flat_m       = flat(rollout["m"])        # (N, memory_size)

        # ---- PPO epochs -----------------------------------------------------
        all_metrics = []
        indices = np.arange(N)
        for _epoch in range(args.n_epochs):
            # np.random.shuffle(indices)
            for start in range(0, N, args.batch_size):
                idx = indices[start : start + args.batch_size]
                if len(idx) < args.batch_size:
                    continue   # drop last incomplete minibatch
                batch = Batch(
                    obs_image    = jnp.array(flat_image[idx]),
                    obs_dir      = jnp.array(flat_dir[idx]),
                    actions      = jnp.array(flat_actions[idx], dtype=jnp.int32),
                    old_log_prob = jnp.array(flat_logprob[idx]),
                    old_values   = jnp.array(flat_values[idx]),
                    advantages   = jnp.array(flat_adv[idx]),
                    returns      = jnp.array(flat_ret[idx]),
                    lmu_h        = jnp.array(flat_h[idx]),
                    lmu_m        = jnp.array(flat_m[idx]),
                )
                train_state, metrics = update_fn(train_state, batch, args.clip_range)
                all_metrics.append({k: float(v) for k, v in metrics.items()})

        n_updates += 1

        # ---- TensorBoard scalar logging -------------------------------------
        if all_metrics:
            keys = all_metrics[0].keys()
            for k in keys:
                writer.add_scalar(k, np.mean([d[k] for d in all_metrics]), total_steps)

        fps = total_steps / (time.time() - t_start)
        writer.add_scalar("train/fps", fps, total_steps)

        if update % 10 == 0:
            mean_kl  = np.mean([d["train/approx_kl"]  for d in all_metrics])
            mean_gn  = np.mean([d["train/grad_norm"]   for d in all_metrics])
            mean_ev  = np.mean([d["train/expl_var"]    for d in all_metrics])
            mean_wm  = np.mean([d["debug/wm_grad"]     for d in all_metrics])
            adv_std  = float(np.std(flat_adv))
            print(f"  update {update:4d} / {total_updates}  "
                  f"steps={total_steps:>9,}  fps={fps:.0f}  "
                  f"kl={mean_kl:.4f}  grad={mean_gn:.2f}  ev={mean_ev:.3f}  "
                  f"wm_g={mean_wm:.4f}  adv_std={adv_std:.3f}")

        # ---- Periodic evaluation --------------------------------------------
        if update % eval_every == 0:
            mean_r = evaluate(eval_env, train_state.params, act_fn,
                              args.hidden_size, args.memory_size,
                              n_episodes=args.n_eval_episodes)
            writer.add_scalar("eval/mean_reward", mean_r, total_steps)
            flag = " ★" if mean_r > best_mean_reward else ""
            print(f"  [eval] steps={total_steps:,}  mean_reward={mean_r:.3f}{flag}")
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r

    # ---- Save ---------------------------------------------------------------
    import orbax.checkpoint as ocp
    ckpt_dir = os.path.abspath(f"lmu_ppo_jax_{args.env}_s{args.seed}")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_dir, train_state.params)
    print(f"\nParams saved to {ckpt_dir}/")

    writer.close()
    train_env.close()
    eval_env.close()

    print(f"\n=== Done  |  best eval reward: {best_mean_reward:.3f} ===")
    return best_mean_reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",           default="MemoryS7", choices=list(ENV_IDS))
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--n_envs",        type=int,   default=8)
    parser.add_argument("--total_steps",   type=int,   default=2_000_000)
    parser.add_argument("--n_steps",       type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--n_epochs",      type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--gamma",         type=float, default=0.999)
    parser.add_argument("--gae_lambda",    type=float, default=0.95)
    parser.add_argument("--clip_range",    type=float, default=0.2)
    parser.add_argument("--ent_coef",      type=float, default=0.01)
    parser.add_argument("--vf_coef",       type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--encoder_dim",   type=int,   default=64)
    parser.add_argument("--hidden_size",   type=int,   default=64)
    parser.add_argument("--memory_size",   type=int,   default=32)
    parser.add_argument("--eval_freq",     type=int,   default=10_000)
    parser.add_argument("--n_eval_episodes",type=int,  default=20)
    parser.add_argument("--tb_log",        default="runs/lmu_ppo_jax")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()