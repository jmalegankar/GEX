from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces

from .buffer import TransitionRolloutBuffer

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


SelfHSWVimePPO = TypeVar("SelfHSWVimePPO", bound="HSWVimePPO")

class HSWVimePPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) with intrinsic rewards.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, etc.)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0). It is used for the
        Adam optimizer.
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_envs where n_envs is number of environment copies running in parallel).
    :param batch_size: Minibatch size for each gradient update. For recurrent policies,
        the value should be a factor of n_steps * n_envs.
    :param n_epochs: Number of epoch when optimizing the surrogate loss.
    :param gamma: Discount factor.
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage
    """