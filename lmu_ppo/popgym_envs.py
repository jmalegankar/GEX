"""
POPGym environment integration utilities.

Two wrappers + one factory:
    SingleKeyDictWrapper       — wrap any env so obs becomes Dict({'obs': space})
    TupleToMultiDiscreteWrapper — convert Tuple(Discrete, ...) → MultiDiscrete
    make_popgym_env             — factory matching make_env in train.py

Why the wrappers:

    LMURolloutBuffer inherits from SB3's DictRolloutBuffer, which expects a
    Dict obs space and allocates one numpy array per top-level key. POPGym
    tasks give Discrete / MultiDiscrete / Tuple / Box obs — flat, not Dict.

    SingleKeyDictWrapper turns each into Dict({'obs': original_space}). From
    the buffer's perspective this is structurally identical to MiniGrid's
    Dict({'image', 'direction'}), so no buffer changes are needed.

    Tuple obs is a separate problem: a Python tuple isn't a single ndarray,
    so the buffer can't store it. TupleToMultiDiscreteWrapper converts
    Tuple(Discrete(n_0), Discrete(n_1), ...) → MultiDiscrete([n_0, n_1, ...])
    upstream of the Dict wrap. AutoencodeMedium needs this; other tasks don't.

Verified obs spaces from the writeup:

    AutoencodeMedium     : Tuple(Discrete(2), Discrete(4)) → MD([2,4]) → Dict
    ConcentrationMedium  : MultiDiscrete([3]*100)          → Dict
    RepeatPreviousMedium : Discrete(4)                     → Dict
    CountRecallMedium    : MultiDiscrete([4, 4])           → Dict

Subprocess note:
    `import popgym` is inside _init, not in the outer factory body. Each
    SubprocVecEnv worker calls _init() in its own process, and popgym's
    `register()` side effects must run in that process for gym.make to
    resolve the popgym-* env IDs. Importing in the parent only works under
    fork-based start (Linux default), not spawn (Windows / new macOS / some
    cluster setups). Importing inside _init is the portable form.
"""

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


class SingleKeyDictWrapper(gym.ObservationWrapper):
    """
    Wrap an env so its observation_space becomes Dict({key: original_space}).

    Otherwise pass-through. Used to make flat obs spaces compatible with
    SB3's DictRolloutBuffer, which our LMURolloutBuffer inherits from.
    """

    def __init__(self, env: gym.Env, key: str = 'obs'):
        super().__init__(env)
        self._key = key
        self.observation_space = spaces.Dict({
            key: env.observation_space,
        })

    def observation(self, obs):
        return {self._key: obs}


class TupleToMultiDiscreteWrapper(gym.ObservationWrapper):
    """
    Convert Tuple(Discrete(n_0), ..., Discrete(n_k)) → MultiDiscrete([n_0,...,n_k]).

    The buffer can store MultiDiscrete as a single int64 ndarray of shape
    (n_components,), but a Python tuple has no defined storage layout in SB3's
    DictRolloutBuffer.

    Asserts every element of the Tuple is Discrete; raises otherwise. POPGym
    AutoencodeMedium is the only task we've seen with Tuple obs. If a future
    task has a Tuple of Box / mixed types, this wrapper rejects it loudly
    rather than silently misbehaving.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Tuple), (
            f"TupleToMultiDiscreteWrapper requires Tuple obs space; "
            f"got {type(env.observation_space).__name__}"
        )
        cardinalities = []
        for i, s in enumerate(env.observation_space.spaces):
            assert isinstance(s, spaces.Discrete), (
                f"TupleToMultiDiscreteWrapper requires every element be Discrete; "
                f"element {i} is {type(s).__name__}"
            )
            cardinalities.append(int(s.n))
        self._cardinalities = cardinalities
        self.observation_space = spaces.MultiDiscrete(cardinalities)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.int64)


def make_popgym_env(
    env_id: str,
    seed: int,
    rank: int = 0,
) -> Callable[[], gym.Env]:
    """
    Factory analogous to make_env in train.py for MiniGrid envs.

    Returns a thunk that, when called inside a worker process, creates a
    single env instance with the appropriate POPGym wrappers applied.

    Wrap order (innermost → outermost):
        gym.make → [TupleToMultiDiscrete if Tuple] → SingleKeyDict → Monitor

    The `import popgym` is inside _init so the env registration runs in
    whatever process actually invokes the thunk (essential for SubprocVecEnv
    under spawn-based multiprocessing).
    """

    def _init():
        import popgym  # noqa: F401  -- registers popgym-* IDs in THIS process
        env = gym.make(env_id)
        if isinstance(env.observation_space, spaces.Tuple):
            env = TupleToMultiDiscreteWrapper(env)
        env = SingleKeyDictWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init
