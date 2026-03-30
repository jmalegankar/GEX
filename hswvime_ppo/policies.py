import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from typing import Any, Optional, Union, Tuple

from models.vae import VAEInterface, TransitionSCVAE
from models.wyner import WynerInterface, WynerVAE

class HSWVIMEFeaturesExtractor(nn.Module):
    """
    HSWVIME features extractor.

    :param observation_space: The observation space
    :param features_dim: The number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space=None, *, mu_dim: int, slot_dim: int = 64):
        super().__init__()
        self.mu_dim = mu_dim
        self.slot_dim = slot_dim
        # Output: mean(slots)(slot_dim) concat pi(mu_dim)
        self._features_dim = slot_dim + mu_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, slot_features: th.Tensor, mu_features: th.Tensor) -> th.Tensor:
        # mu_features = pi (B, mu_dim=32), slot_features = (B, K, slot_dim=64)
        # All slots contain similar content (same turn-around observation),
        # so mean pooling is equivalent to attention but with zero learning overhead.
        slot_mean = slot_features.mean(dim=1)  # (B, slot_dim)
        x = th.cat((mu_features, slot_mean), dim=-1)  # (B, mu_dim + slot_dim)
        return x

class HSWVIMEActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[HSWVIMEFeaturesExtractor] = HSWVIMEFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        vae_features_extractor_class: VAEInterface = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        wyner_features_extractor_class: WynerInterface = WynerVAE,
        wyner_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        assert share_features_extractor, "HSWVIME does not support separate feature extractors for policy and value networks"
        # Store as plain attrs before super().__init__() — nn.Module is not yet
        # initialised so we cannot assign nn.Module instances yet.
        self.vae_features_extractor_class = vae_features_extractor_class
        self.vae_features_extractor_kwargs = vae_features_extractor_kwargs or {}
        self.wyner_features_extractor_class = wyner_features_extractor_class
        self.wyner_features_extractor_kwargs = wyner_features_extractor_kwargs or {}

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
    
    def make_features_extractor(self):
        self.vae_feature_extractor: VAEInterface = self.vae_features_extractor_class(
            **self.vae_features_extractor_kwargs
        )
        self.wyner_feature_extractor: WynerInterface = self.wyner_features_extractor_class(
            **self.wyner_features_extractor_kwargs
        )
        return super().make_features_extractor()

    def forward(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        deterministic: bool = False,
        timestep: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param s_tm1: Previous state
        :param a_tm1: Previous action
        :param s_t: Current state
        :param memory: Wyner GRU hidden state h_{t-1}
        :param slots: Slot memory state (B, K, slot_dim)
        :param deterministic: Whether to sample or use deterministic actions
        :param timestep: Episode timestep indices (B,)
        :return: action, h_t memory, value and log probability of the action
        """
        # Preprocess the observation if needed
        features, memory = self.extract_features(s_tm1, a_tm1, s_t, memory, slots, timestep=timestep)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, memory, values, log_prob
    
    def extract_features(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        s_tm1 = preprocess_obs(s_tm1, self.observation_space, normalize_images=self.normalize_images)
        s_t = preprocess_obs(s_t, self.observation_space, normalize_images=self.normalize_images)
        mu, pi, skips = self.vae_feature_extractor.encode(s_tm1, a_tm1, s_t)
        new_memory, _ = self.wyner_feature_extractor.encode(memory, mu.detach(), skips, timestep=timestep)
        # encode returns (B, latent_dim); restore the seq dim for storage and MHA
        new_memory = new_memory.unsqueeze(1)  # (B, 1, latent_dim)
        # Dual-head: μ (detached) for VAE/Wyner, π for attention/policy.
        # π receives PPO gradients through attention, μ stays protected.
        # Slots contain π values written during rollout (detached).
        features = self.features_extractor(slots.detach(), pi)
        return features, new_memory
    
    def get_distribution(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[Distribution, th.Tensor]:
        features, memory = self.extract_features(s_tm1, a_tm1, s_t, memory, slots, timestep=timestep)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi), memory

    def predict_values(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        features, _ = self.extract_features(s_tm1, a_tm1, s_t, memory, slots, timestep=timestep)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        action: th.Tensor,
        timestep: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        features, new_memory = self.extract_features(s_tm1, a_tm1, s_t, memory, slots, timestep=timestep)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(action)
        return values, log_prob, distribution.entropy(), new_memory
    
    def predict(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        memory: th.Tensor,
        slots: th.Tensor,
        deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(s_tm1, tuple) and len(s_tm1) == 2 and isinstance(s_tm1[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        s_tm1, vectorized_env = self.obs_to_tensor(s_tm1)
        s_t, _ = self.obs_to_tensor(s_t)
        a_tm1 = th.as_tensor(a_tm1, device=s_tm1.device)

        with th.no_grad():
            distribution, memory = self.get_distribution(s_tm1, a_tm1, s_t, memory, slots)
            actions = distribution.get_actions(deterministic=deterministic)
        
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)  # type: ignore[assignment]

        return actions, memory  # type: ignore[return-value]
