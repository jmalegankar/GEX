import warnings

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


class SimpleVAEFeaturesExtractor(nn.Module):
    """
    Features extractor that concatenates sg(mu) and GRU hidden state h_mem.
    If no GRU is used, passes mu directly.
    """

    def __init__(self, observation_space=None, *, mu_dim: int, gru_hidden_dim: int = 0):
        super().__init__()
        self._features_dim = mu_dim + gru_hidden_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, features: th.Tensor) -> th.Tensor:
        return features


class HSWVIMEActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy with transition VAE and optional GRU memory.

    Features = concat(sg(mu), h_mem) where:
      - mu comes from VAE.encode(s_{t-1}, a_{t-1}, s_t)
      - h_mem = GRU(sg(mu), h_{t-1}_mem), trained only through PPO gradients
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
        features_extractor_class: type[SimpleVAEFeaturesExtractor] = SimpleVAEFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        vae_features_extractor_class: VAEInterface = TransitionSCVAE,
        vae_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        gru_hidden_dim: int = 0,
    ):
        assert share_features_extractor, "Does not support separate feature extractors for policy and value networks"
        self.vae_features_extractor_class = vae_features_extractor_class
        self.vae_features_extractor_kwargs = vae_features_extractor_kwargs or {}
        self.gru_hidden_dim = gru_hidden_dim

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
        # Infer mu_dim from the VAE's latent_dim
        mu_dim = self.vae_feature_extractor.latent_dim

        # Build GRU if requested
        if self.gru_hidden_dim > 0:
            self.gru = nn.GRU(
                input_size=mu_dim,
                hidden_size=self.gru_hidden_dim,
                batch_first=False,  # input: (1, B, mu_dim)
            )
        else:
            self.gru = None

        return super().make_features_extractor()

    def _gru_step(
        self,
        mu: th.Tensor,
        h_prev: Optional[th.Tensor],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Single GRU step: h_mem = GRU(sg(mu), h_prev).

        Returns (features, h_mem) where features = concat(sg(mu), h_mem).
        sg(mu) ensures VAE encoder doesn't get gradients from PPO loss.
        """
        mu_sg = mu.detach()  # stop gradient from PPO -> VAE encoder
        if self.gru is None:
            return mu_sg, th.zeros(1, device=mu.device)  # dummy h_mem

        if h_prev is None:
            h_prev = th.zeros(1, mu.size(0), self.gru_hidden_dim, device=mu.device)

        # GRU expects (seq_len=1, batch, input_size)
        gru_out, h_mem = self.gru(mu_sg.unsqueeze(0), h_prev)
        # gru_out: (1, B, H), h_mem: (1, B, H)
        features = th.cat([mu_sg, gru_out.squeeze(0)], dim=-1)
        return features, h_mem

    def forward(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        deterministic: bool = False,
        h_prev: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all networks (actor and critic).

        Returns: (action, value, log_prob, h_mem)
        """
        features, h_mem = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, h_mem

    def extract_features(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        h_prev: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Returns (features, h_mem)."""
        s_tm1 = preprocess_obs(s_tm1, self.observation_space, normalize_images=self.normalize_images)
        s_t = preprocess_obs(s_t, self.observation_space, normalize_images=self.normalize_images)
        mu, _, skips = self.vae_feature_extractor.encode(s_tm1, a_tm1, s_t)

        if self.gru is not None:
            features, h_mem = self._gru_step(mu, h_prev)
        else:
            features = mu.detach()
            h_mem = th.zeros(1, device=mu.device)

        features = self.features_extractor(features)
        return features, h_mem

    def get_distribution(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        h_prev: Optional[th.Tensor] = None,
    ) -> Distribution:
        features, _ = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        h_prev: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        features, _ = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        action: th.Tensor,
        h_prev: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Returns (values, log_prob, entropy, h_mem)."""
        features, h_mem = self.extract_features(s_tm1, a_tm1, s_t, h_prev)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(action)
        return values, log_prob, distribution.entropy(), h_mem

    def predict(
        self,
        s_tm1: th.Tensor,
        a_tm1: th.Tensor,
        s_t: th.Tensor,
        deterministic: bool = False,
        h_prev: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Returns (actions, h_mem)."""
        self.set_training_mode(False)

        if isinstance(s_tm1, tuple) and len(s_tm1) == 2 and isinstance(s_tm1[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API."
            )

        s_tm1, vectorized_env = self.obs_to_tensor(s_tm1)
        s_t, _ = self.obs_to_tensor(s_t)
        a_tm1 = th.as_tensor(a_tm1, device=s_tm1.device)

        with th.no_grad():
            distribution = self.get_distribution(s_tm1, a_tm1, s_t, h_prev)
            actions = distribution.get_actions(deterministic=deterministic)

        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                actions = self.unscale_action(actions)
            else:
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions
