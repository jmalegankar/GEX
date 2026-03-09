import torch as th
import torch.nn as nn

from typing import Optional, List, Tuple

@th.jit.script
class WynerOutput:
    def __init__(
        self,
        w: th.Tensor,
        logvar: th.Tensor,
        recon: th.Tensor,
        recon_next: Optional[th.Tensor] = None,
    ):
        self.recon = recon
        self.recon_next = recon_next
        self.w = w
        self.logvar = logvar


@th.jit.script
class WynerLoss:
    def __init__(
        self,
        kl_loss: th.Tensor,
        recon_loss: Optional[th.Tensor] = None,
        kl_next_loss: Optional[th.Tensor] = None,
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss
        self.kl_loss = kl_loss
        self.kl_next_loss = kl_next_loss

@th.jit.interface
class WynerInterface:
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]]) -> Tuple[th.Tensor, th.Tensor]:
        pass

    def decode(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        pass

    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: Optional[th.Tensor], skips: Optional[List[th.Tensor]]) -> WynerOutput:
        pass

    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor], recon_next_target: Optional[th.Tensor]) -> WynerLoss:
        pass


class WynerVAE(nn.Module):
    def __init__(
        self,
        memory_dim: int = 64,
        mu_dim: int = 32,
        wyner_latent_dim: int = 32,
        recon_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.mu_dim = mu_dim
        self.wyner_latent_dim = wyner_latent_dim
        self.recon_dim = recon_dim

        #gru for encoding (w, mu) into memory
        self.gru = nn.GRUCell(input_size=mu_dim, hidden_size=memory_dim)

        # linear layers for mapping memory to Wyner latent space (w, logvar)
        self.fc_mu = nn.Linear(memory_dim, wyner_latent_dim)
        self.fc_logvar = nn.Linear(memory_dim, wyner_latent_dim)

        # decoder for reconstructing from Wyner latent space + mu
        self.decoder = nn.Sequential(
            nn.Linear(wyner_latent_dim + mu_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, recon_dim),
        )
    
    def _encode_internal(self, w: th.Tensor, mu: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        orig_shape = w.shape
        w_flat = w.reshape(w.size(0), -1)          # (B, memory_dim)

        h = self.gru(mu, w_flat)                   # (B, memory_dim)

        z_mu = self.fc_mu(h)                       # (B, wyner_latent_dim)
        z_logvar = self.fc_logvar(h)               # (B, wyner_latent_dim)

        new_memory = h.view(orig_shape)            # (B, 1, memory_dim)
        return new_memory, z_mu, z_logvar

    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]] = None) -> Tuple[th.Tensor, th.Tensor]:
        new_memory, z_mu, z_logvar = self._encode_internal(w, mu)
        return new_memory, z_logvar
    
    def decode(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        return self.decoder(th.cat([z, mu], dim=-1))
    
    # i think we need a regular kl loss helper func 
    # and reparameterization helper func for sampling z from (mu, logvar) during training
    def forward():
        pass
    
    # do we need separate loss for single step in for rollout?
    # or just 
    # kl = ...
    # kl_next = ...
    # recon = ...
    # recon_next = ...
    def loss():
        pass
    