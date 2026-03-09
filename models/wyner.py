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
        recon_next_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.recon_next_loss = recon_next_loss
        self.kl_loss = kl_loss

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


class WynerDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        latent_tokens: int,
        mu_dim: int,
        recon_dim: int,
        decode_hidden: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_tokens = latent_tokens
        self.mu_dim = mu_dim
        self.recon_dim = recon_dim
        self.decode_hidden = decode_hidden

        self.proj = nn.Linear(self.latent_dim, self.decode_hidden)
        
        self.fc1 = nn.Linear(self.decode_hidden * self.latent_tokens + self.mu_dim, self.decode_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.decode_hidden, self.decode_hidden)
        self.fc_out = nn.Linear(self.decode_hidden, self.recon_dim)
    
    def forward(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        z_proj = self.proj(z)
        z_proj = z_proj.view(z_proj.size(0), self.latent_tokens * self.decode_hidden)
        x = th.cat([z_proj, mu], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        recon = self.fc_out(x)
        return recon


class WynerVAE(nn.Module):
    def __init__(
        self,
        recon_dim: int,
        mu_dim: int,
        latent_dim: int,
        latent_tokens: int = 1,
        decode_hidden: int = 128,
        state_dim: int = 0,
        state_tokens: int = 0,
    ):
        super().__init__()
        assert latent_tokens >= 1, "Latent tokens must be at least 1."
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.state_tokens = state_tokens
        self.latent_tokens = latent_tokens

        self.mu_dim = mu_dim
        self.recon_dim = recon_dim

        if self.state_tokens == 0 and self.latent_tokens == 1:
            self.gru = nn.GRUCell(input_size=mu_dim, hidden_size=latent_dim)
        else:
            raise NotImplementedError("Only single token encoding/decoding is implemented for now.")
        
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.zeros_(self.fc_logvar.bias)

        self.decoder = WynerDecoder(
            latent_dim=self.latent_dim,
            latent_tokens=self.latent_tokens,
            mu_dim=self.mu_dim,
            recon_dim=self.recon_dim,
            decode_hidden=decode_hidden,
        )
    
    def encode(self, w: th.Tensor, mu: th.Tensor, skips: Optional[List[th.Tensor]] = None) -> Tuple[th.Tensor, th.Tensor]:
        h = self.gru(mu, w.squeeze(1))
        z_mu = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mu, z_logvar
    
    def decode(self, z: th.Tensor, mu: th.Tensor) -> th.Tensor:
        return self.decoder(z, mu)
    
    def forward(self, w: th.Tensor, mu: th.Tensor, mu_next: Optional[th.Tensor] = None, skips: Optional[List[th.Tensor]] = None) -> WynerOutput:
        z_mu, z_logvar = self.encode(w, mu, skips)
        std = th.exp(0.5 * z_logvar)
        eps = th.randn_like(std)
        z = z_mu + eps * std
        recon = self.decode(z, mu)
        recon_next = self.decode(z, mu_next) if mu_next is not None else None
        return WynerOutput(w=w, logvar=z_logvar, recon=recon, recon_next=recon_next)
    
    def loss(self, output: WynerOutput, recon_target: Optional[th.Tensor] = None, recon_next_target: Optional[th.Tensor] = None) -> WynerLoss:
        kl_loss = -0.5 * th.sum(1 + output.logvar - output.w.pow(2) - output.logvar.exp(), dim=-1)

        if recon_target is not None:
            recon_loss = nn.functional.mse_loss(output.recon, recon_target, reduction='none').mean(dim=-1)
        else:
            recon_loss = None
        
        if recon_next_target is not None and output.recon_next is not None:
            recon_next_loss = nn.functional.mse_loss(output.recon_next, recon_next_target, reduction='none').mean(dim=-1)
        else:
            recon_next_loss = None

        return WynerLoss(kl_loss=kl_loss, recon_loss=recon_loss, recon_next_loss=recon_next_loss)