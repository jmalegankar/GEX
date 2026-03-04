import torch as th

from typing import Optional, List, Tuple

@th.jit.script
class VAEOutput:
    def __init__(
        self,
        recon: th.Tensor,
        mu: th.Tensor,
        rho: th.Tensor,
        recon_target: th.Tensor,
        skips: Optional[List[th.Tensor]] = None
    ):
        self.recon = recon
        self.mu = mu
        self.rho = rho
        self.recon_target = recon_target
        self.skips = skips


@th.jit.script
class VAELoss:
    def __init__(
        self,
        recon_loss: th.Tensor,
        kl_loss: th.Tensor,
        aux_loss: Optional[th.Tensor] = None,
    ):
        self.recon_loss = recon_loss
        self.kl_loss = kl_loss
        self.aux_loss = aux_loss


@th.jit.interface
class VAEInterface:
    def encode(self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[List[th.Tensor]]]:
        pass

    def decode(self, z_t: th.Tensor, skips: Optional[List[th.Tensor]] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        pass

    def forward(self, s_t: th.Tensor, a_t: th.Tensor, s_tp1: th.Tensor) -> VAEOutput:
        pass

    def loss(self, output: VAEOutput) -> VAELoss:
        pass