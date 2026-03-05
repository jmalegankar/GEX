import torch as th

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