import torch
import torch.nn as nn
from torch.autograd import grad

from mlp import MLP


class HNN(nn.Module):

    def __init__(self,
                 d_input: int = 2,
                 d_hidden: int = 32,
                 activation_fn: str = 'tanh') -> None:

        super().__init__()
        self.net = MLP(d_input=d_input,
                        d_hidden=d_hidden,
                        d_output=1,
                        activation=activation_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.symplectic_gradient(z)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        H = self.net(z).squeeze(-1)
        return H

    def symplectic_gradient(self, z: torch.Tensor) -> torch.Tensor:

        # Ensure z requires gradients
        if not z.requires_grad:
            z = z.requires_grad_(True)

        H = self.net(z).squeeze(-1)

        grad_H = grad(H.sum(), z, create_graph=True)[0]

        dim = z.shape[-1] // 2
        dH_dq = grad_H[..., :dim]
        dH_dp = grad_H[..., dim:]

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return torch.cat([dq_dt, dp_dt], dim=-1)
