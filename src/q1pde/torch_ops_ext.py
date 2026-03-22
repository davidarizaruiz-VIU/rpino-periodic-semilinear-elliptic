"""Extended backbone wrappers for non-cubic nonlinearities.

TorchCoarseBackboneBurgers wraps the steady viscous Burgers PDE solver
for use as a backbone in the RP-INO architecture.
"""
from __future__ import annotations
import numpy as np
import torch
from q1pde.pde_burgers import PeriodicBurgers2D


class TorchCoarseBackboneBurgers:
    """Coarse backbone for steady viscous Burgers: -nu*Lap u + u*du/dx + kappa*u = f."""

    def __init__(self, pde: PeriodicBurgers2D, coarse_modes: int, nonlinear_weight: float = 1.0):
        self.pde = pde
        self.coarse_modes = int(coarse_modes)
        self.nonlinear_weight = float(nonlinear_weight)

    def __call__(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        device = u.device
        u_np = u.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
        out = []
        for uu, ff in zip(u_np, f_np):
            # Burgers nonlinearity: u * du/dx  (not u^3)
            nl = self.nonlinear_weight * self.pde.nonlinear_term(uu[0])
            rhs = ff[0] - nl
            bb = self.pde.coarse_linear_solve(rhs, self.coarse_modes)
            out.append(bb[None, ...])
        arr = np.stack(out).astype(np.float32)
        return torch.tensor(arr, device=device)
