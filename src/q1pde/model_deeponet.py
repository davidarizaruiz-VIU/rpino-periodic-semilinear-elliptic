"""DeepONet baseline for 2-D periodic operator learning.

Architecture
------------
Branch network : small CNN that maps the input function f(x) (on a grid)
                 to a vector in R^P.
Trunk network  : MLP that maps spatial coordinate (x,y) to R^P.
Output         : sum_p branch_p * trunk_p  (dot product).

For a fixed grid the trunk outputs are shared across all samples, so
we precompute them once and store as a buffer.

Target parameter count  ~200 K  to match RP-INO (207 K).
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BranchCNN2d(nn.Module):
    """Small CNN that encodes a 2-D field into a vector of size *width*."""

    def __init__(self, grid_size: int, width: int):
        super().__init__()
        # Conv layers with stride-2 downsampling
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.GELU(),
        )
        # Determine flattened size after conv
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_size, grid_size)
            flat = self.conv(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f : (B, 1, N, N)  ->  (B, width)"""
        return self.fc(self.conv(f))


class TrunkMLP(nn.Module):
    """MLP that maps (x, y) coordinates to width-dim embedding."""

    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords : (M, 2)  ->  (M, width)"""
        return self.net(coords)


class DeepONet2d(nn.Module):
    """Full DeepONet for 2-D operator learning on a fixed grid.

    Parameters
    ----------
    grid_size : int
        Number of grid points per spatial dimension.
    domain_length : float
        Physical domain size (default 2*pi for periodic problems).
    width : int
        Latent dimension P shared by branch and trunk.
    trunk_hidden : int
        Hidden layer width in the trunk MLP.
    """

    def __init__(self, grid_size: int, domain_length: float = 2 * math.pi,
                 width: int = 64, trunk_hidden: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.domain_length = domain_length
        self.width = width

        self.branch = BranchCNN2d(grid_size, width)
        self.trunk = TrunkMLP(width, hidden=trunk_hidden)
        self.bias = nn.Parameter(torch.zeros(1))

        # Build and register the coordinate grid (fixed)
        x = torch.linspace(0, domain_length, grid_size + 1)[:-1]
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (N*N, 2)
        self.register_buffer('coords', coords)
        # Pre-computed trunk outputs will be stored here
        self._trunk_cache: torch.Tensor | None = None

    def _trunk_outputs(self) -> torch.Tensor:
        """Evaluate trunk on the coordinate grid -> (N*N, width)."""
        return self.trunk(self.coords)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """f : (B, 1, N, N)  ->  (B, 1, N, N)"""
        B = f.size(0)
        N = self.grid_size

        # Branch: (B, width)
        b = self.branch(f)

        # Trunk: (N*N, width) — recompute each forward (needed for training)
        t = self._trunk_outputs()

        # Dot product:  out[i, j] = sum_p  b[i,p] * t[j,p]  + bias
        out = torch.einsum('bp,mp->bm', b, t) + self.bias  # (B, N*N)
        return out.view(B, 1, N, N)


def build_deeponet(grid_size: int, domain_length: float = 2 * math.pi,
                   width: int = 64, trunk_hidden: int = 128) -> DeepONet2d:
    return DeepONet2d(grid_size=grid_size, domain_length=domain_length,
                      width=width, trunk_hidden=trunk_hidden)
