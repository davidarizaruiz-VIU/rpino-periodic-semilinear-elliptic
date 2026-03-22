from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, conv, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            conv(channels, channels, kernel_size=kernel_size, padding=pad, padding_mode='circular'),
            nn.GELU(),
            conv(channels, channels, kernel_size=kernel_size, padding=pad, padding_mode='circular'),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualNet(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_channels: int,
        depth: int,
        rho: float,
        kernel_size: int = 5,
        controlled: bool = True,
    ):
        super().__init__()
        self.dimension = dimension
        self.rho = float(rho)
        self.controlled = bool(controlled)
        conv = nn.Conv1d if dimension == 1 else nn.Conv2d
        pad = kernel_size // 2

        self.input_layer = conv(2, hidden_channels, kernel_size=kernel_size, padding=pad, padding_mode='circular')
        self.blocks = nn.ModuleList([ResidualBlock(conv, hidden_channels, kernel_size) for _ in range(depth)])
        self.output_layer = conv(hidden_channels, 1, kernel_size=kernel_size, padding=pad, padding_mode='circular')
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u, f], dim=1)
        x = F.gelu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        out = self.output_layer(x)
        if self.controlled:
            return self.rho * torch.tanh(out)
        return out


class RPINO(nn.Module):
    def __init__(self, backbone_operator, residual_net: ResidualNet, state_iters_train: int, state_iters_eval: int):
        super().__init__()
        self.backbone_operator = backbone_operator
        self.residual_net = residual_net
        self.state_iters_train = state_iters_train
        self.state_iters_eval = state_iters_eval

    def step(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            b = self.backbone_operator(u, f)
        return b + self.residual_net(u, f)

    def solve(self, f: torch.Tensor, training: bool = True, n_iters: int | None = None) -> torch.Tensor:
        n_steps = n_iters if n_iters is not None else (self.state_iters_train if training else self.state_iters_eval)
        u = torch.zeros_like(f)
        for _ in range(n_steps):
            u = self.step(u, f)
        return u

    def backbone_only(self, f: torch.Tensor, eval_mode: bool = False, n_iters: int | None = None) -> torch.Tensor:
        n_steps = n_iters if n_iters is not None else (self.state_iters_eval if eval_mode else self.state_iters_train)
        u = torch.zeros_like(f)
        for _ in range(n_steps):
            with torch.no_grad():
                u = self.backbone_operator(u, f)
        return u

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.solve(f, training=self.training)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, channels, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum('bim,iom->bom', x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / math.sqrt(in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, channels, nx, ny = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batchsize, self.out_channels, nx, x_ft.size(-1), dtype=torch.cfloat, device=x.device)
        mx = min(self.modes_x, nx)
        my = min(self.modes_y, x_ft.size(-1))
        out_ft[:, :, :mx, :my] = torch.einsum(
            'bixy,ioxy->boxy', x_ft[:, :, :mx, :my], self.weight[:, :, :mx, :my]
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int):
        super().__init__()
        self.spec = SpectralConv2d(width, width, modes_x, modes_y)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.w(x))


class FNO1d(nn.Module):
    def __init__(self, width: int, modes: int, depth: int):
        super().__init__()
        self.input = nn.Conv1d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock1d(width, modes) for _ in range(depth)])
        self.output = nn.Sequential(nn.Conv1d(width, width, kernel_size=1), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = self.input(f)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class FNO2d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, depth: int):
        super().__init__()
        self.input = nn.Conv2d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2d(width, modes_x, modes_y) for _ in range(depth)])
        self.output = nn.Sequential(nn.Conv2d(width, width, kernel_size=1), nn.GELU(), nn.Conv2d(width, 1, kernel_size=1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = self.input(f)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def build_fno(dimension: int, width: int, depth: int, modes_x: int, modes_y: int | None = None):
    if dimension == 1:
        return FNO1d(width=width, modes=modes_x, depth=depth)
    return FNO2d(width=width, modes_x=modes_x, modes_y=(modes_y or modes_x), depth=depth)
