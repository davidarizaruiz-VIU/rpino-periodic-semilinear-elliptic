from __future__ import annotations
import math
import numpy as np


class PeriodicNonlinearPoisson:
    def __init__(self, dimension: int, grid_size: int, domain_length: float, kappa: float):
        if dimension not in (1, 2):
            raise ValueError("Only dimensions 1 and 2 are implemented.")
        self.dimension = dimension
        self.n = grid_size
        self.L = domain_length
        self.kappa = float(kappa)
        self.h = domain_length / grid_size
        self._build_wavenumbers()

    def _build_wavenumbers(self) -> None:
        freq = 2.0 * math.pi * np.fft.fftfreq(self.n, d=self.h)
        if self.dimension == 1:
            self.k2 = freq**2
        else:
            kx, ky = np.meshgrid(freq, freq, indexing="ij")
            self.k2 = kx**2 + ky**2
        self.linear_symbol = self.kappa + self.k2
        self.linear_symbol[tuple([0] * self.dimension)] = self.kappa

    def linear_solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_hat = np.fft.fftn(rhs)
        u_hat = rhs_hat / self.linear_symbol
        return np.fft.ifftn(u_hat).real

    def coarse_linear_solve(self, rhs: np.ndarray, coarse_modes: int) -> np.ndarray:
        rhs_hat = np.fft.fftn(rhs)
        mask = self.low_mode_mask(coarse_modes)
        u_hat = np.where(mask, rhs_hat / self.linear_symbol, 0.0)
        return np.fft.ifftn(u_hat).real

    def low_mode_mask(self, coarse_modes: int) -> np.ndarray:
        if self.dimension == 1:
            idx = np.fft.fftfreq(self.n) * self.n
            return np.abs(idx) <= coarse_modes
        idx = np.fft.fftfreq(self.n) * self.n
        kx, ky = np.meshgrid(idx, idx, indexing="ij")
        return (np.abs(kx) <= coarse_modes) & (np.abs(ky) <= coarse_modes)

    def fixed_point_map(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        return self.linear_solve(f - u**3)

    def coarse_backbone_map(self, u: np.ndarray, f: np.ndarray, coarse_modes: int) -> np.ndarray:
        return self.coarse_linear_solve(f - u**3, coarse_modes=coarse_modes)

    def solve_reference(self, f: np.ndarray, max_iter: int = 400, tol: float = 1e-10, damping: float = 1.0):
        u = np.zeros_like(f)
        history = []
        for it in range(max_iter):
            t = self.fixed_point_map(u, f)
            new_u = (1.0 - damping) * u + damping * t
            inc = np.linalg.norm((new_u - u).ravel()) / math.sqrt(new_u.size)
            residual = np.linalg.norm((new_u - self.fixed_point_map(new_u, f)).ravel()) / math.sqrt(new_u.size)
            history.append({"iter": it + 1, "increment": inc, "residual": residual})
            u = new_u
            if inc < tol:
                break
        return u, history

    def l2_error(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.sqrt(np.mean((u - v) ** 2)))

    def forcing_sample(self, rng: np.random.Generator, amplitude: float, modes: int) -> np.ndarray:
        if self.dimension == 1:
            x = np.linspace(0.0, self.L, self.n, endpoint=False)
            f = np.zeros_like(x)
            for k in range(1, modes + 1):
                a = rng.normal(scale=amplitude / (k**2))
                b = rng.normal(scale=amplitude / (k**2))
                f += a * np.cos(k * x) + b * np.sin(k * x)
            return f
        x = np.linspace(0.0, self.L, self.n, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = np.zeros((self.n, self.n), dtype=float)
        for k1 in range(0, modes + 1):
            for k2 in range(0, modes + 1):
                if k1 == 0 and k2 == 0:
                    continue
                scale = amplitude / ((1 + k1 + k2) ** 2)
                coeffs = rng.normal(scale=scale, size=4)
                f += coeffs[0] * np.cos(k1 * X) * np.cos(k2 * Y)
                f += coeffs[1] * np.cos(k1 * X) * np.sin(k2 * Y)
                f += coeffs[2] * np.sin(k1 * X) * np.cos(k2 * Y)
                f += coeffs[3] * np.sin(k1 * X) * np.sin(k2 * Y)
        return f
