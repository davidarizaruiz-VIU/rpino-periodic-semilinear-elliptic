from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def _add_channel(x: np.ndarray, dimension: int) -> np.ndarray:
    if dimension == 1:
        return x[:, None, :]
    return x[:, None, :, :]


def make_loader(f: np.ndarray, u: np.ndarray, dimension: int, batch_size: int, shuffle: bool) -> DataLoader:
    f_t = torch.tensor(_add_channel(f, dimension), dtype=torch.float32)
    u_t = torch.tensor(_add_channel(u, dimension), dtype=torch.float32)
    ds = TensorDataset(f_t, u_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
