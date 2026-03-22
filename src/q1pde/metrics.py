from __future__ import annotations
import numpy as np


def relative_l2_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    num = np.sqrt(np.mean((pred - true) ** 2, axis=tuple(range(1, pred.ndim))))
    den = np.sqrt(np.mean(true**2, axis=tuple(range(1, true.ndim)))) + 1e-12
    return num / den
