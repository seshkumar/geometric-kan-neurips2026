from __future__ import annotations
import torch

def carve_validation_split(X, y, val_frac: float=0.2, seed: int=0):
    n = X.shape[0]
    n_val = int(round(val_frac * n)) if n >= 5 else 0
    if n_val < 1:
        return (X, y, None, None)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, fit_idx = (perm[:n_val], perm[n_val:])
    return (X[fit_idx], y[fit_idx], X[val_idx], y[val_idx])
