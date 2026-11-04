"""Shared sample-standard-deviation convention for fresh NeurIPS experiments."""
import numpy as np
import torch


def standard_deviation(values, axis=None, ddof=1):
    """Sample SD by default; ddof=0 exists only for declared legacy regeneration."""
    if ddof not in (0, 1):
        raise ValueError('SD correction must be zero or one')
    if isinstance(values, torch.Tensor):
        return values.std(dim=axis, correction=ddof)
    return np.std(values, axis=axis, ddof=ddof)


def normalize_targets(y_train, y_test):
    """The original core runner's training-only sample-SD normalization."""
    mu = y_train.mean()
    scale = standard_deviation(y_train)
    if scale < 1e-8:
        scale = torch.tensor(1.0, dtype=y_train.dtype, device=y_train.device)
    return (y_train-mu)/scale, (y_test-mu)/scale, mu, scale


def symbolic_data(eq, n_train, sigma, seed, n_test=1000):
    """Reuse the core generator so overlapping studies receive identical tensors."""
    from experiments import feynman_runner as core
    if eq[0] not in core.FEYNMAN_EQUATIONS:
        raise ValueError('Equation absent from the canonical NeurIPS generator')
    x, y = core.generate_data(eq[0], n_train+n_test, seed)
    train, test, _, scale = normalize_targets(y[:n_train], y[n_train:])
    if sigma > 0:
        torch.manual_seed(seed+int(sigma*1000))
        train = train+torch.randn(n_train)*sigma
    return x[:n_train], train, x[n_train:], test, float(scale)
