import numpy as np
import torch
import time
from scipy.interpolate import BSpline
EPS = 1e-08
JP_K_EDGES = 5
JP_N_INITS = 18
JP_N_LBFGS_OUTER = 100
JP_SEED = 1729

def heaviside_step(x: np.ndarray) -> np.ndarray:
    return (x >= 0.0).astype(float) - 0.5

def five_level_step(x: np.ndarray) -> np.ndarray:
    levels = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    edges = np.linspace(-1.0, 1.0, 6)
    y = np.full_like(x, levels[-1])
    for i in range(5):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        y[mask] = levels[i]
    return y

def bspline_basis(x: np.ndarray, K: int, k: int=3) -> np.ndarray:
    n_interior = K - k - 1
    if n_interior < 0:
        raise ValueError('K too small for cubic B-spline')
    if n_interior == 0:
        inner = np.array([])
    else:
        inner = np.linspace(-1.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([np.full(k + 1, -1.0), inner, np.full(k + 1, 1.0)])
    x_clip = np.clip(x, -1.0 + 1e-09, 1.0 - 1e-09)
    basis = np.zeros((x.size, K))
    for i in range(K):
        c = np.zeros(K)
        c[i] = 1.0
        spl = BSpline(knots, c, k, extrapolate=False)
        basis[:, i] = spl(x_clip)
    return basis

def chebyshev_basis(x: np.ndarray, K: int) -> np.ndarray:
    basis = np.zeros((x.size, K))
    basis[:, 0] = 1.0
    if K > 1:
        basis[:, 1] = x
    for i in range(2, K):
        basis[:, i] = 2.0 * x * basis[:, i - 1] - basis[:, i - 2]
    return basis

def rbf_basis(x: np.ndarray, K: int) -> np.ndarray:
    centres = np.linspace(-1.0, 1.0, K)
    width = 2.0 / max(K - 1, 1)
    return np.exp(-((x[:, None] - centres[None, :]) / width) ** 2)

def haar_basis(x: np.ndarray, K: int) -> np.ndarray:
    basis = np.zeros((x.size, K))
    basis[:, 0] = 1.0
    idx = 1
    j = 0
    while idx < K:
        n_at_j = 2 ** j
        for kk in range(n_at_j):
            if idx >= K:
                break
            a = -1.0 + 2.0 * kk / n_at_j
            mid = -1.0 + 2.0 * (kk + 0.5) / n_at_j
            b = -1.0 + 2.0 * (kk + 1.0) / n_at_j
            v = np.zeros_like(x)
            v[(x >= a) & (x < mid)] = 1.0
            v[(x >= mid) & (x < b)] = -1.0
            basis[:, idx] = v
            idx += 1
        j += 1
    return basis

def fourier_basis(x: np.ndarray, K: int) -> np.ndarray:
    basis = np.zeros((x.size, K))
    basis[:, 0] = 1.0
    idx = 1
    m = 1
    while idx < K:
        basis[:, idx] = np.cos(m * np.pi * x)
        idx += 1
        if idx < K:
            basis[:, idx] = np.sin(m * np.pi * x)
            idx += 1
        m += 1
    return basis

def _stack_p(theta):
    return 1.0 + torch.exp(theta)

def _stack_predict(x: torch.Tensor, a, w, b, theta, d) -> torch.Tensor:
    p = _stack_p(theta).unsqueeze(0)
    z = w.unsqueeze(0) * x.unsqueeze(1) + b.unsqueeze(0)
    phi = z * (z.square() + EPS ** 2) ** ((p - 2.0) / 2.0)
    return (phi * a.unsqueeze(0)).sum(dim=1) + d

def jp_fit(x_train: np.ndarray, y_train: np.ndarray, x_dense: np.ndarray, *, return_details=False, observer=None):
    rng = np.random.default_rng(JP_SEED)
    x_tr = torch.tensor(x_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    x_de = torch.tensor(x_dense, dtype=torch.float64)
    best_loss = float('inf')
    best_pred_tr = None
    best_pred_de = None
    best_p = None
    best_gains = (float('nan'), float('nan'))
    details = {'trials': [], 'selection': 'lowest noisy-training mean squared error', 'duality_map': 'c1', 'epsilon': EPS}
    for trial in range(JP_N_INITS):
        if trial == 0:
            centres = np.linspace(-0.8, 0.8, JP_K_EDGES)
            w0 = np.full(JP_K_EDGES, 8.0)
            b0 = -centres * w0
            a0 = rng.standard_normal(JP_K_EDGES) * 0.2
            theta0 = np.full(JP_K_EDGES, np.log(0.01))
        else:
            w0 = np.exp(rng.standard_normal(JP_K_EDGES) * 0.6 + 1.0)
            b0 = rng.standard_normal(JP_K_EDGES) * 1.5
            a0 = rng.standard_normal(JP_K_EDGES) * 0.4
            theta0 = rng.standard_normal(JP_K_EDGES) * 1.2
        d0 = np.array([float(np.mean(y_train))])
        a = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
        theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        d = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
        parameters = [a, w, b, theta, d]
        count = sum((t.numel() for t in parameters if t.requires_grad))
        if count != 4 * JP_K_EDGES + 1:
            raise ValueError('Unexpected fitted parameter count')
        started = time.perf_counter()
        executed, failure = (0, None)
        opt = torch.optim.LBFGS([a, w, b, theta, d], lr=0.5, max_iter=20, line_search_fn='strong_wolfe', tolerance_grad=1e-09, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            pred = _stack_predict(x_tr, a, w, b, theta, d)
            loss = ((pred - y_tr) ** 2).mean()
            loss.backward()
            return loss
        prev = float('inf')
        for _ in range(JP_N_LBFGS_OUTER):
            try:
                cur = float(opt.step(closure))
                executed += 1
            except RuntimeError as error:
                failure = str(error)
                break
            if abs(prev - cur) < 1e-12:
                break
            prev = cur
        with torch.no_grad():
            pred_tr = _stack_predict(x_tr, a, w, b, theta, d)
            loss = ((pred_tr - y_tr) ** 2).mean().item()
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_pred_tr = pred_tr.detach().numpy()
                best_pred_de = _stack_predict(x_de, a, w, b, theta, d).detach().numpy()
                best_p = _stack_p(theta).detach().numpy().tolist()
                best_gains = (float(a.abs().max()), float(w.abs().max()))
                details['selected_trial'] = trial
                details['parameters'] = {name: t.detach().numpy().tolist() for name, t in zip(('a', 'w', 'b', 'theta', 'd'), parameters)}
                details['n_trainable_parameters'] = count
            record = {'trial': trial, 'train_mse': loss if np.isfinite(loss) else None, 'outer_steps': executed, 'failure': failure, 'time_s': time.perf_counter() - started}
            details['trials'].append(record)
            if observer is not None:
                observer(record)
    if best_pred_tr is None or not np.isfinite(best_pred_de).all():
        raise RuntimeError('No finite fitted power response')
    result = (best_pred_tr, best_pred_de, best_p)
    return result + (details,) if return_details else result
