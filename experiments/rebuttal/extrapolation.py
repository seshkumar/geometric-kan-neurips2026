import numpy as np, torch

def gen_extrap(eq, seed, n_train=500, n_test=1000, inner=0.6, reach=0.3):
    name, _, nv, _, formula, var_ranges = eq
    rng = np.random.default_rng(seed)

    def sample(n, lo_f, hi_f):
        cols = []
        for lo, hi in var_ranges:
            span = hi - lo
            cols.append(rng.uniform(lo + lo_f * span, lo + hi_f * span, size=n))
        return np.stack(cols, 1).astype(np.float32)
    m = (1 - inner) / 2
    Xtr = sample(int(n_train * 1.6), m, 1 - m)
    ytr = formula(Xtr).astype(np.float32)
    ok = np.isfinite(ytr) & (np.abs(ytr) < 1000000.0)
    Xtr, ytr = (Xtr[ok][:n_train], ytr[ok][:n_train])
    Xte = sample(int(n_test * 1.6), 1 - m, 1 - m + reach)
    yte = formula(Xte).astype(np.float32)
    ok = np.isfinite(yte) & (np.abs(yte) < 1000000.0)
    Xte, yte = (Xte[ok][:n_test], yte[ok][:n_test])
    mu, sd = (ytr.mean(), ytr.std() or 1.0)
    return (torch.tensor(Xtr), torch.tensor((ytr - mu) / sd), torch.tensor(Xte), torch.tensor((yte - mu) / sd))
