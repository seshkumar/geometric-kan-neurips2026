import numpy as np
from scipy.stats import rankdata
from src.result_integrity import grid_inventory

def replace_power_rows(rows, replacements, expected, models, experiment):
    fields = ('eq_id', 'model', 'seed', 'sigma', 'n_train')
    inventory = grid_inventory(replacements, fields, [r for r in expected if r['model'] in models])
    if not inventory['complete_and_valid'] or any((r.get('duality_map') != 'c1' or r.get('experiment') != experiment or r['nrmse'] < 0 for r in replacements)):
        raise ValueError(f'{experiment}: invalid C1 replacement: {inventory}')
    combined = [r for r in rows if r['model'] not in models] + replacements
    if not grid_inventory(combined, fields, expected)['complete_and_valid']:
        raise ValueError(f'{experiment}: invalid merged population')
    return (combined, inventory)

def describe(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or not x.size or (not np.isfinite(x).all()) or (x < 0).any():
        raise ValueError('Expected a finite, nonnegative equation-by-model matrix')
    best = x == x.min(axis=1, keepdims=True)
    return {'median': np.median(x, axis=0).tolist(), 'rank': rankdata(x, axis=1).mean(axis=0).tolist(), 'wins': (best / best.sum(axis=1, keepdims=True)).sum(axis=0).tolist()}

def paired_ratios(comparisons, draws=100000, seed=20260929):
    rng = np.random.default_rng(seed)
    result = []
    for label, numerator, denominator in comparisons:
        a, b = (np.asarray(numerator), np.asarray(denominator))
        if a.shape != b.shape or a.ndim != 1 or (not np.isfinite(a + b).all()) or (min(a.min(), b.min()) <= 0):
            raise ValueError('Paired positive finite errors required')
        logs = np.log(a / b)
        boot = np.exp(logs[rng.integers(len(a), size=(draws, len(a)))].mean(axis=1))
        alpha = 0.05 / len(comparisons)
        result.append({'comparison': label, 'equations': len(a), 'ratio': float(np.exp(logs.mean())), 'ci95': np.quantile(boot, [0.025, 0.975]).tolist(), 'family_ci': np.quantile(boot, [alpha / 2, 1 - alpha / 2]).tolist()})
    return result
