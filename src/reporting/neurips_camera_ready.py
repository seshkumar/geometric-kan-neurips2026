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

def summary_cells(summary, model_index):
    cells = []
    for field, spec, optimum in (('median', '.3f', min), ('wins', 'g', max), ('rank', '.2f', min)):
        value = summary[field][model_index]
        if value is None:
            cells.append('--')
            continue
        display = format(value, spec)
        if value == optimum((v for v in summary[field] if v is not None)):
            display = '\\textbf{' + display + '}'
        cells.append(display)
    return cells

def _aligned_clean_summaries(clean, models, core=None):
    core = core or dict(models=clean['models'], **clean['core'])
    summaries = []
    for source in (dict(models=clean['models'], **clean['full']), core):
        summaries.append({field: [source[field][source['models'].index(model)] if model in source['models'] else None for model in models] for field in ('median', 'wins', 'rank')})
    return summaries

def clean_noise_table(clean, noise, labels, core=None):
    models = noise['models']
    summaries = _aligned_clean_summaries(clean, models, core)
    values = np.asarray(noise['values'])
    lines = ['\\begin{tabular}{l' + 'r' * (6 + len(noise['columns'])) + '}', '\\toprule', '& \\multicolumn{3}{c}{Full benchmark, clean} & \\multicolumn{3}{c}{Core-18, clean}' + ' & \\multicolumn{' + str(len(noise['columns'])) + '}{c}{Core-18, label noise $\\sigma$} \\\\', '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-' + str(7 + len(noise['columns'])) + '}', 'Model & NRMSE & Wins & Rank & NRMSE & Wins & Rank & ' + ' & '.join(noise['columns']) + ' \\\\', '\\midrule']
    for index, model in enumerate(models):
        cells = [cell for summary in summaries for cell in summary_cells(summary, index)]
        for column, value in enumerate(values[index]):
            display = f'{value:.3f}'
            if value == values[:, column].min():
                display = '\\textbf{' + display + '}'
            cells.append(display)
        lines.append(labels[model] + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule', '\\end{tabular}'])

def clean_noise_scarcity_table(clean, noise, scarcity, labels, core):
    models = noise['models']
    if scarcity['models'] != models:
        raise ValueError('Noise and scarcity model order must match')
    summaries = _aligned_clean_summaries(clean, models, core)
    lines = ['\\textbf{(a) Clean fitting}\\par\\smallskip', '\\begin{tabular}{lrrrrrr}', '\\toprule', '& \\multicolumn{3}{c}{Full benchmark} & \\multicolumn{3}{c}{Core-18} \\\\', '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}', 'Model & NRMSE & Wins & Rank & NRMSE & Wins & Rank \\\\', '\\midrule']
    for index, model in enumerate(models):
        cells = [cell for summary in summaries for cell in summary_cells(summary, index)]
        lines.append(labels[model] + ' & ' + ' & '.join(cells) + ' \\\\')
    values = np.concatenate([noise['values'], scarcity['values']], axis=1)
    n_noise, n_scarcity = (len(noise['columns']), len(scarcity['columns']))
    lines += ['\\bottomrule', '\\end{tabular}', '\\par\\medskip', '\\textbf{(b) Core-18 noise and noisy scarcity: median NRMSE}\\par\\smallskip', '\\begin{tabular}{l' + 'r' * (n_noise + n_scarcity) + '}', '\\toprule', '& \\multicolumn{' + str(n_noise) + '}{c}{Label noise $\\sigma$ ($n=500$)}' + ' & \\multicolumn{' + str(n_scarcity) + '}{c}{Training-pool size $n$ ($\\sigma=0.3$)} \\\\', '\\cmidrule(lr){2-' + str(1 + n_noise) + '}\\cmidrule(lr){' + str(2 + n_noise) + '-' + str(1 + n_noise + n_scarcity) + '}', 'Model & ' + ' & '.join(noise['columns'] + scarcity['columns']) + ' \\\\', '\\midrule']
    for index, model in enumerate(models):
        cells = []
        for column, value in enumerate(values[index]):
            display = f'{value:.3f}'
            if value == values[:, column].min():
                display = '\\textbf{' + display + '}'
            cells.append(display)
        lines.append(labels[model] + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule', '\\end{tabular}'])

def matched_error_table(summary):
    from src.reporting.neurips_public_formatting import LABEL, ROSTER
    models = summary['roster']
    if models != ROSTER:
        raise ValueError('Unexpected matched model roster')
    keys = ['clean'] + [f'sigma={s}' for s in (0.1, 0.3, 0.5, 1.0)] + [f'n={n}' for n in (50, 100, 200, 1000)]
    for key in keys:
        if set(summary['cells'][key]) != set(models) or not np.isclose(sum((summary['cells'][key][m]['wins'] for m in models)), 50.0, rtol=0, atol=1e-12):
            raise ValueError('Expected full 50-target summaries for every displayed condition')
    values = np.array([[summary['cells'][key][model]['median'] for key in keys] for model in models])
    ranks = [summary['cells']['clean'][model]['rank'] for model in models]
    wins = [summary['cells']['clean'][model]['wins'] for model in models]
    lines = ['\\begin{tabular}{lrrrrrrrrrrr}', '\\toprule', '& \\multicolumn{3}{c}{Clean} & \\multicolumn{4}{c}{Label noise $\\sigma$ ($n=500$)} & \\multicolumn{4}{c}{Clean training size $n$} \\\\', '\\cmidrule(lr){2-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}', 'Model & NRMSE & Wins & Rank & 0.1 & 0.3 & 0.5 & 1.0 & 50 & 100 & 200 & 1000 \\\\', '\\midrule']
    labels = dict(LABEL, spline_lip='Spline (Adam, Lip.-reg.)', pow='Learned power', adaslope='Adaptive-slope tanh', mixture='Activation mixture', rational="Rational/Pad\\'e")
    for index, model in enumerate(models):
        cells = []
        for column, value in enumerate(values[index]):
            cell = f'{value:.3f}'
            cells.append('\\textbf{' + cell + '}' if value == values[:, column].min() else cell)
        rank = f'{ranks[index]:.2f}'
        win = f'{wins[index]:g}'
        cells.insert(1, '\\textbf{' + rank + '}' if ranks[index] == min(ranks) else rank)
        cells.insert(1, '\\textbf{' + win + '}' if wins[index] == max(wins) else win)
        lines.append(labels[model] + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule', '\\end{tabular}'])

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
