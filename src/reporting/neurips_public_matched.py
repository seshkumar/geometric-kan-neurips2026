from pathlib import Path
import json
import numpy as np
from src.result_integrity import read_records, grid_inventory
from src.reporting.neurips_camera_ready import describe, paired_ratios, matched_error_table, matched_per_equation_table
from src.reporting.neurips_public_formatting import C1, ROSTER, LABEL, CELLS, PAIR_TARGETS, EXTRAP_NEURIPS, paired_table, curves
FIELDS = ('eq_id', 'model', 'seed', 'sigma', 'n_train')
EXTRAP_MODELS = C1 + ['banach', 'lp', 'tanh', 'spline', 'cheby', 'rbf', 'fourier']
EXTRAP_SLOTS = [('banach_c1', 'banach'), ('lp_c1', 'lp'), None, None, ('banach_c1', 'spline'), ('lp_c1', 'spline'), ('banach_c1', 'tanh'), ('lp_c1', 'tanh')]
GRIDS = {'baseline': ([0.0], [500]), 'noisy': ([0.0, 0.1, 0.3, 0.5, 1.0], [500]), 'small_sample': ([0.0], [50, 100, 200, 500, 1000])}

def seed_median_matrix(rows, models, equations, keep, field='eq_id', seeds=None):
    bucket = {}
    for row in rows:
        if keep(row) and row['model'] in models:
            key = (row[field], row['model'])
            by_seed = bucket.setdefault(key, {})
            if row['seed'] in by_seed:
                raise ValueError('Duplicate seed in matched cell')
            by_seed[row['seed']] = row['nrmse']
    for eq in equations:
        for model in models:
            if set(bucket.get((eq, model), {})) != set(seeds):
                raise ValueError(f'Incomplete matched cell: {eq}/{model}')
    return np.asarray([[np.median(list(bucket[eq, model].values())) for model in models] for eq in equations])

def extrapolation_pairs(matrix, draws=100000, seed=20260929):
    comparisons = []
    ones = np.ones(len(matrix))
    for index, pair in enumerate(EXTRAP_SLOTS):
        if pair is None:
            comparisons.append((f'__reserved_{index}', ones, ones))
        else:
            a, b = pair
            comparisons.append((f'{a}/{b}', matrix[:, EXTRAP_MODELS.index(a)], matrix[:, EXTRAP_MODELS.index(b)]))
    return [row for row in paired_ratios(comparisons, draws=draws, seed=seed) if not row['comparison'].startswith('__reserved_')]

def export(input_dir, output_dir):
    input_dir, output_dir = (Path(input_dir), Path(output_dir))
    configuration = json.loads((input_dir / 'configuration.json').read_text())
    equations, seeds = (configuration['equations'], configuration['seeds'])
    if len(equations) != 50 or len(set(equations)) != 50 or seeds != list(range(1729, 1734)):
        raise ValueError('Unexpected matched equation or seed configuration')
    loaded, inputs, inventories = ({}, [], {})
    for condition, (sigmas, sizes) in GRIDS.items():
        rows, files = read_records([input_dir / (condition + '.jsonl')])
        expected = [dict(eq_id=e, model=m, seed=s, sigma=sig, n_train=n) for e in equations for m in ROSTER for s in seeds for sig in sigmas for n in sizes]
        check = grid_inventory(rows, FIELDS, expected)
        if not check['complete_and_valid'] or any((r.get('diverged') or (r['model'] in C1 and r.get('duality_map') != 'c1') for r in rows)):
            raise ValueError(f'Invalid public matched grid: {condition}: {check}')
        loaded[condition] = rows
        inventories[condition] = check
        inputs += files
    rows, files = read_records([input_dir / 'extrapolation.jsonl'])
    expected = [dict(eq=e, model=m, seed=s) for e in equations for m in EXTRAP_MODELS for s in range(5)]
    check = grid_inventory(rows, ('eq', 'model', 'seed'), expected)
    if not check['complete_and_valid'] or any((r.get('diverged') for r in rows)):
        raise ValueError(f'Invalid public extrapolation grid: {check}')
    inputs += files
    inventories['extrapolation'] = check
    summary = dict(command='python experiments/aggregate_neurips_public.py', roster=ROSTER, cells={}, paired={})
    comparisons = []
    for label, condition, keep in CELLS:
        x = seed_median_matrix(loaded[condition], ROSTER, equations, keep, seeds=seeds)
        d = describe(x)
        summary['cells'][label] = {m: {key: d[key][i] for key in ('median', 'rank', 'wins')} for i, m in enumerate(ROSTER)}
        for geometry in C1:
            for target in PAIR_TARGETS:
                comparisons.append((f'{label}:{geometry}/{target}', x[:, ROSTER.index(geometry)], x[:, ROSTER.index(target)]))
    summary['paired'] = paired_ratios(comparisons)
    xe = seed_median_matrix(rows, EXTRAP_MODELS, equations, lambda r: True, field='eq', seeds=range(5))
    summary['extrapolation'] = {'median': {m: float(np.median(xe[:, i])) for i, m in enumerate(EXTRAP_MODELS)}, 'paired': extrapolation_pairs(xe), 'family_slots': len(EXTRAP_SLOTS), 'reserved_slots': [2, 3], 'reservation_reason': 'Preserve the original correction and bootstrap draw positions; reserved slots have no reported estimate or experimental assertion.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'tab_matched_nrmse.tex').write_text(matched_error_table(summary) + '\n')
    clean = seed_median_matrix(loaded['baseline'], ROSTER, equations, lambda r: True, seeds=seeds)
    (output_dir / 'tab_matched_per_equation.tex').write_text(matched_per_equation_table(clean, equations) + '\n')
    (output_dir / 'inventory.json').write_text(json.dumps(dict(inputs=inputs, grids=inventories), indent=2) + '\n')
    curves(summary, output_dir)
    paired_table(summary, output_dir)
    cols = [c[0] for c in CELLS]
    head = ['clean'] + [c.split('=')[1] for c in cols[1:5]] + [c.split('=')[1] for c in cols[5:]]
    best = {c: min((summary['cells'][c][m]['rank'] for m in ROSTER)) for c in cols}
    tex = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule', ' & & \\multicolumn{4}{c}{noise $\\sigma$ ($n=500$)} & \\multicolumn{4}{c}{clean training size $n$} \\\\', '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}', 'model & ' + ' & '.join(head) + ' \\\\', '\\midrule']
    for model in ROSTER:
        cells = []
        for cell in cols:
            value = summary['cells'][cell][model]['rank']
            cells.append(f'\\textbf{{{value:.1f}}}' if abs(value - best[cell]) < 1e-09 else f'{value:.1f}')
        tex.append(f'{LABEL[model]} & ' + ' & '.join(cells) + ' \\\\')
        if model == 'lp_c1':
            tex.append('\\midrule')
    tex += ['\\bottomrule', '\\end{tabular}']
    (output_dir / 'tab_matched_ranks.tex').write_text('\n'.join(tex) + '\n')
    med = summary['extrapolation']['median']
    lowest = min((med[m] for m in EXTRAP_NEURIPS))
    tex = ['\\begin{tabular}{lc}', '\\toprule', 'model & out-of-range NRMSE \\\\', '\\midrule'] + [f'{LABEL[m]} & ' + (f'\\textbf{{{med[m]:.3f}}}' if med[m] == lowest else f'{med[m]:.3f}') + ' \\\\' for m in EXTRAP_NEURIPS] + ['\\bottomrule', '\\end{tabular}']
    (output_dir / 'tab_extrap_neurips.tex').write_text('\n'.join(tex) + '\n')
    return summary
