import hashlib
import json
from pathlib import Path
import numpy as np
from src.reporting.neurips_camera_ready import describe
from src.reporting.neurips_public_formatting import ROSTER
from src.result_integrity import read_records
DATASET_LABELS = [('Geyser', 'geophysics'), ('Environment', 'environmental'), ('ESL', 'ordinal'), ('ERA', 'ordinal'), ('Cloud', 'meteorology'), ('Machine CPU', 'computing'), ('Sleuth case 2002', 'biology'), ('CPU', 'computing'), ('PM10', 'pollution'), ('NO$_2$', 'pollution'), ('LA data', 'epidemiology'), ('SWD', 'ordinal')]
MAIN_MODELS = ['spline', 'tanh', 'banach_c1', 'lp_c1', 'mlp']
HEADERS = {'spline': 'Spline', 'tanh': 'Tanh', 'banach_c1': 'Banach', 'lp_c1': '$\\ell^p$', 'mlp': 'MLP', 'pow': '\\shortstack{Learned\\\\power}', 'adaslope': '\\shortstack{Adaptive\\\\slope}', 'mixture': 'Mixture', 'rational': "Rational/Pad\\'e", 'spline_lip': '\\shortstack{Spline\\\\Lipschitz-reg.}', 'cheby': 'Chebyshev', 'rbf': 'RBF', 'fourier': 'Fourier', 'wavelet': 'Wavelet', 'jacobi': 'Jacobi'}

def export(root, run):
    root, run = (Path(root), Path(run))
    summary = json.loads((run / 'summary.json').read_text())
    if summary['models'] != ROSTER or not summary['inventory']['complete_and_valid']:
        raise ValueError('Validated complete 15-model summary required')
    for item in summary['inputs']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Stale PMLB summary input')
    rows, _ = read_records(run.glob('*/*.jsonl'))
    matrix = np.asarray(summary['values'])
    datasets = summary['datasets']
    if len(datasets) != len(DATASET_LABELS):
        raise ValueError('Unexpected dataset roster')
    full = describe(matrix)
    if full != summary['scorecard']:
        raise ValueError('Summary scorecard mismatch')
    metadata = []
    for dataset, (name, domain) in zip(datasets, DATASET_LABELS):
        cases = [r for r in rows if r['dataset'] == dataset]
        shapes = {(r['n_features'], r['n_train'] + r['n_test']) for r in cases}
        if len(shapes) != 1:
            raise ValueError('Dataset dimensions vary across fits')
        d, n = shapes.pop()
        metadata.append(dict(dataset=dataset, name=name, domain=domain, d=d, n=n))

    def render(models, meta=False, ranking=None):
        columns = [ROSTER.index(m) for m in models]
        values = matrix[:, columns]
        score = ranking if ranking is not None else describe(values)
        lines = ['\\begin{tabular}{l' + ('lrr' if meta else '') + 'r' * len(models) + '}', '\\toprule', 'Dataset' + (' & Domain & $d$ & $n$' if meta else '') + ' & ' + ' & '.join((HEADERS[m] for m in models)) + ' \\\\', '\\midrule']
        for info, row in zip(metadata, values):
            prefix = info['name'] + (f" & {info['domain']} & {info['d']} & {info['n']}" if meta else '')
            minimum = row.min()
            if ranking is not None:
                minimum = matrix[datasets.index(info['dataset'])].min()
            cells = ['\\textbf{' + f'{v:.3f}' + '}' if v == minimum else f'{v:.3f}' for v in row]
            lines.append(prefix + ' & ' + ' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')
        for label, field, spec in [('Median NRMSE', 'median', '.3f'), ('Wins', 'wins', 'g'), ('Mean rank', 'rank', '.2f')]:
            vals = score[field] if ranking is None else [score[field][j] for j in columns]
            best = (max if field == 'wins' else min)(score[field])
            cells = ['\\textbf{' + format(v, spec) + '}' if v == best else format(v, spec) for v in vals]
            lines.append(label + (' & \\multicolumn{3}{c}{}' if meta else '') + ' & ' + ' & '.join(cells) + ' \\\\')
        return '\n'.join(lines + ['\\bottomrule\\end{tabular}'])
    out = root / 'experiments/results/neurips_pmlb15_manuscript'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'main.tex').write_text(render(MAIN_MODELS, meta=True) + '\n')
    panels = []
    for letter, title, models in [('a', 'Proposed and adaptive responses', ROSTER[:7]), ('b', 'Fixed-basis responses and MLP', ROSTER[7:])]:
        panels += ['\\par\\smallskip\\textbf{(' + letter + ') ' + title + '}\\par\\smallskip', render(models, ranking=full)]
    (out / 'full.tex').write_text('\n'.join(panels) + '\n')
    lines = ['\\begin{tabular}{llrrl}', '\\toprule Dataset & Domain & $d$ & $n$ & Full PMLB identifier \\\\ \\midrule']
    for m in metadata:
        identifier = m['dataset'].replace('_', '\\_')
        lines.append(f"{m['name']} & {m['domain']} & {m['d']} & {m['n']} & \\texttt{{{identifier}}}" + ' \\\\')
    (out / 'datasets.tex').write_text('\n'.join(lines + ['\\bottomrule\\end{tabular}']) + '\n')
    evidence = dict(command='python experiments/export_neurips_pmlb_tables.py', inputs=summary['inputs'], run=str(run.relative_to(root)), metadata=metadata, main_models=MAIN_MODELS, main_scorecard=describe(matrix[:, [ROSTER.index(m) for m in MAIN_MODELS]]), full_models=ROSTER, full_scorecard=full, protocol=summary['protocol'], runtime_protocol=json.loads((run / 'protocol.json').read_text()))
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return out
