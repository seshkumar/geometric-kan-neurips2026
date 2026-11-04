from __future__ import annotations
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import numpy as np
from src import neurips_figure1 as fitting
from src.neurips_normalization import standard_deviation
ROOT = Path(__file__).resolve().parents[2]
MODELS = ('spline', 'cheby', 'rbf', 'haar', 'fourier', 'lp')
LABELS = {'spline': 'B-spline', 'cheby': 'Chebyshev', 'rbf': 'Gaussian RBF', 'haar': 'Wavelet (Haar)', 'fourier': 'Fourier', 'lp': '$\\ell^p$-KAN'}
BASES = dict(zip(MODELS[:-1], (fitting.bspline_basis, fitting.chebyshev_basis, fitting.rbf_basis, fitting.haar_basis, fitting.fourier_basis)))
TARGETS = {'step': fitting.heaviside_step, 'staircase': fitting.five_level_step}
DEFAULT = dict(metric_ddof=1, parameters=21, n_train=2000, n_eval=4000, sigma=0.03, data_seeds=list(range(1729, 1734)), init_seed=1729, edges=5, restarts=18, lbfgs_outer=100, lbfgs_inner=20, epsilon=1e-08, evaluation='fixed midpoint grid; clean targets', selection='noisy-training MSE only; no clean-target selection', noise_stream='default_rng(data_seed); step then staircase', comparison='equal trainable parameter count; not equal computation', figure_curve_seed=1729, figure_band='pointwise minimum and maximum across noisy refits; not a confidence interval', scope='two illustrative monotone targets; no universal or adversarial robustness claim')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def shape_metrics(x, truth, pred):
    x, truth, pred = map(np.asarray, (x, truth, pred))
    if x.ndim != 1 or truth.shape != x.shape or pred.shape != x.shape:
        raise ValueError('Expected aligned scalar-function samples')
    if not np.isfinite(pred).all() or not np.all(np.diff(x) > 0):
        raise ValueError('Predictions must be finite on an increasing grid')
    scale = float(np.std(truth, ddof=1))
    if scale <= 0:
        raise ValueError('Nonconstant target required')
    return dict(clean_grid_nrmse=float(np.sqrt(np.mean((pred - truth) ** 2)) / scale), negative_variation=float(np.maximum(-np.diff(pred), 0).sum()), range_overshoot=float(max(0.0, pred.max() - truth.max(), truth.min() - pred.min())))

def prepare(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Fresh study directory required: {out}')
    out.mkdir(parents=True)
    sources = ['src/neurips_figure1.py', 'src/reporting/neurips_figure1_study.py', 'experiments/neurips_figure1_matched.py']
    config = DEFAULT | {'source_sha256': {p: digest(ROOT / p) for p in sources}}
    write_json(out / 'config.json', config)
    x = np.linspace(-1.0, 1.0, config['n_train'])
    xe = -1.0 + (np.arange(config['n_eval']) + 0.5) * 2 / config['n_eval']
    for seed in config['data_seeds']:
        rng = np.random.default_rng(seed)
        for target, fn in TARGETS.items():
            path = out / f'{target}_{seed}'
            path.mkdir()
            clean = fn(x)
            noise = config['sigma'] * rng.standard_normal(x.size)
            np.savez_compressed(path / 'inputs.npz', x=x, y=clean + noise, clean_train=clean, noise=noise, x_eval=xe, truth=fn(xe))
    return config

def fit_case(task):
    out, target, seed = task
    import torch
    torch.set_num_threads(1)
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    for name, expected in config['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Source changed during study: ' + name)
    if (fitting.JP_K_EDGES, fitting.JP_N_INITS, fitting.JP_N_LBFGS_OUTER, fitting.JP_SEED) != (config['edges'], config['restarts'], config['lbfgs_outer'], config['init_seed']):
        raise ValueError('Fitter settings differ from declaration')
    case = out / f'{target}_{seed}'
    data = np.load(case / 'inputs.npz')
    x, y, xe, truth = (data[k] for k in ('x', 'y', 'x_eval', 'truth'))
    for model in MODELS:
        if (case / f'{model}.json').exists():
            continue
        started = time.perf_counter()
        if model == 'lp':

            def observe(r):
                if r['trial'] in (0, 5, 11, 17):
                    print(json.dumps(dict(target=target, seed=seed, model=model, restart=r['trial'] + 1, train_mse=r['train_mse'])), flush=True)
            pred, grid, p, state = fitting.jp_fit(x, y, xe, return_details=True, observer=observe)
            count = state['n_trainable_parameters']
        else:
            design = BASES[model](x, config['parameters'])
            coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
            count = coefficients.size
            if rank != count:
                raise ValueError('Rank-deficient fixed basis: ' + model)
            pred = design @ coefficients
            grid = BASES[model](xe, count) @ coefficients
            p = []
            state = dict(coefficients=coefficients.tolist(), rank=int(rank), singular_values=singular.tolist())
        fit_time = time.perf_counter() - started
        if count != config['parameters']:
            raise ValueError('Parameter mismatch: ' + model)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite fit: ' + model)
        np.savez_compressed(case / f'{model}_predictions.npz', train=pred, evaluation=grid)
        record = dict(target=target, seed=seed, model=model, n_trainable_parameters=int(count), training_time_s=fit_time, learned_p=p, metric_ddof=config.get('metric_ddof', 0), train_noisy_nrmse=float(np.sqrt(np.mean((pred - y) ** 2)) / standard_deviation(y, ddof=config.get('metric_ddof', 0))), metrics=shape_metrics(xe, truth, grid), state=state, input_sha256=digest(case / 'inputs.npz'), predictions_sha256=digest(case / f'{model}_predictions.npz'))
        write_json(case / f'{model}.json', record)
        print(json.dumps({k: record[k] for k in ('target', 'seed', 'model', 'n_trainable_parameters', 'train_noisy_nrmse', 'training_time_s')}), flush=True)
    return dict(target=target, seed=seed, complete=True)

def run(out, workers=5):
    out = Path(out)
    config = json.loads((out / 'config.json').read_text()) if out.exists() else prepare(out)
    tasks = [(str(out.resolve()), t, s) for t in TARGETS for s in config['data_seeds']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(fit_case, t) for t in tasks]):
            f.result()

def export(out):
    out = Path(out)
    config = json.loads((out / 'config.json').read_text())
    rows = []
    summaries = {}
    inputs = []
    for target in TARGETS:
        summaries[target] = {}
        for model in MODELS:
            selected = []
            curves = []
            for seed in config['data_seeds']:
                case = out / f'{target}_{seed}'
                path = case / f'{model}.json'
                r = json.loads(path.read_text())
                if r['n_trainable_parameters'] != 21:
                    raise ValueError('Unmatched fit')
                if digest(case / 'inputs.npz') != r['input_sha256'] or digest(case / f'{model}_predictions.npz') != r['predictions_sha256']:
                    raise ValueError('Changed retained input or prediction')
                rows.append(r)
                selected.append(r)
                curves.append(np.load(case / f'{model}_predictions.npz')['evaluation'])
                inputs.append(dict(path=str(path.relative_to(out)), sha256=digest(path)))
            metrics = {k: float(np.median([r['metrics'][k] for r in selected])) for k in selected[0]['metrics']}
            curves = np.asarray(curves)
            summaries[target][model] = dict(parameters=21, train_noisy_nrmse=float(np.median([r['train_noisy_nrmse'] for r in selected])), **metrics, refit_variation_rms=float(np.sqrt(np.mean(np.var(curves, axis=0, ddof=1)))), training_time_s=float(sum((r['training_time_s'] for r in selected))))
    summary = dict(protocol=config, metrics=summaries, inputs=inputs, counts_match=all((r['n_trainable_parameters'] == 21 for r in rows)), training_nrmse='against noisy training labels; ' + ('sample' if config.get('metric_ddof', 0) == 1 else 'population') + ' standard deviation', clean_nrmse='against clean midpoint-grid targets; sample standard deviation', shape_scope='negative variation and range overshoot concern these monotone targets only', robustness_scope='refit variation across independent label-noise draws with fixed initialisation stream')
    write_json(out / 'summary.json', summary)
    (out / 'records.jsonl').write_text(''.join((json.dumps(r, allow_nan=False) + '\n' for r in rows)))
    plot(out, summary)
    lines = ['# Matched-parameter motivation figure', '', '## Protocol', '', 'The declared protocol and full fit states are retained in `config.json` and the per-fit records. Every model has the same number of trainable scalar parameters. Fixed feature definitions are not counted as fitted parameters. The power stack uses the existing continuously differentiable response and a fixed restart stream; restart selection uses only noisy-training mean squared error. This is not an equal-compute comparison.', '', '## Metrics', '', '| Target | Model | Parameters | Noisy train NRMSE | Clean grid NRMSE | Negative variation | Range overshoot | Refit variation RMS |', '|:---|:---|---:|---:|---:|---:|---:|---:|']
    for t in TARGETS:
        for m in MODELS:
            r = summaries[t][m]
            lines.append(f"| {t} | {m} | {r['parameters']} | {r['train_noisy_nrmse']:.5f} | {r['clean_grid_nrmse']:.5f} | {r['negative_variation']:.5f} | {r['range_overshoot']:.5f} | {r['refit_variation_rms']:.5f} |")
    lines += ['', 'NRMSE means normalised root mean squared error. RMS means root mean square. Error and shape metrics are medians across noisy refits. Refit variation is the root mean of pointwise sample variances, in target units, with the same optimiser initialisation stream across noise draws. Neither the bands nor the seed medians are confidence intervals. Lower negative variation means fewer downward excursions for these nondecreasing targets. A sharp transition does not imply global input-perturbation robustness.', '', '## Reproduction', '', '```bash', 'python experiments/neurips_figure1_matched.py run --out NEW_DIRECTORY --workers 5', 'python experiments/neurips_figure1_matched.py export --out NEW_DIRECTORY', '```']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary

def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.85), sharex=True, sharey=True)
    low, high = (-0.5, 0.5)
    for i, t in enumerate(TARGETS):
        d = np.load(out / f"{t}_{summary['protocol']['figure_curve_seed']}" / 'inputs.npz')
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            curves = np.array([np.load(out / f'{t}_{s}' / f'{m}_predictions.npz')['evaluation'] for s in summary['protocol']['data_seeds']])
            low = min(low, float(curves.min()))
            high = max(high, float(curves.max()))
            ax.scatter(d['x'][::20], d['y'][::20], s=1.0, c='#aaaaaa', alpha=0.4, rasterized=True)
            ax.fill_between(d['x_eval'], curves.min(0), curves.max(0), color='#2471a3', alpha=0.22, lw=0)
            ax.plot(d['x_eval'], d['truth'], c='black', ls='--', lw=0.8)
            ax.plot(d['x_eval'], curves[0], c='#2471a3', lw=0.9)
            r = summary['metrics'][t][m]
            ax.text(0.03, 0.05, f"NRMSE {r['train_noisy_nrmse']:.3f}", transform=ax.transAxes, fontsize=6, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.7))
            if i == 0:
                ax.set_title(LABELS[m] + '\n21 parameters', fontsize=7)
            if j == 0:
                ax.set_ylabel('Step' if i == 0 else 'Staircase')
            if i == 1:
                ax.set_xlabel('$x$', labelpad=0)
            ax.tick_params(labelsize=6, pad=1)
    pad = 0.07 * (high - low)
    for ax in axes.flat:
        ax.set_ylim(low - pad, high + pad)
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    fig.savefig(out / 'figure1_matched.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out / 'figure1_matched.png', dpi=220)
    plt.close(fig)
