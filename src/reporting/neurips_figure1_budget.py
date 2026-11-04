from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import numpy as np
from src.reporting import neurips_figure1_study as study
BUDGETS = (13, 21, 41, 125)
SEEDS = (1729,)

def harmonize_metrics(root):
    directory = Path(root) / 'experiments/results/neurips_figure1_matched'
    config = json.loads((directory / 'config.json').read_text())
    changes = []
    for target in study.TARGETS:
        for seed in config['data_seeds']:
            case = directory / f'{target}_{seed}'
            with np.load(case / 'inputs.npz') as data:
                y = data['y']
            for model in study.MODELS:
                path = case / f'{model}.json'
                record = json.loads(path.read_text())
                prediction = case / f'{model}_predictions.npz'
                if study.digest(prediction) != record['predictions_sha256'] or study.digest(case / 'inputs.npz') != record['input_sha256']:
                    raise ValueError('Changed motivation-study input or prediction')
                if record.get('metric_ddof', 0) == 1:
                    continue
                with np.load(prediction) as saved:
                    pred = saved['train']
                changes.append(dict(path=str(path.relative_to(root)), before_sha256=study.digest(path), previous_train_noisy_nrmse=record['train_noisy_nrmse']))
                record['train_noisy_nrmse'] = float(np.sqrt(np.mean((pred - y) ** 2)) / np.std(y, ddof=1))
                record['metric_ddof'] = 1
                study.write_json(path, record)
    config['metric_ddof'] = 1
    study.write_json(directory / 'config.json', config)
    if changes:
        study.write_json(directory / 'metric_conversion.json', dict(command='python experiments/neurips_figure1_budget.py run', changes=changes, fitted_parameters_and_predictions_unchanged=True, metric_ddof=1))
    study.export(directory)

def fit(task):
    path, target, seed = task
    config = json.loads((Path(path) / 'config.json').read_text())
    study.fitting.JP_K_EDGES = config['edges']
    return study.fit_case(task)

def run(root, workers=3):
    root = Path(root)
    harmonize_metrics(root)
    out = root / 'experiments/results/neurips_figure1_budget'
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'experiments/results/neurips_figure1_matched'
    tasks = []
    for budget in BUDGETS:
        directory = out / f'parameters_{budget}'
        if not directory.exists():
            study.prepare(directory, budget, SEEDS)
            if budget == 21:
                for target in study.TARGETS:
                    for seed in SEEDS:
                        case = source / f'{target}_{seed}'
                        for path in case.iterdir():
                            if path.suffix in ('.json', '.npz'):
                                shutil.copy2(path, directory / case.name / path.name)
        for target in study.TARGETS:
            for seed in SEEDS:
                if not all(((directory / f'{target}_{seed}' / f'{m}.json').exists() for m in study.MODELS)):
                    tasks.append((str(directory), target, seed))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed((pool.submit(fit, task) for task in tasks)):
            future.result()
    return export(root)

def export(root):
    root = Path(root)
    out = root / 'experiments/results/neurips_figure1_budget'
    rows = []
    inputs = []
    for budget in BUDGETS:
        directory = out / f'parameters_{budget}'
        config = json.loads((directory / 'config.json').read_text())
        inputs.append(dict(path=str((directory / 'config.json').relative_to(root)), sha256=study.digest(directory / 'config.json')))
        for target in study.TARGETS:
            for model in study.MODELS:
                values = []
                for seed in SEEDS:
                    case = directory / f'{target}_{seed}'
                    path = case / f'{model}.json'
                    record = json.loads(path.read_text())
                    count = sum((np.asarray(v).size for v in record['state']['parameters'].values())) if model == 'lp' else len(record['state']['coefficients'])
                    if count != budget or record['n_trainable_parameters'] != budget or record['metric_ddof'] != 1:
                        raise ValueError('Budget or SD convention mismatch')
                    for name, key in [('inputs.npz', 'input_sha256'), (f'{model}_predictions.npz', 'predictions_sha256')]:
                        if study.digest(case / name) != record[key]:
                            raise ValueError('Changed retained arrays')
                    inputs.append(dict(path=str(path.relative_to(root)), sha256=study.digest(path)))
                    values.append(record['train_noisy_nrmse'])
                rows.append(dict(target=target, model=model, parameters=budget, nrmse=float(np.median(values))))
    result = dict(command='python experiments/neurips_figure1_budget.py run', budgets=BUDGETS, seeds=SEEDS, selection='Existing noisy-training MSE restart selection; clean targets never select fits', metric='Noisy-training NRMSE with sample SD', equal_compute=False, reused_budget_21=True, rows=rows, inputs=inputs)
    study.write_json(out / 'summary.json', result)
    lookup = {(r['target'], r['model'], r['parameters']): r['nrmse'] for r in rows}
    lines = ['\\begin{tabular}{llrrrr}', '\\toprule', 'Target & Response & ' + ' & '.join((str(b) for b in BUDGETS)) + ' \\\\', '\\midrule']
    for target in study.TARGETS:
        for model in study.MODELS:
            lines.append(('Step' if target == 'step' else 'Staircase') + ' & ' + study.LABELS[model] + ' & ' + ' & '.join((f'{lookup[target, model, b]:.3f}' for b in BUDGETS)) + ' \\\\')
        if target == 'step':
            lines.append('\\midrule')
    text = '\n'.join(lines + ['\\bottomrule\\end{tabular}']) + '\n'
    (out / 'budget_sweep.tex').write_text(text)
    doc = root / 'docs/neurips2026/camera_ready_tables'
    if doc.exists():
        (doc / 'figure1_budget.tex').write_text(text)
    return result
