from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch
import numpy as np
import torch
from src import pmlb_data as pb
from src import matched_models as rb, fit_records as sr
from src.reporting.neurips_camera_ready import describe
from src.reporting.neurips_public_formatting import ROSTER
from src.result_integrity import read_records, grid_inventory, relative_input_records
from src.training import carve_validation_split
ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(1729, 1734))
STEPS = 5000

def protocol():
    return dict(study='neurips_pmlb15', datasets=pb.DATASETS, models=ROSTER, seeds=SEEDS, expected_fits=len(pb.DATASETS) * len(ROSTER) * len(SEEDS), data='Original PMLB loader: seeded 80/20 split, training-pool feature and target standardization', models_and_training='Table 2 builders and trainer; cubic G=3 Adam spline, not original pykan L-BFGS', architecture='KAN [d,K,1]: K=5 for d<=2, K=4 for d=3,4, otherwise K=3; MLP [d,50,1] for d<=3, otherwise [d,50,50,1]', steps=STEPS, learning_rate=0.01, validation_fraction=0.2, validation_seed=0, checkpoint_every=500, schedule='cosine', gradient_clip=10.0, metric='Original PMLB NRMSE: original-scale test RMSE / population std of original test targets; ddof=0; zero-denominator policy=1', selection='Training-derived validation only; test labels evaluated after fitting', historical_results='Original experiments/results/pmlb predates leakage fix b5ad5ab9; not reused', accuracy_evidence=False)

@contextmanager
def file_lock(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def prepare_data(dataset, cache):
    import pmlb
    cache = Path(cache).resolve()
    with file_lock(cache / dataset / '.prepare.lock'):
        pmlb.fetch_data(dataset, local_cache_dir=str(cache))
    path = cache / dataset / (dataset + '.tsv.gz')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data(dataset, seed, cache):
    import pmlb
    original = pmlb.fetch_data

    def cached(name, *args, **kwargs):
        kwargs['local_cache_dir'] = str(Path(cache).resolve())
        return original(name, *args, **kwargs)
    with patch.object(pmlb, 'fetch_data', cached):
        return pb.load_pmlb_dataset(dataset, seed)

def source_identity():
    paths = [Path(__file__), ROOT / 'experiments/neurips_pmlb15.py', ROOT / 'src/pmlb_data.py', ROOT / 'src/matched_models.py', ROOT / 'src/fit_records.py', ROOT / 'src/kan_layer.py', ROOT / 'src/training.py', ROOT / 'src/reporting/neurips_public_formatting.py']
    return dict(code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), source_fingerprint=sr.source_stamp(paths))

def fit_one(task):
    dataset, model_name, seed, cache, data_hash, stamp = task
    if model_name not in ROSTER:
        raise ValueError('Model outside the declared 15-model roster')
    torch.set_num_threads(1)
    np.random.seed(seed)
    x, y, xt, yt, _, target_scale, original_test = load_data(dataset, seed, cache)
    torch.manual_seed(seed)
    model = rb.MODEL_BUILDERS[model_name](x.shape[1])
    start = time.perf_counter()
    executed = sr.train_counted(model, x, y, steps=STEPS, lr=0.01, **rb.train_kwargs(model_name))
    elapsed = time.perf_counter() - start
    model.eval()
    xf, yf, xv, yv = carve_validation_split(x, y)
    xv, yv = (xf, yf) if xv is None else (xv, yv)
    with torch.no_grad():
        val_rmse = float(torch.sqrt(torch.mean((model(xv).reshape(-1) - yv.reshape(-1)) ** 2)))
    inference_start = time.perf_counter()
    with torch.no_grad():
        prediction = model(xt).reshape(-1)
        rmse_normalized = float(torch.sqrt(torch.mean((prediction - yt.reshape(-1)) ** 2)))
    inference = time.perf_counter() - inference_start
    denominator = float(np.std(original_test))
    if denominator < 1e-08:
        denominator = 1.0
    error = rmse_normalized * float(target_scale) / denominator
    bad = executed != STEPS or not (math.isfinite(error) and math.isfinite(val_rmse))
    return dict(dataset=dataset, model=model_name, seed=seed, n_features=int(x.shape[1]), n_train=len(x), n_test=len(xt), n_params=sum((p.numel() for p in model.parameters())), nrmse=error if math.isfinite(error) else None, train_time_s=elapsed, val_rmse=val_rmse if math.isfinite(val_rmse) else None, inference_time_s=inference, steps=STEPS, executed_steps=executed, diverged=bad, data_sha256=data_hash, metric_ddof=0, validation_only_selection=True, duality_map='c1' if model_name in ('banach_c1', 'lp_c1') else 'comparator', **stamp)

def run(dataset_index, output, cache, workers):
    dataset = pb.DATASETS[dataset_index]
    output, cache = (Path(output).resolve(), Path(cache).resolve())
    stamp = source_identity()
    manifest = protocol() | stamp | dict(environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pmlb=version('pmlb')))
    with file_lock(output / '.manifest.lock'):
        path = output / 'protocol.json'
        if path.exists():
            old = json.loads(path.read_text())
            for key in (*protocol(), 'source_fingerprint', 'environment'):
                if old[key] != manifest[key]:
                    raise ValueError('Existing run has different source/protocol/environment; use a new run ID')
        else:
            if any((p.name != '.manifest.lock' for p in output.iterdir())):
                raise ValueError('Existing output preserved; an empty run root or matching protocol manifest is required')
            sr.write_shard(path, [manifest], ('study',))
    data_hash = prepare_data(dataset, cache)
    directory = output / dataset
    with file_lock(directory / '.writer.lock', nonblocking=True):
        pending = []
        for model in ROSTER:
            for seed in SEEDS:
                path = directory / f'{model}_{seed}.jsonl'
                if path.exists():
                    rows, _ = read_records([path])
                    expected = [dict(dataset=dataset, model=model, seed=seed)]
                    check = grid_inventory(rows, ('dataset', 'model', 'seed'), expected)
                    if not check['complete_and_valid'] or len(rows) != 1 or any((rows[0].get(k) != v for k, v in dict(dataset=dataset, model=model, seed=seed, data_sha256=data_hash, source_fingerprint=stamp['source_fingerprint'], steps=STEPS, executed_steps=STEPS, diverged=False, metric_ddof=0, validation_only_selection=True).items())):
                        raise ValueError(f'Existing output preserved but not reusable: {path}')
                    continue
                pending.append((dataset, model, seed, str(cache), data_hash, stamp))
        failures = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = {pool.submit(fit_one, task): task for task in pending}
            for future in as_completed(jobs):
                task = jobs[future]
                try:
                    record = future.result()
                    sr.write_shard(directory / f"{record['model']}_{record['seed']}.jsonl", [record], ('dataset', 'model', 'seed'))
                    print(f"{dataset} | {record['model']} | seed={record['seed']} | params={record['n_params']} | NRMSE={record['nrmse']} | time={record['train_time_s']:.2f}s", flush=True)
                    if record['diverged']:
                        failures.append(task[1:3])
                except Exception as error:
                    failures.append(task[1:3])
                    with (directory / f'{task[1]}_{task[2]}.error.txt').open('a') as log:
                        log.write(repr(error) + '\n')
        if failures:
            raise RuntimeError(f'Failed or diverged fits retained: {failures}')

def summarize(output):
    output = Path(output)
    rows, inputs = read_records(output.glob('*/*.jsonl'))
    expected = [dict(dataset=d, model=m, seed=s) for d in pb.DATASETS for m in ROSTER for s in SEEDS]
    check = grid_inventory(rows, ('dataset', 'model', 'seed'), expected)
    if not check['complete_and_valid'] or any((r.get('diverged') or r.get('executed_steps') != STEPS for r in rows)):
        raise ValueError('Complete valid 12-dataset, 15-model, five-seed population required before reporting')
    if len({r.get('source_fingerprint') for r in rows}) != 1:
        raise ValueError('Mixed source populations')
    manifest_path = output / 'protocol.json'
    if not manifest_path.exists():
        raise ValueError('Runtime protocol manifest required')
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get('source_fingerprint'), str) or len(manifest['source_fingerprint']) != 64:
        raise ValueError('Valid runtime source fingerprint required')
    if any((r.get('source_fingerprint') != manifest.get('source_fingerprint') or r.get('metric_ddof') != 0 or r.get('validation_only_selection') is not True or (r['model'] in ('banach_c1', 'lp_c1') and r.get('duality_map') != 'c1') for r in rows)):
        raise ValueError('Incompatible source, metric, selection or response-map metadata')
    matrix = np.array([[np.median([r['nrmse'] for r in rows if r['dataset'] == d and r['model'] == m]) for m in ROSTER] for d in pb.DATASETS])
    summary = dict(protocol=protocol() | {'accuracy_evidence': True}, inputs=relative_input_records(inputs, ROOT), inventory=check, datasets=pb.DATASETS, models=ROSTER, values=matrix.tolist(), scorecard=describe(matrix))
    sr.write_shard(output / 'summary.json', [summary], ('datasets',))
    headers = ['B', 'L', 'T', 'P', 'A', 'M', 'Q', 'S', '$S_L$', 'C', 'R', 'F', 'W', 'J', 'N']
    lines = ['{\\scriptsize\\setlength{\\tabcolsep}{1pt}', '\\begin{tabular}{l' + 'r' * len(ROSTER) + '}', '\\toprule', 'Dataset & ' + ' & '.join(headers) + ' \\\\', '\\midrule']
    for dataset, row in zip(pb.DATASETS, matrix):
        lines.append(dataset.replace('_', '\\_') + ' & ' + ' & '.join(('\\textbf{' + f'{v:.3f}' + '}' if v == row.min() else f'{v:.3f}' for v in row)) + ' \\\\')
    lines.append('\\midrule')
    for label, field, spec in [('Median NRMSE', 'median', '.3f'), ('Wins', 'wins', 'g'), ('Mean rank', 'rank', '.2f')]:
        values = summary['scorecard'][field]
        best = (max if field == 'wins' else min)(values)
        lines.append(label + ' & ' + ' & '.join(('\\textbf{' + format(v, spec) + '}' if v == best else format(v, spec) for v in values)) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '}']
    (output / 'table.tex').write_text('\n'.join(lines) + '\n')
    return summary
