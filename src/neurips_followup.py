from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from src import matched_models as rb, fit_records as sr
from src.extrapolation import gen_extrap
from src.reporting.neurips_camera_ready import describe, paired_ratios
from src.reporting.neurips_public_formatting import ROSTER
from src.result_integrity import grid_inventory, read_records, relative_input_records
from src.training import carve_validation_split
ROOT = Path(__file__).resolve().parents[1]
STEPS = 5000
TIED_MODELS = ['independent_c1', 'tied_c1_matched', 'tied_c1', 'tied_c1_p2_init']
FIELDS = ('eq', 'model', 'seed', 'n_train', 'sigma')

def specification(study):
    if study not in ('extrap15', 'tied_affine'):
        raise ValueError('Unknown study')
    extrap = study == 'extrap15'
    equations = rb.select_equations('50' if extrap else '18')
    conditions = [(500, 0.0)] if extrap else [(500, s) for s in (0.0, 0.1, 0.3, 0.5, 1.0)] + [(n, 0.3) for n in (50, 100, 200, 1000)]
    models = ROSTER if extrap else TIED_MODELS
    seeds = list(range(5)) if extrap else list(range(1729, 1734))
    return dict(study=study, equations=[e[0] for e in equations], models=models, seeds=seeds, conditions=[dict(n_train=n, sigma=s) for n, s in conditions], expected_fits=len(equations) * len(models) * len(seeds) * len(conditions), steps=STEPS, learning_rate=0.01, n_test=1000, validation_fraction=0.2, validation_seed=0, checkpoint_every=500, schedule='cosine', gradient_clip=10.0, selection='Training-only validation; test inputs and labels are never passed to the trainer', metric='Existing evaluate: RMSE / (sample std of standardized test targets + 1e-12), ddof=1', data_generator='extrapolation.gen_extrap(inner=.6,reach=.3)' if extrap else 'run_baselines.generate_data', seed_policy='Preserve the earlier extrapolation seeds' if extrap else 'Preserve the core-control seeds', architecture='Existing Table-2 builders and dimension-dependent widths; C1 proposed responses, epsilon=1e-8', tied_controls=None if extrap else dict(independent_c1=dict(affines='independent', p_init=2.5, power_gain_init=0.5), tied_c1_matched=dict(affines='shared', p_init=2.5, power_gain_init=0.5), tied_c1=dict(affines='shared', p_init=2.5, power_gain_init=0.1), tied_c1_p2_init=dict(affines='shared', p_init=2.0, power_gain_init=0.1)), tied_interpretation='Signed gains are unconstrained; tying affines alone is not a convex-duality guarantee', comparison_family='Two proposed power models versus the other 13 models: 26 actual comparisons, no reserved slots' if extrap else 'Matched tied/independent comparisons per declared condition; other initializations are separate controls', historical_results_reused=False, performance_evidence=False)

@contextmanager
def lock(path, nonblocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

def source_identity():
    paths = [Path(__file__), ROOT / 'experiments/neurips_followup.py', ROOT / 'src/matched_models.py', ROOT / 'src/extrapolation.py', ROOT / 'src/fit_records.py', ROOT / 'src/kan_layer.py', ROOT / 'src/edge_models.py', ROOT / 'src/training.py', ROOT / 'src/result_integrity.py', ROOT / 'src/reporting/neurips_public_formatting.py', ROOT / 'src/reporting/neurips_camera_ready.py']
    return dict(code_revision='public-release', source_fingerprint=sr.source_stamp(paths), source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def build(study, name, dimension):
    if study == 'extrap15':
        return rb.MODEL_BUILDERS[name](dimension)
    if name == 'independent_c1':
        return rb.MODEL_BUILDERS['banach_c1'](dimension)
    if name not in TIED_MODELS:
        raise ValueError('Unknown control')
    return rb.KANetwork(rb.kan_widths(dimension), rb.BanachTiedKANLayer, c1=True, p_init=2.0 if name == 'tied_c1_p2_init' else 2.5, a2_init=0.5 if name == 'tied_c1_matched' else 0.1)

def tensors(study, index, n, sigma, seed):
    eq = rb.select_equations('50')[index]
    values = gen_extrap(eq, seed, n_train=n, n_test=1000) if study == 'extrap15' else rb.generate_data(eq, n, sigma, seed)[:4]
    if len(values[0]) != n or len(values[2]) != 1000 or any((not torch.isfinite(v).all() for v in values)):
        raise ValueError('Generator returned insufficient or nonfinite inputs; protocol not silently changed')
    return values

def prepare_input(directory, study, index, n, sigma, seed):
    path = directory / f'input_n{n}_sigma{sigma:g}_seed{seed}.npz'
    keys = ('train_x', 'train_y', 'test_x', 'test_y')
    expected = dict(zip(keys, [v.numpy() for v in tensors(study, index, n, sigma, seed)]))
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(keys) or any((not np.array_equal(data[k], expected[k]) for k in keys)):
                raise ValueError('Existing input preserved but differs from declared generator: ' + str(path))
    else:
        temporary = path.with_suffix(f'.tmp{os.getpid()}')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **expected)
        os.replace(temporary, path)
    return (path, hashlib.sha256(path.read_bytes()).hexdigest())

def fit_one(task):
    study, index, name, n, sigma, seed, path, digest, stamp = task
    torch.set_num_threads(1)
    dependencies = [ROOT / p for p in stamp['source_files']]
    if sr.source_stamp(dependencies) != stamp['source_fingerprint']:
        raise ValueError('Source changed before fitting')
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError('Input changed during run')
    with np.load(path, allow_pickle=False) as data:
        x, y, xt, yt = [torch.from_numpy(data[k].copy()) for k in ('train_x', 'train_y', 'test_x', 'test_y')]
    torch.manual_seed(seed)
    model = build(study, name, x.shape[1])
    started = time.perf_counter()
    executed = sr.train_counted(model, x, y, steps=STEPS, lr=0.01, **rb.train_kwargs(name))
    training_time = time.perf_counter() - started
    _, _, xv, yv = carve_validation_split(x, y)
    model.eval()
    with torch.no_grad():
        validation_rmse = float(torch.sqrt(torch.mean((model(xv).reshape(-1) - yv.reshape(-1)) ** 2)))
    started = time.perf_counter()
    value = rb.evaluate(model, xt, yt)
    inference_time = time.perf_counter() - started
    bad = executed != STEPS or not (math.isfinite(value) and math.isfinite(validation_rmse))
    if sr.source_stamp(dependencies) != stamp['source_fingerprint']:
        raise ValueError('Source changed during fitting')
    return dict(study=study, eq=rb.select_equations('50')[index][0], model=name, seed=seed, n_train=n, sigma=sigma, n_test=len(xt), n_features=x.shape[1], n_params=sum((p.numel() for p in model.parameters())), nrmse=value if math.isfinite(value) else None, validation_rmse=validation_rmse if math.isfinite(validation_rmse) else None, train_time_s=training_time, inference_time_s=inference_time, steps=STEPS, executed_steps=executed, diverged=bad, input_file=Path(path).name, input_sha256=digest, validation_only_selection=True, duality_map='c1' if study == 'tied_affine' or name in ('banach_c1', 'lp_c1') else 'comparator', code_revision=stamp['code_revision'], source_fingerprint=stamp['source_fingerprint'])

def expected(config, equations=None):
    return [dict(eq=e, model=m, seed=s, **c) for e in equations or config['equations'] for m in config['models'] for s in config['seeds'] for c in config['conditions']]

def validate(rows, config, stamp, input_dir, eqs=None, complete=False):
    check = grid_inventory(rows, FIELDS, expected(config, eqs), metrics=('nrmse', 'validation_rmse', 'train_time_s', 'inference_time_s'))
    if any((check[k] for k in ('extra_count', 'duplicate_keys', 'invalid_count'))) or (complete and check['missing_count']):
        raise ValueError('Incomplete or invalid result grid: ' + str(check))
    for r in rows:
        if r.get('study') != config['study'] or r.get('source_fingerprint') != stamp['source_fingerprint'] or r.get('steps') != STEPS or (r.get('executed_steps') != STEPS) or r.get('diverged') or (r.get('validation_only_selection') is not True):
            raise ValueError('Incompatible fit preserved; use a fresh run ID')
        if (config['study'] == 'tied_affine' or r['model'] in ('banach_c1', 'lp_c1')) and r.get('duality_map') != 'c1':
            raise ValueError('Missing C1 response identity')
        eqindex = config['equations'].index(r['eq'])
        path = input_dir / f'eq_{eqindex:02d}' / r['input_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != r['input_sha256']:
            raise ValueError('Changed retained input')
    return check

def run(study, index, out, workers):
    config = specification(study)
    if not 0 <= index < len(config['equations']) or workers < 1:
        raise ValueError('Invalid case or worker count')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = source_identity()
    manifest = dict(protocol=config, **stamp, environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__))
    with lock(out / '.manifest.lock', nonblocking=False):
        path = out / 'protocol.json'
        if path.exists():
            prior = json.loads(path.read_text())
            if any((prior[k] != manifest[k] for k in ('protocol', 'source_fingerprint', 'environment'))):
                raise ValueError('Existing run has different code, protocol or environment; use a fresh run ID')
        else:
            if any((p.name != '.manifest.lock' for p in out.iterdir())):
                raise ValueError('Nonempty output without manifest preserved')
            sr.write_shard(path, [manifest], ('source_fingerprint',))
    directory = out / f'eq_{index:02d}'
    directory.mkdir(exist_ok=True)
    with lock(directory / '.writer.lock'):
        path = directory / 'fits.jsonl'
        rows = read_records([path])[0] if path.exists() else []
        validate(rows, config, stamp, out, [config['equations'][index]])
        completed = {tuple((r[f] for f in FIELDS)) for r in rows}
        tasks = []
        for condition in config['conditions']:
            n, sigma = (condition['n_train'], condition['sigma'])
            for seed in config['seeds']:
                data, digest = prepare_input(directory, study, index, n, sigma, seed)
                for name in config['models']:
                    key = (config['equations'][index], name, seed, n, sigma)
                    if key not in completed:
                        tasks.append((study, index, name, n, sigma, seed, str(data), digest, stamp))
        failures = []
        if tasks:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {pool.submit(fit_one, t): t for t in tasks}
                for future in as_completed(futures):
                    try:
                        r = future.result()
                        rows.append(r)
                        sr.write_shard(path, rows, FIELDS)
                        print(f"{r['eq']} | {r['model']} | seed={r['seed']} n={r['n_train']} sigma={r['sigma']} | params={r['n_params']} NRMSE={r['nrmse']} | {r['train_time_s']:.2f}s", flush=True)
                        if r['diverged']:
                            failures.append(str(futures[future][2:6]))
                    except Exception as error:
                        failures.append(repr(error))
                        with (directory / 'errors.txt').open('a') as stream:
                            stream.write(repr(error) + '\n')
        if failures:
            raise RuntimeError('Failed fits retained: ' + str(failures))
        validate(rows, config, stamp, out, [config['equations'][index]], complete=True)
        print(f"COMPLETE {study} {config['equations'][index]}: {len(rows)} fits", flush=True)

def summarize(study, out):
    out = Path(out).resolve()
    config = specification(study)
    manifest = json.loads((out / 'protocol.json').read_text())
    if manifest['protocol'] != config:
        raise ValueError('Protocol mismatch')
    rows, inputs = read_records(sorted(out.glob('eq_*/fits.jsonl')))
    inventory = validate(rows, config, manifest, out, complete=True)
    cells = {}
    for c in config['conditions']:
        matrix = np.asarray([[np.median([r['nrmse'] for r in rows if r['eq'] == e and r['model'] == m and (r['n_train'] == c['n_train']) and (r['sigma'] == c['sigma'])]) for m in config['models']] for e in config['equations']])
        key = f"n={c['n_train']}/sigma={c['sigma']:g}"
        pairs = []
        if study == 'extrap15':
            pairs = [(f'{a}/{b}', matrix[:, config['models'].index(a)], matrix[:, config['models'].index(b)]) for a in ('banach_c1', 'lp_c1') for b in config['models'] if b not in ('banach_c1', 'lp_c1')]
        else:
            pairs = [(f'{m}/independent_c1', matrix[:, j], matrix[:, 0]) for j, m in enumerate(config['models']) if j]
        cells[key] = dict(values=matrix.tolist(), scorecard=describe(matrix), paired=paired_ratios(pairs))
    result = dict(protocol=config | {'performance_evidence': True}, inputs=relative_input_records(inputs, ROOT), inventory=inventory, models=config['models'], equations=config['equations'], cells=cells)
    sr.write_shard(out / 'summary.json', [result], ('models',))
    return result
