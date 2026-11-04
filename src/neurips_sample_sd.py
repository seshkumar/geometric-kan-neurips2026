from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments import feynman_runner as core
from src import fit_records as sr
from src import neurips_followup as followup
from src.neurips_normalization import symbolic_data
from src.neurips_rerun import declare
from src.edge_models import LpC1KANLayer, BanachC1KANLayer
from src.reporting.neurips_public_formatting import ROSTER
from src.reporting.neurips_camera_ready import describe, paired_ratios
from src.result_integrity import grid_inventory, read_records, relative_input_records
from src.neurips_model_sources import expected_source
ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(1729, 1734))
FIELDS = ('eq', 'model', 'seed', 'n_train', 'sigma')

def cases(profile, index):
    if profile == 'extrapolation':
        return [(m, 500, 0.0) for m in ROSTER]
    if profile != 'symbolic':
        raise ValueError('Unknown sample-SD profile')
    main = [(500, s) for s in (0.0, 0.1, 0.3, 0.5, 1.0)] + [(n, 0.0) for n in (50, 100, 200, 1000)]
    result = {(m, n, s) for m in ROSTER for n, s in main}
    result.add(('core_spline_g3', 500, 0.0))
    if index < 18:
        noisy_sizes = [(n, 0.3) for n in (50, 100, 200, 1000)]
        controls = [(500, s) for s in (0.0, 0.1, 0.3, 0.5, 1.0)] + noisy_sizes
        result.update(((m, n, s) for m in ('banach_c1', 'lp_c1', 'tanh', 'mlp', 'cheby') for n, s in noisy_sizes))
        result.update(((m, n, s) for m in ('core_spline_g3', 'core_spline_cv_g3') for n, s in controls))
        result.update(((m, n, s) for m in followup.TIED_MODELS[1:] for n, s in controls))
        fixed = [f'core_lp_fixed_{p:.1f}' for p in (1.5, 2.0, 2.5, 3.0, 4.0)] + ['core_banach_fixed_2.0']
        result.update(((m, 500, s) for m in fixed for s in (0.0, 0.3)))
        result.update(((m, 500, 0.0) for m in ('core_banach_tanhonly', 'core_spline_adam_g3', 'core_spline_adam_steps10000_g3')))
    return sorted(result)

def plan(profile):
    seeds = list(range(5)) if profile == 'extrapolation' else SEEDS
    return dict(profile=profile, target_sd_ddof=1, metric_ddof=1, canonical_generator=('Existing extrapolation sampler' if profile == 'extrapolation' else 'Original core generator') + '; training-only sample-SD normalization', seeds=seeds, equations=core.EQUATION_IDS, validation_seed=0, validation_fraction=0.2, reference_policy='Fit each unique configuration once; every table reads the same canonical reference', tasks_by_case=[len(cases(profile, i)) * len(seeds) for i in range(50)], aliases={'banach': 'banach_c1', 'banach_learned': 'banach_c1', 'independent_c1': 'banach_c1', 'lp': 'lp_c1', 'lp_learned': 'lp_c1'}, performance_evidence=False)

def identity():
    stamp = followup.source_identity()
    for name in ('src/neurips_sample_sd.py', 'src/neurips_rerun.py', 'experiments/neurips_sample_sd.py'):
        stamp['source_files'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    stamp['source_fingerprint'] = sr.source_stamp([ROOT / p for p in stamp['source_files']])
    return stamp

def fit(task):
    profile, index, name, n, sigma, seed, path, digest, stamp = task
    torch.set_num_threads(1)
    if not name.startswith('core_'):
        study = 'extrap15' if name in ROSTER else 'tied_affine'
        record = followup.fit_one((study, index, name, n, sigma, seed, path, digest, stamp, 1))
    else:
        dependencies = [ROOT / p for p in stamp['source_files']]
        if sr.source_stamp(dependencies) != stamp['source_fingerprint'] or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source or retained input changed before fitting')
        previous = (core.LpKANLayer, core.BanachKANLayer)
        core.LpKANLayer, core.BanachKANLayer = (LpC1KANLayer, BanachC1KANLayer)
        try:
            args = SimpleNamespace(experiment='sample_sd', eq_idx=index, model=name[5:], n_train=n, sigma=sigma, seed=seed, c1=True)
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                core.run_single(args)
            rows = [json.loads(l) for l in capture.getvalue().splitlines() if l.startswith('{')]
            if len(rows) != 1:
                raise ValueError('Expected exactly one core-control record')
            original = rows[0]
        finally:
            core.LpKANLayer, core.BanachKANLayer = previous
        if sr.source_stamp(dependencies) != stamp['source_fingerprint']:
            raise ValueError('Source changed during fitting')
        record = dict(eq=core.EQUATION_IDS[index], model=name, seed=seed, n_train=n, sigma=sigma, nrmse=original['nrmse'], train_time_s=original['train_time_s'], n_params=original['n_params_trainable'], original_core_record=original, input_file=Path(path).name, input_sha256=digest, **stamp)
    record.update(study='neurips_sample_sd_' + profile, profile=profile, target_sd_ddof=1, metric_ddof=1, reference_id=f'{profile}/{index}/{name}/{n}/{sigma:g}/{seed}')
    return record

def validate(rows, profile, indices, out, stamp, complete):
    seeds = plan(profile)['seeds']
    expected = [dict(eq=core.EQUATION_IDS[i], model=m, seed=seed, n_train=n, sigma=s) for i in indices for m, n, s in cases(profile, i) for seed in seeds]
    check = grid_inventory(rows, FIELDS, expected, metrics=('nrmse', 'train_time_s'))
    if any((check[k] for k in ('extra_count', 'duplicate_keys', 'invalid_count'))) or (complete and check['missing_count']):
        raise ValueError('Incomplete or invalid sample-SD population: ' + str(check))
    for r in rows:
        if r.get('source_fingerprint') != expected_source(r, stamp) or r.get('target_sd_ddof') != 1 or r.get('metric_ddof') != 1 or (r.get('profile') != profile):
            raise ValueError('Mixed protocol/source/SD conventions')
        i = core.EQUATION_IDS.index(r['eq'])
        p = out / f'eq_{i:02d}' / r['input_file']
        if hashlib.sha256(p.read_bytes()).hexdigest() != r['input_sha256']:
            raise ValueError('Retained input hash mismatch')
    return check

def run(profile, index, out, workers, replace_existing=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = identity()
    declaration = dict(protocol=plan(profile), **stamp)
    with followup.lock(out / '.manifest.lock', nonblocking=False):
        declare(out, declaration, replace_existing)
    directory = out / f'eq_{index:02d}'
    directory.mkdir(exist_ok=True)
    with followup.lock(directory / '.writer.lock'):
        path = directory / 'fits.jsonl'
        rows = read_records([path])[0] if path.exists() else []
        if rows and any((r.get('source_fingerprint') != stamp['source_fingerprint'] or r.get('profile') != profile or r.get('target_sd_ddof') != 1 for r in rows)):
            if not replace_existing:
                raise ValueError('Existing fits require --replace-existing')
            rows = []
        validate(rows, profile, [index], out, stamp, False)
        done = {tuple((r[k] for k in FIELDS)) for r in rows}
        tasks = []
        prepared = {}
        for name, n, sigma in cases(profile, index):
            for seed in plan(profile)['seeds']:
                if (core.EQUATION_IDS[index], name, seed, n, sigma) in done:
                    continue
                input_path = directory / f'input_n{n}_sigma{sigma:g}_seed{seed}.npz'
                study = 'extrap15' if profile == 'extrapolation' else 'tied_affine'
                key = (n, sigma, seed)
                if key not in prepared:
                    if replace_existing and (not rows):
                        input_path.unlink(missing_ok=True)
                    prepared[key] = followup.prepare_input(directory, study, index, n, sigma, seed, 1)
                input_path, digest = prepared[key]
                tasks.append((profile, index, name, n, sigma, seed, str(input_path), digest, stamp))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(fit, t) for t in tasks]
            for future in as_completed(futures):
                r = future.result()
                rows.append(r)
                sr.write_shard(path, rows, FIELDS)
                print(f"{r['eq']} | {r['model']} | n={r['n_train']} sigma={r['sigma']} seed={r['seed']} | params={r['n_params']} NRMSE={r['nrmse']}", flush=True)
        validate(rows, profile, [index], out, stamp, True)

def summarize(profile, out):
    out = Path(out).resolve()
    stamp = json.loads((out / 'protocol.json').read_text())
    if stamp['protocol'] != plan(profile):
        raise ValueError('Protocol differs from the declared sample-SD population')
    rows, inputs = read_records(sorted(out.glob('eq_*/fits.jsonl')))
    check = validate(rows, profile, range(50), out, stamp, True)
    result = dict(protocol=plan(profile) | {'performance_evidence': True}, inventory=check, inputs=relative_input_records(inputs, ROOT), reference_policy=plan(profile)['reference_policy'])
    result['model_replacements'] = stamp.get('model_replacements', {})
    if profile == 'extrapolation':
        matrix = np.asarray([[np.median([r['nrmse'] for r in rows if r['eq'] == e and r['model'] == m]) for m in ROSTER] for e in core.EQUATION_IDS])
        comparisons = [(f'{a}/{b}', matrix[:, ROSTER.index(a)], matrix[:, ROSTER.index(b)]) for a in ('banach_c1', 'lp_c1') for b in ROSTER if b not in ('banach_c1', 'lp_c1')]
        result.update(models=ROSTER, equations=core.EQUATION_IDS, cells={'n=500/sigma=0': dict(values=matrix.tolist(), scorecard=describe(matrix), paired=paired_ratios(comparisons))})
    sr.write_shard(out / 'summary.json', [result], ('reference_policy',))
    return result
