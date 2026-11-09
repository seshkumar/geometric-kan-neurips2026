from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]

def verify(root=ROOT):
    manifest = json.loads((root / 'release_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Release file changed: ' + name)
    return {'verified_files': len(manifest['sha256'])}

def tables():
    verify()
    canonical = (ROOT / 'experiments/results/neurips_sample_sd/symbolic/summary.json').exists()
    if canonical:
        subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_sample_sd.py')], cwd=ROOT, check=True)
    else:
        subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_camera_ready.py')], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'experiments/aggregate_neurips_public.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/neurips_figure1_matched.py'), 'export'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_figure1_manuscript.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/neurips_pmlb15.py'), 'summarize', '--out', str(ROOT / 'experiments/results/neurips_pmlb15/pmlb15_pbs_v1')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_pmlb_tables.py')], cwd=ROOT, check=True)
    for study, run in [] if canonical else [('extrap15', 'extrap15_v1'), ('tied_affine', 'tied_affine_v1')]:
        result_dir = ROOT / 'experiments/results/neurips_followup' / study / run
        declared = json.loads((result_dir / 'protocol.json').read_text())['protocol']
        if declared.get('profile') == 'extrapolation':
            command = [sys.executable, str(ROOT / 'experiments/neurips_sample_sd.py'), 'summarize', '--profile', 'extrapolation', '--out', str(result_dir)]
        else:
            command = [sys.executable, str(ROOT / 'experiments/neurips_followup.py'), 'summarize', '--study', study, '--out', str(result_dir)]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    if not canonical:
        subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_followup_tables.py')], cwd=ROOT, check=True)
    if (ROOT / 'experiments/results/neurips_figure1_budget').exists():
        subprocess.run([sys.executable, str(ROOT / 'experiments/neurips_figure1_budget.py'), 'export'], cwd=ROOT, check=True)
    parameter_table()
    checks = {}
    reference_artifacts = json.loads((ROOT / 'release_manifest.json').read_text())['reference_artifact_sha256']
    for name, digest in reference_artifacts.items():
        base = ROOT / 'experiments/results'
        source = base / 'neurips_figure1_budget/budget_sweep.tex' if name == 'figure1_budget.tex' else base / 'neurips_followup_tables' / name.removeprefix('followup_') if name.startswith('followup_') else base / 'neurips_pmlb15_manuscript' / name.removeprefix('pmlb_') if name.startswith('pmlb_') else base / 'neurips_figure1_matched' / name if name.startswith(('figure1_', 'shape_')) else base / 'neurips_public_matched' / name if name.startswith(('tab_', 'matched_')) else base / 'neurips_camera_ready' / name
        checks[name] = hashlib.sha256(source.read_bytes()).hexdigest() == digest
    result = {'all_tables_and_figure_match': all(checks.values()), 'artifacts': checks}
    (ROOT / 'experiments/results/reproduction_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_tables_and_figure_match']:
        raise ValueError('Regenerated paper artifact mismatch')
    return result

def parameter_table():
    from src import matched_models as rb
    from src.reporting.neurips_public_parameters import parameter_counts
    from src.reporting.neurips_public_formatting import LABEL, ROSTER
    canonical = ROOT / 'experiments/results/neurips_sample_sd/symbolic'
    if canonical.exists():
        rows = [dict(r, n_vars=r['n_features']) for path in canonical.glob('eq_*/fits.jsonl') for line in path.read_text().splitlines() for r in [json.loads(line)] if r['model'] in ROSTER and r['n_train'] == 500 and (r['sigma'] == 0.0)]
    else:
        rows = [json.loads(line) for line in (ROOT / 'experiments/results/neurips_public_inputs/baseline.jsonl').read_text().splitlines()]
    counts, dimensions = parameter_counts(rows)
    for model in ROSTER:
        for d in dimensions:
            if sum((p.numel() for p in rb.MODEL_BUILDERS[model](d).parameters() if p.requires_grad)) != counts[model, d]:
                raise ValueError('Parameter budget mismatch')
    tex = ['\\begin{tabular}{l' + 'r' * len(dimensions) + '}', '\\toprule', 'Model & ' + ' & '.join((f'$d={d}$' for d in dimensions)) + ' \\\\', '\\midrule']
    for model in ROSTER:
        tex.append(LABEL[model] + ' & ' + ' & '.join((str(counts[model, d]) for d in dimensions)) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    (ROOT / 'experiments/results/neurips_camera_ready/parameter_counts.tex').write_text('\n'.join(tex) + '\n')

def run_fit(args):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    import torch
    torch.set_num_threads(1)
    if args.command == 'core':
        from experiments import feynman_runner as fr
        from src.edge_models import LpC1KANLayer, BanachC1KANLayer
        previous = (fr.LpKANLayer, fr.BanachKANLayer)
        fr.LpKANLayer, fr.BanachKANLayer = (LpC1KANLayer, BanachC1KANLayer)
        try:
            namespace = SimpleNamespace(experiment=args.experiment, eq_idx=args.eq_idx, seed=args.seed, model=args.model, n_train=args.n_train, sigma=args.sigma, c1=True)
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                fr.run_single(namespace)
            rows = [json.loads(line) for line in capture.getvalue().splitlines() if line.lstrip().startswith('{')]
            if len(rows) != 1:
                raise ValueError('Expected one core fit record')
            record = rows[0]
        finally:
            fr.LpKANLayer, fr.BanachKANLayer = previous
    else:
        from src import matched_models as rb
        if args.command == 'matched':
            record = rb.run_one_config(rb.select_equations('50')[args.eq_idx], args.model, args.seed, args.n_train, args.sigma, 5000, 0.01)
            record['condition'] = args.condition
            record['duality_map'] = 'c1' if args.model in ('banach_c1', 'lp_c1') else 'comparator'
        else:
            from src.extrapolation import gen_extrap
            eq = rb.select_equations('50')[args.eq_idx]
            x, y, xt, yt = gen_extrap(eq, args.seed)
            torch.manual_seed(args.seed)
            model = rb.MODEL_BUILDERS[args.model](eq[2])
            start = time.perf_counter()
            rb.train_one(model, x, y, xt, yt, steps=5000, lr=0.01, **rb.train_kwargs(args.model))
            record = {'eq': eq[0], 'model': args.model, 'seed': args.seed, 'nrmse': rb.evaluate(model, xt, yt), 'n_params': sum((p.numel() for p in model.parameters())), 'train_time_s': time.perf_counter() - start}
    record.update(target_sd_ddof=1, metric_ddof=1)
    if not math.isfinite(record['nrmse']):
        raise ValueError('Nonfinite fit result')
    with output.open('x') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    return record

def run_local(args):
    if args.workers < 1:
        raise ValueError('Workers must be positive')
    if args.study == 'pmlb':
        from src.neurips_pmlb import protocol
        count = len(protocol()['datasets'])
        driver = 'experiments/neurips_pmlb15.py'
        selection = ['--dataset-idx']
        profile = []
    else:
        from src.neurips_sample_sd import plan
        count = len(plan(args.study)['equations'])
        driver = 'experiments/neurips_sample_sd.py'
        selection = ['--case-index']
        profile = ['--profile', args.study]
    out = (args.out or ROOT / 'experiments/results/local_runs' / args.study).resolve()
    protected = [ROOT / 'experiments/results/neurips_sample_sd/symbolic', ROOT / 'experiments/results/neurips_followup/extrap15/extrap15_v1', ROOT / 'experiments/results/neurips_pmlb15/pmlb15_pbs_v1']
    if any((out == p or out.is_relative_to(p) for p in protected)):
        raise ValueError('Choose a fresh output tree; bundled evidence is read-only')
    if args.all_cases:
        indices = list(range(count))
    elif args.case_index is not None:
        indices = [args.case_index]
    else:
        indices = [] if args.aggregate else [0]
    if any((i < 0 or i >= count for i in indices)):
        raise ValueError('Case index outside the declared study')
    commands = [[sys.executable, str(ROOT / driver), 'run', *profile, *selection, str(i), '--out', str(out), '--workers', str(args.workers)] for i in indices]
    if args.aggregate:
        commands.append([sys.executable, str(ROOT / driver), 'summarize', *profile, '--out', str(out)])
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return dict(study=args.study, cases=indices, workers=args.workers, output=str(out), dry_run=args.dry_run, aggregate=args.aggregate)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Verify bundled file hashes; no fitting')
    sub.add_parser('tables', help='Regenerate all numerical tables and matched figure; no fitting')
    p = sub.add_parser('run', help='Train one declared study on a single machine')
    p.add_argument('--study', choices=('symbolic', 'extrapolation', 'pmlb'), default='symbolic')
    cases = p.add_mutually_exclusive_group()
    cases.add_argument('--case-index', type=int, help='Equation or dataset index; default: first case')
    cases.add_argument('--all-cases', action='store_true')
    p.add_argument('--workers', type=int, default=1, help='Concurrent fits on this machine')
    p.add_argument('--out', type=Path, help='Fresh output tree; default: experiments/results/local_runs/<study>')
    p.add_argument('--aggregate', action='store_true', help='Summarize a complete population; alone, skips training')
    p.add_argument('--dry-run', action='store_true', help='Print commands without fitting or downloading data')
    for command in ('core', 'matched', 'extrapolation'):
        p = sub.add_parser(command, help='Run one full-budget condition into a fresh output file')
        p.add_argument('--eq-idx', type=int, default=0, choices=range(50))
        p.add_argument('--seed', type=int, default=0 if command == 'extrapolation' else 1729)
        p.add_argument('--model', default='banach' if command == 'core' else 'banach_c1')
        p.add_argument('--output', required=True, type=Path)
        if command != 'extrapolation':
            p.add_argument('--n-train', type=int, default=500)
            p.add_argument('--sigma', type=float, default=0.0)
        if command == 'core':
            p.add_argument('--experiment', default='baseline50')
        if command == 'matched':
            p.add_argument('--condition', choices=('baseline', 'noisy', 'small_sample'), default='baseline')
    args = parser.parse_args()
    result = verify() if args.command == 'verify' else tables() if args.command == 'tables' else run_local(args) if args.command == 'run' else run_fit(args)
    print(json.dumps(result, indent=2))
