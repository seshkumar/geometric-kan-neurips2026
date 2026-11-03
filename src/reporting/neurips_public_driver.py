from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import math
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
    subprocess.run([sys.executable, str(ROOT / 'experiments/export_neurips_camera_ready.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/aggregate_neurips_public.py')], cwd=ROOT, check=True)
    parameter_table()
    checks = {}
    for reference in (ROOT / 'reference_tables').iterdir():
        base = ROOT / 'experiments/results'
        source = base / 'neurips_public_matched' / reference.name if reference.name.startswith(('tab_', 'matched_')) else base / 'neurips_camera_ready' / reference.name
        checks[reference.name] = source.read_bytes() == reference.read_bytes()
    result = {'all_tables_and_figure_match': all(checks.values()), 'artifacts': checks}
    (ROOT / 'experiments/results/reproduction_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['all_tables_and_figure_match']:
        raise ValueError('Regenerated paper artifact mismatch')
    return result

def parameter_table():
    from experiments.rebuttal import run_baselines as rb
    from src.reporting.neurips_public_parameters import parameter_counts
    from src.reporting.neurips_public_formatting import LABEL, ROSTER
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
        from src.neurips_rebuttal_models import LpC1KANLayer, BanachC1KANLayer
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
        from experiments.rebuttal import run_baselines as rb
        if args.command == 'matched':
            record = rb.run_one_config(rb.select_equations('50')[args.eq_idx], args.model, args.seed, args.n_train, args.sigma, 5000, 0.01)
            record['condition'] = args.condition
            record['duality_map'] = 'c1' if args.model in ('banach_c1', 'lp_c1') else 'comparator'
        else:
            from experiments.rebuttal.extrapolation import gen_extrap
            eq = rb.select_equations('50')[args.eq_idx]
            x, y, xt, yt = gen_extrap(eq, args.seed)
            torch.manual_seed(args.seed)
            model = rb.MODEL_BUILDERS[args.model](eq[2])
            start = time.perf_counter()
            rb.train_one(model, x, y, xt, yt, steps=5000, lr=0.01, **rb.train_kwargs(args.model))
            record = {'eq': eq[0], 'model': args.model, 'seed': args.seed, 'nrmse': rb.evaluate(model, xt, yt), 'n_params': sum((p.numel() for p in model.parameters())), 'train_time_s': time.perf_counter() - start}
    if not math.isfinite(record['nrmse']):
        raise ValueError('Nonfinite fit result')
    with output.open('x') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='Verify bundled file hashes; no fitting')
    sub.add_parser('tables', help='Regenerate all numerical tables and matched figure; no fitting')
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
    result = verify() if args.command == 'verify' else tables() if args.command == 'tables' else run_fit(args)
    print(json.dumps(result, indent=2))
