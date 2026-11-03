import subprocess
import sys
import os
import argparse
import json
import math
from multiprocessing import Pool
SEEDS = [1729, 1730, 1731, 1732, 1733]
STANDARD_MODELS = ['spline_g3', 'spline_g5', 'spline_g10', 'spline_g20', 'spline_adam_g3', 'spline_adam_lr0.003_g3', 'spline_adam_lr0.001_g3', 'spline_adam_lr0.0003_g3', 'spline_adam_steps10000_g3', 'tanh', 'banach', 'banach_tanhonly', 'lp', 'mlp', 'cheby']
SPLINE_REG_MODELS = ['spline_reg_g3_l0', 'spline_reg_g3_l0.001', 'spline_reg_g3_l0.01', 'spline_reg_g3_l0.1', 'spline_cv_g3']
ABLATION_MODELS = ['lp_fixed_1.5', 'lp_fixed_2.0', 'lp_fixed_2.5', 'lp_fixed_3.0', 'lp_fixed_4.0', 'lp_learned', 'banach_fixed_2.0', 'banach_learned']
TIED_AFFINE_MODELS = ['banach', 'banach_tied']
C1_MODELS = {'banach', 'lp', 'lp_fixed_1.5', 'lp_fixed_2.0', 'lp_fixed_2.5', 'lp_fixed_3.0', 'lp_fixed_4.0', 'lp_learned', 'banach_fixed_2.0', 'banach_learned'}
NOISE_SIGMAS = [0.0, 0.1, 0.3, 0.5, 1.0]
SCARCITY_NTRAINS = [50, 100, 200, 500, 1000]
BASELINE50_MODELS = ['spline_g3', 'spline_adam_g3', 'spline_adam_steps10000_g3', 'tanh', 'banach', 'banach_tanhonly', 'lp', 'mlp', 'cheby']

def run_one(config):
    cmd = [sys.executable, '-u', os.path.join(os.path.dirname(__file__), 'feynman_runner.py'), '--experiment', config['experiment'], '--eq_idx', str(config['eq_idx']), '--seed', str(config['seed']), '--model', config['model'], '--n_train', str(config['n_train']), '--sigma', str(config['sigma'])] + (['--c1'] if config.get('c1') else [])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None if config.get('c1') else 1800)
        if result.returncode == 0 and result.stdout.strip():
            try:
                records = [json.loads(line) for line in result.stdout.splitlines() if line.lstrip().startswith('{')]
                if len(records) != 1:
                    raise ValueError('Expected exactly one result record')
                row = records[0]
                if not isinstance(row.get('nrmse'), (int, float)) or not math.isfinite(row['nrmse']) or row['nrmse'] < 0:
                    raise ValueError('Missing or invalid NRMSE')
                for key in ('experiment', 'model', 'seed', 'n_train', 'sigma'):
                    if row.get(key) != config[key]:
                        raise ValueError(f'Result identity mismatch: {key}')
                if config.get('c1') and row.get('duality_map') != 'c1':
                    raise ValueError('C1 activation tag missing or incorrect')
            except (ValueError, TypeError) as error:
                return json.dumps(dict(config, error=str(error)))
            return result.stdout.strip()
        else:
            err = result.stderr.strip() if result.stderr else 'no output'
            return json.dumps({'error': err[-2000:], 'experiment': config['experiment'], 'eq_idx': config['eq_idx'], 'model': config['model'], 'seed': config['seed'], 'n_train': config['n_train'], 'sigma': config['sigma']})
    except subprocess.TimeoutExpired:
        return json.dumps({'error': 'timeout_1800s', 'experiment': config['experiment'], 'eq_idx': config['eq_idx'], 'model': config['model'], 'seed': config['seed']})

def build_configs(args):
    configs = []
    if args.experiment == 'baseline':
        for model in STANDARD_MODELS:
            for seed in SEEDS:
                configs.append({'experiment': 'baseline', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': 0.0})
    elif args.experiment == 'noise':
        for sigma in NOISE_SIGMAS:
            for model in STANDARD_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'noise', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': sigma})
    elif args.experiment == 'scarcity':
        for n_train in SCARCITY_NTRAINS:
            for model in STANDARD_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'scarcity', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': n_train, 'sigma': 0.3})
    elif args.experiment == 'noise_spline_reg':
        for sigma in NOISE_SIGMAS:
            for model in SPLINE_REG_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'noise_spline_reg', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': sigma})
    elif args.experiment == 'scarcity_spline_reg':
        for n_train in SCARCITY_NTRAINS:
            for model in SPLINE_REG_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'scarcity_spline_reg', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': n_train, 'sigma': 0.3})
    elif args.experiment == 'ablation_p':
        for sigma in [0.0, 0.3]:
            for model in ABLATION_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'ablation_p', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': sigma})
    elif args.experiment == 'baseline50':
        for model in BASELINE50_MODELS:
            for seed in SEEDS:
                configs.append({'experiment': 'baseline50', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': 0.0})
    elif args.experiment == 'baseline18':
        for model in STANDARD_MODELS + SPLINE_REG_MODELS:
            for seed in SEEDS:
                configs.append({'experiment': 'baseline18', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': 500, 'sigma': 0.0})
    elif args.experiment == 'clean_scarcity':
        for n_train in SCARCITY_NTRAINS:
            for model in STANDARD_MODELS + SPLINE_REG_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'clean_scarcity', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': n_train, 'sigma': 0.0})
    elif args.experiment == 'tied_affine':
        regimes = [(500, 0.0), (500, 0.3), (50, 0.0)]
        for n_train, sigma in regimes:
            for model in TIED_AFFINE_MODELS:
                for seed in SEEDS:
                    configs.append({'experiment': 'tied_affine', 'eq_idx': args.eq_idx, 'model': model, 'seed': seed, 'n_train': n_train, 'sigma': sigma})
    return configs

def main():
    parser = argparse.ArgumentParser(description='Dispatch all configs for one equation')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--eq_idx', type=int, required=True)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--models', type=str, default=None, help="Comma-separated model filter (e.g. 'cheby' or 'tanh,banach')")
    parser.add_argument('--out_dir', type=str, default=None, help='default: results/feynman, or results/feynman_c1 with --c1')
    parser.add_argument('--c1', action='store_true', help='rerun only the J_p models with the C^1 duality map, into a separate directory')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(__file__), 'results', 'feynman_c1' if args.c1 else 'feynman')
    configs = build_configs(args)
    if args.c1:
        configs = [dict(c, c1=True) for c in configs if c['model'] in C1_MODELS]
    if args.models:
        allowed = set(args.models.split(','))
        configs = [c for c in configs if c['model'] in allowed]
    n_configs = len(configs)
    if not n_configs:
        parser.error('No configurations match this experiment/model selection')
    if args.workers < 1 or not 0 <= args.eq_idx < 50:
        parser.error('Expected positive workers and equation index in 0..49')
    print(f'Experiment: {args.experiment} | Equation idx: {args.eq_idx} | Configs: {n_configs} | Workers: {args.workers}', file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f'_{args.models}' if args.models else ''
    out_file = os.path.join(args.out_dir, f'results_{args.experiment}{suffix}_eq{args.eq_idx:02d}.jsonl')
    with Pool(args.workers) as pool:
        results = pool.map(run_one, configs)
    mode = 'a' if args.models else 'w'
    with open(out_file, mode) as f:
        for r in results:
            f.write(r + '\n')
    n_ok = sum((1 for r in results if '"error"' not in r))
    n_err = n_configs - n_ok
    print(f'Done: {n_ok}/{n_configs} succeeded, {n_err} errors. Results: {out_file}', file=sys.stderr)
    if n_err:
        sys.exit(1)
if __name__ == '__main__':
    main()
