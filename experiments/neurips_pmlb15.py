import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.neurips_pmlb import protocol, run, summarize
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'summarize'))
    parser.add_argument('--dataset-idx', type=int, choices=range(12))
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/results/neurips_pmlb15/pmlb15_v1')
    parser.add_argument('--cache', type=Path, default=ROOT / 'experiments/results/pmlb_cache')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--replace-existing', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(protocol(), indent=2))
    elif args.command == 'run':
        if args.dataset_idx is None or args.workers < 1:
            parser.error('run requires --dataset-idx and positive workers')
        run(args.dataset_idx, args.out, args.cache, args.workers, args.replace_existing)
    else:
        print(json.dumps(summarize(args.out), indent=2))
