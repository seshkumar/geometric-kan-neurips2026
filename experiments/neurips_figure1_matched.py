import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_figure1_study import run, export
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'export'))
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/results/neurips_figure1_matched')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out, args.workers)
    else:
        export(args.out)
