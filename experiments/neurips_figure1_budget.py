import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_figure1_budget import run, export
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'export'))
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    result = run(ROOT, a.workers) if a.command == 'run' else export(ROOT)
    print('Verified matched budgets:', result['budgets'])
