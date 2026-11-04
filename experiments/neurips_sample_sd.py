import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.neurips_sample_sd import plan, run, summarize
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'run', 'summarize'))
    p.add_argument('--profile', choices=('symbolic', 'extrapolation'), required=True)
    p.add_argument('--case-index', type=int, choices=range(50))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--replace-existing', action='store_true')
    a = p.parse_args()
    if a.command == 'plan':
        result = plan(a.profile)
    elif a.command == 'run':
        if a.case_index is None or a.workers < 1:
            p.error('case index and positive workers required')
        result = run(a.profile, a.case_index, a.out, a.workers, a.replace_existing)
    else:
        result = summarize(a.profile, a.out)
    print(json.dumps(result, indent=2))
