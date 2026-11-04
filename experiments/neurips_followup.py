import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.neurips_followup import specification, run, summarize
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'summarize'))
    parser.add_argument('--study', required=True, choices=('extrap15', 'tied_affine'))
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    a = parser.parse_args()
    if a.command == 'plan':
        print(json.dumps(specification(a.study), indent=2))
    else:
        if a.out is None:
            parser.error('--out required')
        if a.command == 'run':
            if a.case_index is None:
                parser.error('--case-index required')
            run(a.study, a.case_index, a.out, a.workers)
        else:
            print(json.dumps(summarize(a.study, a.out), indent=2))
