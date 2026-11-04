import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_public_matched import export

def main():
    if len(sys.argv) == 1 and (ROOT / 'experiments/results/neurips_sample_sd/symbolic/summary.json').exists():
        from src.reporting.neurips_sample_sd_results import export as canonical_export
        print(canonical_export(ROOT))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'experiments/results/neurips_public_inputs')
    parser.add_argument('--out', type=Path, default=ROOT / 'experiments/results/neurips_public_matched')
    args = parser.parse_args()
    export(args.input, args.out)
    print(args.out)
if __name__ == '__main__':
    main()
