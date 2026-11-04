from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_sample_sd_results import export, audit
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only', action='store_true')
    a = p.parse_args()
    print(audit(ROOT) if a.audit_only else export(ROOT))
