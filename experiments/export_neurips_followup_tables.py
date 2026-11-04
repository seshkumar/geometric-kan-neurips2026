import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reporting.neurips_followup_tables import fragments

def main():
    paths = [ROOT / 'experiments/results' / p for p in ('neurips_camera_ready/evidence.json', 'neurips_public_matched/summary.json', 'neurips_followup/extrap15/extrap15_v1/summary.json', 'neurips_followup/tied_affine/tied_affine_v1/summary.json')]
    geometry, matched, extrap, tied = [json.loads(p.read_text()) for p in paths]
    output = fragments(geometry['geometry'], matched, extrap, tied)
    out = ROOT / 'experiments/results/neurips_followup_tables'
    out.mkdir(parents=True, exist_ok=True)
    for name, text in output.items():
        (out / f'{name}.tex').write_text(text)
    evidence = dict(command='python experiments/export_neurips_followup_tables.py', inputs=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths], populations={name: s['inventory'] for name, s in [('extrap15', extrap), ('tied_affine', tied)]}, extrapolation_comparison_family=26, tied_comparison_family_per_condition=3, scope='Unified 15-model, 50-equation extrapolation; separate four-arm, 18-equation tied-affine controls. No historical extrapolation rows pooled.')
    (out / 'evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(out)
if __name__ == '__main__':
    main()
