"""Explicit per-model provenance for a targeted comparator replacement."""


def expected_source(record, manifest):
    replacements=manifest.get('model_replacements',{})
    if set(replacements)-{'rational'}:
        raise ValueError('Unreviewed model replacement in canonical manifest')
    replacement=replacements.get(record['model'])
    return replacement['source_fingerprint'] if replacement else manifest['source_fingerprint']


def require_completed_repair(root):
    """Do not publish superseded results after a comparator repair was declared."""
    import json
    from pathlib import Path
    path=Path(root)/'experiments/results/neurips_rational_repair/plan.json'
    if path.exists() and json.loads(path.read_text()).get('status')!='complete':
        raise ValueError('Corrected rational comparator runs are pending; old tables are superseded')
