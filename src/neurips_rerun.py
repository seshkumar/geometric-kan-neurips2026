"""Explicit replacement of canonical NeurIPS result manifests."""
import hashlib
import json
from pathlib import Path


def declare(out, declaration, replace_existing=False):
    """Caller owns the manifest lock. Invalidate derived outputs on replacement."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out/'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if {k:v for k,v in previous.items() if k != 'code_revision'} == {k:v for k,v in declaration.items() if k != 'code_revision'}:
            return
        if not replace_existing:
            raise ValueError('Existing output preserved: explicit --replace-existing required for different source/protocol')
        history = dict(previous=previous, replacement=declaration,
            overwritten_files={str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in out.rglob('*.jsonl')},
            recovery='Superseded committed artifacts remain recoverable through Git history')
        _write(out/'replacement.json', history)
        for name in ('summary.json', 'table.tex'):
            (out/name).unlink(missing_ok=True)
    elif any(p.name != '.manifest.lock' for p in out.iterdir()):
        raise ValueError('Existing output preserved: nonempty result tree has no manifest')
    _write(path, declaration)


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
