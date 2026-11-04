from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

def relative_input_records(records, root):
    root = Path(root).resolve()
    result = []
    for record in records:
        path = Path(record['path'])
        if not path.is_absolute():
            path = root / path
        result.append(dict(record, path=os.path.relpath(path.resolve(), root)))
    return result

def read_records(paths, allow_console=False):
    rows, files = ([], [])
    for path in sorted(map(Path, paths)):
        raw = path.read_bytes()
        messages = 0
        count = 0
        for lineno, line in enumerate(raw.decode().splitlines(), 1):
            if not line.strip():
                continue
            if allow_console and (not line.lstrip().startswith('{')):
                messages += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{lineno}: malformed result JSON') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{lineno}: expected a result object')
            rows.append(row)
            count += 1
        files.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'records': count, 'console_lines': messages})
    return (rows, files)

def grid_inventory(rows, fields, expected, metrics=('nrmse',)):
    by = defaultdict(list)
    invalid = []
    for row in rows:
        key = tuple((row.get(field) for field in fields))
        by[key].append(row)
        if row.get('error') or row.get('err') or any((not isinstance(row.get(m), (int, float)) or not math.isfinite(row[m]) for m in metrics)):
            invalid.append(key)
    want = {tuple((row.get(field) for field in fields)) for row in expected}
    missing, extra = (want - set(by), set(by) - want)
    duplicates = [k for k, values in by.items() if len(values) > 1]
    conflicts = [k for k in duplicates if len({json.dumps({m: r.get(m) for m in metrics}, sort_keys=True) for r in by[k]}) > 1]
    return {'records': len(rows), 'unique': len(by), 'expected': len(want), 'missing_count': len(missing), 'missing_examples': sorted(missing, key=str)[:20], 'extra_count': len(extra), 'extra_examples': sorted(extra, key=str)[:20], 'duplicate_keys': len(duplicates), 'conflicting_keys': len(conflicts), 'invalid_count': len(invalid), 'invalid_examples': invalid[:20], 'complete_and_valid': not (missing or extra or duplicates or invalid)}
