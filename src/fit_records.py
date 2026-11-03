from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from torch.optim.optimizer import register_optimizer_step_post_hook
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from src import matched_models as rb

def source_stamp(paths):
    digest = hashlib.sha256()
    for path in sorted({Path(p).resolve() for p in paths}):
        digest.update(str(path.relative_to(ROOT)).encode() + b'\x00' + path.read_bytes() + b'\x00')
    return digest.hexdigest()

def train_counted(model, *args, **kwargs):
    count = [0]
    handle = register_optimizer_step_post_hook(lambda *_: count.__setitem__(0, count[0] + 1))
    try:
        rb.train_one(model, *args, **kwargs)
    finally:
        handle.remove()
    return count[0]

def write_shard(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: tuple((str(r[f]) for f in fields)))
    tmp = path.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(''.join((json.dumps(r) + '\n' for r in rows)))
    os.replace(tmp, path)
