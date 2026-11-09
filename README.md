# Reproducibility kit — Geometry-Constrained Kolmogorov–Arnold Networks

Code and retained outputs for *Geometry-Constrained Kolmogorov–Arnold Networks:
Learning Edge Geometry via Banach Duality*.

The repository is self-contained and runs on a single machine. CPU execution is
sufficient; no cluster scheduler is required. `reproduce.py` runs the existing
model-fitting programs or regenerates the paper artifacts from supplied results.

## Publication reference

```bibtex
@inproceedings{kumar2026geometry,
  author    = {{K S Sesh Kumar}},
  title     = {Geometry-Constrained Kolmogorov--Arnold Networks:
               Learning Edge Geometry via Banach Duality},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2608.25807}
}
```

Read the [paper on arXiv](https://arxiv.org/abs/2608.25807).
Citation metadata is also available in [CITATION.cff](CITATION.cff).

---

## 1. Setup

```bash
conda create -n geometric_kan python=3.11 -y
conda activate geometric_kan
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The requirements record the direct dependencies used for release checking.
Different hardware or numerical libraries can produce different training
trajectories; bitwise reproduction is not promised.

## 2. Verify and regenerate the paper results

```bash
python reproduce.py verify
python reproduce.py tables
```

These commands verify the bundled file inventory and regenerate tables and
figures from retained outputs. They do **not** train models or download datasets.
Generated artifacts are written under `experiments/results/`.

## 3. Train models on one machine

Run the first symbolic-regression equation with one worker:

```bash
python reproduce.py run --study symbolic --case-index 0 --workers 1
```

An equation case includes its declared models, conditions and seeds. It is a
full-budget experiment, not a short smoke test. For a single model fit instead:

```bash
python reproduce.py matched --eq-idx 0 --model banach_c1 --seed 1729 \
  --output experiments/results/local_runs/example/banach.json
```

Run the complete symbolic study and summarize it:

```bash
python reproduce.py run --study symbolic --all-cases --workers 4 --aggregate
```

Run the other complete studies:

```bash
python reproduce.py run --study extrapolation --all-cases --workers 4 --aggregate
python reproduce.py run --study pmlb --all-cases --workers 4 --aggregate
```

Cases run sequentially, while `--workers` controls concurrent fits within each
case on the same machine. Choose a worker count appropriate to available CPU
and memory. The established models, seed lists, validation selection and fitting
budgets are unchanged. Symbolic inputs are generated locally. PMLB inputs are
public benchmark data; missing files download through the existing loader into
`experiments/results/pmlb_cache/`.

Preview commands without training or downloading data:

```bash
python reproduce.py run --study symbolic --all-cases --workers 4 --dry-run
```

Summarize an already completed local study without training:

```bash
python reproduce.py run --study symbolic --aggregate
```

Use `--out PATH` to choose a different output tree. Matching completed fits
resume; incompatible sources or protocols are refused. The local runner never
passes the replacement option and rejects the bundled canonical output trees.
Keep the same output path when resuming. Aggregation requires the complete
published case/seed grid; a partial run is not a complete reproduction.

## 4. Outputs and protocol

Fresh study outputs default to:

```text
experiments/results/local_runs/symbolic/
experiments/results/local_runs/extrapolation/
experiments/results/local_runs/pmlb/
```

They contain retained inputs, configuration/source identities and per-fit
metrics. Full-study summaries appear in each output tree. `reproduce.py tables`
continues to use the supplied paper population, rather than silently mixing in
new fits. Fresh-run summaries can be compared separately with the retained
results.

Target normalization and normalized root mean squared error use sample standard
deviation, with correction one. Published seed schedules are preserved,
including the separate extrapolation schedule. Checkpoints and hyperparameters
are selected using training-derived validation data; test labels do not select
fits. Shared reference configurations are fitted once within each declared
population. Equal parameter counts do not imply equal flexibility or compute.

## 5. Package contents

| Entry | Purpose |
|:---|:---|
| `reproduce.py` | Single-machine entry point for fitting, verification and reporting |
| `src/` | Models, data handling, fitting and reporting library |
| `experiments/` | Existing individual experiment and export entry points |
| `experiments/results/` | Retained evidence and generated outputs |
| `release_manifest.json` | File inventory and paper-artifact checks |
| `tests/` | Included implementation and orchestration checks |

For all command-line options, run `python reproduce.py --help` or
`python reproduce.py run --help`. The single-fit commands `core`, `matched` and
`extrapolation` remain available. They require a fresh output file.

## 6. Licence

Original release code and generated outputs use [Apache-2.0](LICENSE).
Third-party software retains its own licences; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream datasets retain their
own terms. Raw PMLB files are downloaded separately rather than redistributed.
