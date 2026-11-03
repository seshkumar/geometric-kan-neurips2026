# Reproducibility kit — Geometry-Constrained Kolmogorov–Arnold Networks

Code and retained experiment outputs for *Geometry-Constrained Kolmogorov–Arnold Networks: Learning Edge Geometry via Banach Duality*.

The repository is self-contained: its drivers, model library, equation generators and retained outputs support the camera-ready symbolic-regression tables and matched-comparison figure. CPU execution is sufficient; no GPU, external dataset download or access to another research repository is required. New training is separate from regeneration of tables using the supplied outputs.

## Publication reference

K S Sesh Kumar. **Geometry-Constrained Kolmogorov–Arnold Networks: Learning Edge Geometry via Banach Duality.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2026. Accepted as a poster. Preprint: [arXiv:2608.25807](https://arxiv.org/abs/2608.25807). DOI: [10.48550/arXiv.2608.25807](https://doi.org/10.48550/arXiv.2608.25807).

Final proceedings pagination is not supplied. The arXiv link identifies the work; the retained outputs and reporting programs in this repository specify the numerical population for the camera-ready.

```bibtex
@inproceedings{kumar2026geometry,
  author    = {{K S Sesh Kumar}},
  title     = {Geometry-Constrained Kolmogorov--Arnold Networks:
               Learning Edge Geometry via Banach Duality},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Accepted as a poster; preprint arXiv:2608.25807},
  url       = {https://arxiv.org/abs/2608.25807}
}
```

---

## 1. Package contents

| Entry point or directory | Purpose |
|:---|:---|
| `reproduce.py` | Verify the release, regenerate tables, or run one declared fit |
| `experiments/feynman_runner.py` | Core symbolic-regression models, data generation and exact pykan baseline |
| `experiments/feynman_dispatch.py` | Declared per-equation configuration grids |
| `src/matched_models.py` | Matched models, generator and validation-selected training |
| `experiments/export_neurips_camera_ready.py` | Core tables, ablations, exponent summaries and paired intervals |
| `experiments/aggregate_neurips_public.py` | Matched ranks, paired comparisons, extrapolation and figure |
| `src/` | Reusable model, selection, integrity and reporting functions |
| `experiments/results/` | Retained inputs and generated reporting artifacts |
| `release_manifest.json` | File inventory and integrity checks |
| `THIRD_PARTY_NOTICES.md` | Dependency sources and licence information |

The release includes the paper's geometric responses and its declared comparators. It does not include unrelated research projects, private repository history or reviewer correspondence. The pykan baseline and the matched Adam spline remain separate implementations.

---

## 2. Setup

```bash
conda create -n geometric_kan python=3.11 -y
conda activate geometric_kan
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The requirements record direct dependency versions used for local checking. They are not a reconstruction of a historical machine or a claim of bitwise agreement across hardware. No fresh environment installation or full training sweep is implied by the release validation. See the retained validation record for the checks actually performed.

---

## 3. Verify and regenerate the reported artifacts

From the repository root:

```bash
python reproduce.py verify
python reproduce.py tables
```

The first command checks the release inventory. The second reads the supplied per-fit outputs, validates their experimental identities, and regenerates the tables and matched-comparison figure. It does not retrain models.

| Paper content | Reporting program | Generated tree |
|:---|:---|:---|
| Clean core and full-benchmark comparison | `experiments/export_neurips_camera_ready.py` | `experiments/results/neurips_camera_ready/` |
| Noise and noisy scarcity | Same exporter | Same tree: `noise.tex`, `scarcity.tex` and paired intervals |
| Fixed-exponent and branch controls | Same exporter | Same tree: ablation and branch tables |
| Learned-exponent distributions | Same exporter | Same tree: `geometry.tex` |
| Per-equation clean fitting | Same exporter | Same tree: `per_equation.tex` |
| Matched activations and fixed bases | `experiments/aggregate_neurips_public.py` | Public matched-summary output tree printed by the driver |
| Matched figure and extrapolation | Same public matched aggregator | Same tree: tables, numerical summary and `matched_curves.pdf` |

Regeneration refreshes derived reporting artifacts. The retained raw fit records are inputs. Use a separate checkout if you want to preserve derived files byte-for-byte while comparing a different environment.

---

## 4. Run individual fits

New fit commands refuse an existing output file. Choose a fresh path for each fit.

### Core comparison

```bash
python reproduce.py core --experiment baseline50 --eq-idx 0 \
  --seed 1729 --model banach --output fresh_runs/core_banach_eq0_seed1729.jsonl
```

The proposed power branches use the continuously differentiable response. The exact core baseline uses cubic pykan with its declared fixed L-BFGS budget. Cross-validated spline runs retain their training-only folds. Use the experiment grid in `experiments/feynman_dispatch.py` when scheduling the full population.

### Matched comparison

```bash
python reproduce.py matched --condition baseline --eq-idx 0 \
  --seed 1729 --model banach_c1 --output fresh_runs/matched_banach_eq0_seed1729.jsonl
```

The matched protocol uses the same generator and validation-selection rule across its model roster. Its spline is Adam-trained, so its scores are not interchangeable with the core pykan baseline.

### Extrapolation

```bash
python reproduce.py extrapolation --eq-idx 0 --seed 0 --model banach_c1 \
  --output fresh_runs/extrap_banach_eq0_seed0.jsonl
```

The extrapolation experiment uses its separately declared seed population and sampling ranges. It is not pooled with the in-domain runs. Run `python reproduce.py --help` and the relevant subcommand's `--help` for supported conditions. Individual fresh outputs do not silently replace the bundled populations.

---

## 5. Data and selection protocol

Targets are generated from the supplied formulas and sampling ranges. The equation identifiers name implemented functions; the generators are authoritative for this experiment. The [AI Feynman benchmark](https://arxiv.org/abs/1905.11481) provides the originating symbolic-regression context. No measured dataset archive is redistributed.

Training pools and clean test samples are distinct. Target standardisation uses training-pool statistics. Adam checkpoint selection uses a training-derived validation carve, and test labels do not select weights, checkpoints or hyperparameters. The inclusion of validation labels in training-pool standardisation is explicit in the paper; this is not fully nested preprocessing.

In-domain seeds are declared in the generators and configuration files, beginning at 1729. Extrapolation preserves its separate seed list beginning at zero. Seed medians are aggregated within equations before benchmark summaries. Normalised root mean squared error (NRMSE), mean ranks, descriptive wins and paired geometric-mean ratios answer different questions and are not interchangeable.

The matched extrapolation reporting retains its declared family-adjustment size and random-draw allocation, including unreported comparison slots. This preserves the intervals in the paper without releasing outcomes outside the paper's scope. The reporting code records this allocation explicitly.

---

## 6. Compute and interpretation

Each fit runs on CPU. Fits can be scheduled independently by equation, model, seed and condition. Per-fit times are retained where recorded, but these are not a controlled hardware-efficiency benchmark, and a complete total research-compute estimate is not claimed.

The release demonstrates the specified model and reporting protocols. It does not establish general superiority over fixed-basis models, a causal benefit of geometric interpretation, or an optimisation, deep-trainability or adversarial-robustness guarantee. Missing broader-task studies are not represented as completed experiments.

---

## 7. Licence and citation

Original release code and generated result artifacts are distributed under [Apache-2.0](LICENSE), following the same release pattern as the [VCGP reproducibility kit](https://github.com/seshkumar/vcgp-graph-gp). Installed dependencies retain their own licences and bundled notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Manuscript files and the conference style are not included in this code licence.

Please use the full publication reference above, the BibTeX entry, or [CITATION.cff](CITATION.cff) when citing this work.
