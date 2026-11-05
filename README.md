# Reproducibility kit — Geometry-Constrained Kolmogorov–Arnold Networks

Code and retained experiment outputs for *Geometry-Constrained Kolmogorov–Arnold Networks: Learning Edge Geometry via Banach Duality*.

The repository is self-contained: its drivers, model library, equation generators and retained outputs support the camera-ready symbolic-regression and PMLB tables and motivation figure. CPU execution is sufficient. Regeneration uses retained outputs; fresh PMLB training downloads missing inputs through the official PMLB loader. New training is separate from regeneration of tables using the supplied outputs.

## Sample-standard-deviation protocol

Fresh experiments now use sample standard deviation (SD, correction one) for target normalization and normalized root mean squared error. PMLB feature scaling also uses sample SD. In-domain symbolic studies share the original core generator and target-normalization routine, so overlapping configurations receive identical inputs and labels. Existing model families, seeds, splits, optimizers and fitting budgets are retained.

The bundled results and regenerated tables use sample SD. Every repeated Banach reference reads one canonical fit. The retained repair audit preserves the failed extrapolation predecessor and its first unchanged-protocol local replay; its very large finite error is included. Completed valid fits are unchanged. No claim of bitwise agreement across hardware is made.

The unified symbolic rerun fits repeated configurations once. Its learned Banach row supplies both the exponent-control and affine-control references. Fresh outputs use one canonical tree; explicit replacement records the previous manifest and invalidates stale summaries. Incomplete or mixed-convention populations are rejected.

On a PBS cluster, submit from the repository root after preserving previous results in Git:

```bash
mkdir -p logs
qsub -N nips-sd-main pbs/run_neurips_sample_sd.pbs
qsub -N nips-sd-extra -v SAMPLE_SD_PROFILE=extrapolation pbs/run_neurips_sample_sd.pbs
qsub -N nips-sd-pmlb pbs/run_neurips_sample_sd_pmlb.pbs
```

These jobs do not submit other jobs. Extrapolation and PMLB replace their existing canonical result trees; the symbolic study uses `experiments/results/neurips_sample_sd/symbolic/` as its single replacement population. It supplies all main, exponent, branch, spline and affine comparisons, rather than fitting separate reference rows for each table. The publication update must use this complete population and overwrite the existing table artifacts after validation.

## Publication reference

```bibtex
@inproceedings{kumar2026geometry,
  author    = {{K S Sesh Kumar}},
  title     = {Geometry-Constrained Kolmogorov--Arnold Networks:
               Learning Edge Geometry via Banach Duality},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
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
| `experiments/aggregate_neurips_public.py` | In-domain matched ranks, paired comparisons and figure |
| `experiments/neurips_followup.py` | Unified 15-model extrapolation and four-arm tied-affine runs and validation |
| `experiments/export_neurips_followup_tables.py` | Current extrapolation and affine-control tables, including updated paired intervals |
| `experiments/neurips_figure1_matched.py` | Exactly matched noisy step/staircase motivation study |
| `src/neurips_normalization.py` | Shared sample-SD normalization |
| `experiments/neurips_sample_sd.py` | Deduplicated canonical rerun and validation |
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
| Learned-exponent distributions | Same exporter | `experiments/results/neurips_sample_sd_results/core.json` and `neurips_followup_tables/geometry_extrapolation.tex` |
| Per-equation clean fitting | Same exporter | `experiments/results/neurips_public_matched/tab_matched_per_equation.tex` |
| Matched activations and fixed bases | `experiments/aggregate_neurips_public.py` | Public matched-summary output tree printed by the driver |
| Matched figure and extrapolation | Same public matched aggregator | Same tree: tables, numerical summary and `matched_curves.pdf` |
| Figure 1 and shape diagnostics | `experiments/neurips_figure1_matched.py export`; `experiments/export_neurips_figure1_manuscript.py` | `experiments/results/neurips_figure1_matched/` |

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

Original release code and generated result artifacts are distributed under [Apache-2.0](LICENSE). Installed dependencies retain their own licences and bundled notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Manuscript files and the conference style are not included in this code licence.

Please use the BibTeX entry above when citing this work.

## 8. Parameter-matched motivation figure

The step and staircase illustration uses B-spline, Chebyshev, Gaussian radial-basis, Wavelet (Haar), Fourier and power-response constructions. Each fit has exactly 21 trainable scalars. Inputs, noisy labels, coefficients, nonlinear states and predictions are retained. This motivates studying fitted response shapes; it does not claim a universally preferable family, equal fitting compute or general robustness.

Regenerate the figure and its descriptive diagnostics without fitting:

```bash
python experiments/neurips_figure1_matched.py export
python experiments/export_neurips_figure1_manuscript.py
```

To repeat the declared fitting protocol in a fresh directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/neurips_figure1_matched.py run --out fresh_runs/figure1 --workers 5
python experiments/neurips_figure1_matched.py export --out fresh_runs/figure1
```

Noise seeds and the fixed optimiser-initialisation stream are declared in the configuration. Restart selection uses noisy-training error only. Clean-grid errors and refit variation are supplemental diagnostics. The main annotations are noisy-training NRMSE; pointwise bands are ranges across refits, not confidence intervals.

## PMLB tabular regression

The retained comparison contains all 15 models on the 12 declared datasets. Regenerate the manuscript tables without fitting:

```bash
python experiments/neurips_pmlb15.py summarize --out experiments/results/neurips_pmlb15/pmlb15_pbs_v1
python experiments/export_neurips_pmlb_tables.py
```

For new training, choose a fresh output directory and run every dataset index from 0 through 11:

```bash
python experiments/neurips_pmlb15.py run --dataset-idx 0 --out experiments/results/neurips_pmlb15/fresh_v1 --workers 4
```

Missing inputs download into the shared PMLB cache. The original PMLB split and population-standard-deviation error convention are retained. Checkpoints use training-derived validation labels; held-out test labels do not select fits. Summarization requires the complete population. Generated main-text wins and ranks use the five displayed models; appendix panels use all 15 models together. Full dataset identifiers and the retained protocol accompany the tables. Upstream datasets retain their own terms; raw dataset files are not redistributed in this kit.

## Additive epsilon comparator label

The model key `pow` is displayed as ℓp (additive ε) in the paper and tables. It learns a positive exponent through softplus coordinates and uses the additive-epsilon response. The proposed C1 response uses exponential exponent coordinates. Legacy additive-map controls used for the separate extrapolation map comparison retain their original exponential coordinates; these are distinguished in the paper. This is a display-name clarification and does not alter fitted models or scores.

## Unified extrapolation and affine-sharing controls

The active extrapolation table uses the complete 15-model population. The tied-affine comparison has four arms on the core subset; its independent Banach reference is the same canonical fit used by the exponent controls. Signed gains remain unconstrained; tying the affines alone does not guarantee a convex potential. Every arm learns the exponent, including the arm initialised at exponent two.

`python reproduce.py tables` validates both populations, checks retained input hashes and regenerates the current tables. To regenerate just these studies without training:

```bash
python experiments/neurips_sample_sd.py summarize --profile symbolic --out experiments/results/neurips_sample_sd/symbolic
python experiments/neurips_sample_sd.py summarize --profile extrapolation --out experiments/results/neurips_followup/extrap15/extrap15_v1
python experiments/export_neurips_sample_sd.py
```

For a fresh declared per-equation run, use a new output tree:

```bash
python experiments/neurips_followup.py run --study tied_affine --case-index 0 --out fresh_runs/tied_affine --workers 4
```

The same driver accepts `--study extrap15`. Existing bundled result trees are not fresh-run destinations. The extrapolation bootstrap family has 26 actual contrasts; the affine controls use three contrasts separately within each condition. These analyses do not establish uniform superiority across tasks or noise levels.

## Motivation budget sweep

The appendix extends the noisy scalar illustration to exactly matched parameter budgets, reusing the retained figure column. It is not a test-set or equal-compute comparison. Regenerate it without fitting:

```bash
python experiments/neurips_figure1_budget.py export
```

Run missing budget fits into the declared study with `python experiments/neurips_figure1_budget.py run`; completed per-model records are reused.
