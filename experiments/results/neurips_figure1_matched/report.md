# Matched-parameter motivation figure

## Protocol

The declared protocol and full fit states are retained in `config.json` and the per-fit records. Every model has the same number of trainable scalar parameters. Fixed feature definitions are not counted as fitted parameters. The power stack uses the existing continuously differentiable response and a fixed restart stream; restart selection uses only noisy-training mean squared error. This is not an equal-compute comparison.

## Metrics

| Target | Model | Parameters | Noisy train NRMSE | Clean grid NRMSE | Negative variation | Range overshoot | Refit variation RMS |
|:---|:---|---:|---:|---:|---:|---:|---:|
| step | spline | 21 | 0.17471 | 0.16506 | 0.44417 | 0.10441 | 0.00354 |
| step | cheby | 21 | 0.18458 | 0.17629 | 0.84870 | 0.09295 | 0.00333 |
| step | rbf | 21 | 0.16545 | 0.15456 | 0.57487 | 0.10106 | 0.00356 |
| step | haar | 21 | 0.06345 | 0.00668 | 0.03424 | 0.00600 | 0.00349 |
| step | fourier | 21 | 0.21092 | 0.20108 | 1.91924 | 0.09394 | 0.00354 |
| step | lp | 21 | 0.05940 | 0.03174 | 0.02096 | 0.00950 | 0.01285 |
| staircase | spline | 21 | 0.13613 | 0.10695 | 0.24940 | 0.02903 | 0.00275 |
| staircase | cheby | 21 | 0.14105 | 0.11307 | 0.31276 | 0.01977 | 0.00297 |
| staircase | rbf | 21 | 0.14269 | 0.11485 | 0.37074 | 0.03049 | 0.00272 |
| staircase | haar | 21 | 0.17812 | 0.15431 | 0.01882 | 0.00577 | 0.00289 |
| staircase | fourier | 21 | 0.24051 | 0.22429 | 1.52427 | 0.09690 | 0.00276 |
| staircase | lp | 21 | 0.14506 | 0.11447 | 0.13420 | 0.02976 | 0.03218 |

NRMSE means normalised root mean squared error. RMS means root mean square. Error and shape metrics are medians across noisy refits. Refit variation is the root mean of pointwise sample variances, in target units, with the same optimiser initialisation stream across noise draws. Neither the bands nor the seed medians are confidence intervals. Lower negative variation means fewer downward excursions for these nondecreasing targets. A sharp transition does not imply global input-perturbation robustness.

## Reproduction

```bash
python experiments/neurips_figure1_matched.py run --out NEW_DIRECTORY --workers 5
python experiments/neurips_figure1_matched.py export --out NEW_DIRECTORY
```
