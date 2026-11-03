from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import subprocess
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import feynman_dispatch as fd, feynman_runner as fr, aggregate_numbers as an
from src.result_integrity import read_records, grid_inventory
from src.reporting.neurips_camera_ready import describe, paired_ratios, replace_power_rows, summary_cells, clean_noise_table, clean_noise_scarcity_table
OUT = ROOT / 'experiments/results/neurips_camera_ready'
LABELS = {'spline_g3': 'Spline G=3', 'spline_cv_g3': 'CV-regularized G3', 'spline_adam_g3': 'Spline Adam', 'spline_adam_steps10000_g3': 'Spline Adam (10k)', 'tanh': 'Tanh', 'banach': 'Banach', 'lp': '$\\ell^p$', 'mlp': 'MLP', 'cheby': 'Chebyshev', 'banach_tanhonly': 'Banach, power branch off'}
LABELS.update({f'lp_fixed_{p:.1f}': f'$\\ell^p$, fixed $p={p:g}$' for p in (1.5, 2.0, 2.5, 3.0, 4.0)})
LABELS.update({'lp_learned': '$\\ell^p$, learned $p$', 'banach_fixed_2.0': 'Banach, fixed $p=2$', 'banach_learned': 'Banach, learned $p$'})

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {'command': 'python experiments/export_neurips_camera_ready.py', 'revision': 'public-release', 'scope': 'Geometric-response symbolic-regression population', 'provenance_limit': 'Declared input identities are checked before reporting; file hashes accompany generated summaries.', 'inventory': {}, 'inputs': [], 'tables': {}}
    populations = {}
    for name in ('baseline50', 'baseline', 'noise', 'noise_spline_reg', 'scarcity', 'scarcity_spline_reg', 'ablation_p'):
        n = 50 if name == 'baseline50' else 18
        paths = sorted((ROOT / 'experiments/results/feynman').glob(f'results_{name}_eq*.jsonl'))
        rows, files = read_records(paths, allow_console=True)
        expected = [c | {'eq_id': fr.EQUATION_IDS[i]} for i in range(n) for c in fd.build_configs(SimpleNamespace(experiment=name, eq_idx=i))]
        for r in rows + expected:
            r['eq_id'] = an.EQ_ID_CANON.get(r['eq_id'], r['eq_id'])
        inv = grid_inventory(rows, ('eq_id', 'model', 'seed', 'sigma', 'n_train'), expected)
        if not inv['complete_and_valid']:
            raise ValueError(f'{name}: {inv}')
        if name in ('baseline50', 'baseline', 'noise', 'scarcity', 'ablation_p'):
            c1, c1files = read_records((ROOT / 'experiments/results/feynman_c1').glob(f'results_{name}_eq*.jsonl'))
            for r in c1:
                r['eq_id'] = an.EQ_ID_CANON.get(r['eq_id'], r['eq_id'])
            rows, c1inv = replace_power_rows(rows, c1, expected, fd.C1_MODELS, name)
            files += c1files
            evidence.setdefault('c1_inventory', {})[name] = c1inv
            inv = grid_inventory(rows, ('eq_id', 'model', 'seed', 'sigma', 'n_train'), expected)
            if not inv['complete_and_valid']:
                raise ValueError(f'{name}: invalid merged population: {inv}')
        for file in files:
            file['path'] = str(Path(file['path']).relative_to(ROOT))
        evidence['inputs'] += files
        evidence['inventory'][name] = inv
        populations[name] = rows
    evidence['activation'] = 'C1 for every active power branch; unchanged repaired results for other models'

    def matrix(rows, models, eqs):
        per = an.per_equation_medians(rows, models)
        return np.array([[per[e][m] for m in models] for e in eqs])

    def save(name, text):
        (OUT / f'{name}.tex').write_text(text + '\n')

    def table(name, models, columns, values, bold=True, label_overrides=None):
        x = np.asarray(values)
        lines = ['\\begin{tabular}{l' + 'r' * len(columns) + '}', '\\toprule', 'Model & ' + ' & '.join(columns) + ' \\\\', '\\midrule']
        for m, row in zip(models, x):
            cells = ['\\textbf{' + f'{v:.3f}' + '}' if bold and v == x[:, j].min() else f'{v:.3f}' for j, v in enumerate(row)]
            label = (label_overrides or {}).get(m, LABELS.get(m, m.replace('_', '\\_')))
            lines.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        save(name, '\n'.join(lines))
        evidence['tables'][name] = {'models': models, 'columns': columns, 'values': x.tolist()}
    models = an.MODELS
    full = matrix(populations['baseline50'], models, fr.EQUATION_IDS)
    core = matrix(populations['baseline'], models, fr.EQUATION_IDS[:18])
    stats = [describe(full), describe(core)]
    lines = ['\\begin{tabular}{lrrrrrr}', '\\toprule', '& \\multicolumn{3}{c}{Full benchmark} & \\multicolumn{3}{c}{Core subset} \\\\', 'Model & NRMSE & Wins & Rank & NRMSE & Wins & Rank \\\\', '\\midrule']
    for j, m in enumerate(models):
        cells = [' & '.join(summary_cells(s, j)) for s in stats]
        lines.append(LABELS[m] + ' & ' + ' & '.join(cells) + ' \\\\')
    save('baseline', '\n'.join(lines + ['\\bottomrule', '\\end{tabular}']))
    evidence['tables']['baseline'] = {'models': models, 'full': stats[0], 'core': stats[1]}
    comparisons = [('clean50:Banach/Spline', full[:, models.index('banach')], full[:, 0])]
    extended = ['spline_g3', 'spline_cv_g3', 'tanh', 'banach', 'lp', 'mlp', 'cheby']
    for regime, field, conditions in [('noise', 'sigma', fd.NOISE_SIGMAS), ('scarcity', 'n_train', fd.SCARCITY_NTRAINS)]:
        data = populations[regime] + populations[regime + '_spline_reg']
        matrices = [matrix([r for r in data if r[field] == v], extended, fr.EQUATION_IDS[:18]) for v in conditions]
        table(regime, extended, [str(v) for v in conditions], np.array([np.median(x, axis=0) for x in matrices]).T)
        for v, x in zip(conditions, matrices):
            for m in ('banach', 'lp'):
                comparisons.append((f'{regime}/{v}:{m}/SplineCV', x[:, extended.index(m)], x[:, 1]))
        if regime == 'noise':
            evidence['degradation'] = dict(zip(extended, (np.median(matrices[-1], axis=0) / np.median(matrices[0], axis=0)).tolist()))
            core_with_cv = dict(models=extended, **describe(matrices[conditions.index(0.0)]))
    save('clean_noise', clean_noise_table(evidence['tables']['baseline'], evidence['tables']['noise'], LABELS, core=core_with_cv))
    evidence['tables']['clean_noise'] = {'full_clean_source': 'baseline50', 'noise_source': 'noise', 'full_clean_ranking_models': models, 'core_clean_source': {'experiments': ['noise', 'noise_spline_reg'], 'sigma': 0.0}, 'core_clean': core_with_cv, 'noise_models': extended, 'unavailable_full_clean_entries': ['spline_cv_g3'], 'bold_rule': 'Best unrounded column value; minimum error/rank, maximum wins; exact ties bolded'}
    save('clean_noise_scarcity', clean_noise_scarcity_table(evidence['tables']['baseline'], evidence['tables']['noise'], evidence['tables']['scarcity'], LABELS, core_with_cv))
    evidence['tables']['clean_noise_scarcity'] = {'sources': ['baseline', 'clean_noise', 'noise', 'scarcity'], 'scarcity_columns': evidence['tables']['scarcity']['columns'], 'layout': 'Clean panel above shared noise and noisy-scarcity rows; no omitted sample sizes'}
    for sig in (0.0, 0.3):
        ms = ['lp_fixed_1.5', 'lp_fixed_2.0', 'lp_fixed_2.5', 'lp_fixed_3.0', 'lp_fixed_4.0', 'lp_learned', 'banach_fixed_2.0', 'banach_learned']
        x = matrix([r for r in populations['ablation_p'] if r['sigma'] == sig], ms, fr.EQUATION_IDS[:18])
        table('ablation_' + str(sig).replace('.', ''), ms, ['Median NRMSE'], np.median(x, axis=0)[:, None])
        medians = np.median(x, axis=0)
        best_fixed = int(np.argmin(medians[:5]))
        keep = [best_fixed, 5, 6, 7]
        compact = 'ablation_compact_' + str(sig).replace('.', '')
        table(compact, [ms[i] for i in keep], ['Median NRMSE'], medians[keep, None])
        evidence['tables'][compact]['display_selection'] = {'criterion': 'Lowest benchmark median among the five fixed lp exponents, separately per condition; descriptive display only, not validation-selected tuning', 'chosen_fixed_model': ms[best_fixed], 'full_sweep': 'ablation_' + str(sig).replace('.', '')}
        comparisons.append((f'fixed_p/{sig}:learned/fixed2.5', x[:, 5], x[:, 2]))
    ms = ['banach', 'banach_tanhonly', 'tanh', 'lp', 'spline_g3', 'spline_adam_g3', 'spline_adam_steps10000_g3']
    x = matrix(populations['baseline'], ms, fr.EQUATION_IDS[:18])
    table('branch_optimizer', ms, ['Clean NRMSE'], np.median(x, axis=0)[:, None])
    keep = [0, 1, 4, 5]
    table('branch_optimizer_compact', [ms[i] for i in keep], ['Clean NRMSE'], np.median(x, axis=0)[keep, None], label_overrides={'banach_tanhonly': 'Banach, power off'})
    comparisons.append(('branch:Banach/power_off', x[:, 0], x[:, 1]))
    evidence['paired'] = paired_ratios(comparisons)
    evidence['bootstrap'] = {'draws': 100000, 'seed': 20260929, 'family_size': len(comparisons), 'unit': 'equation, after within-equation seed median'}
    lines = ['\\begin{longtable}{lrrr}', '\\caption{Paired geometric-mean NRMSE ratios; below one favours the numerator.}\\label{tab:paired}\\\\', '\\toprule Comparison & Ratio & 95\\% interval & Family interval \\\\ \\midrule \\endhead']
    for r in evidence['paired']:
        display = r['comparison'].replace('SplineCV', LABELS['spline_cv_g3']).replace('_', '\\_')
        lines.append(display + f" & {r['ratio']:.3f} & [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] & [{r['family_ci'][0]:.3f}, {r['family_ci'][1]:.3f}]" + ' \\\\')
    save('paired', '\n'.join(lines + ['\\bottomrule\\end{longtable}']))
    lines = ['\\begin{longtable}{lrrrrrr}', '\\caption{Clean full-benchmark seed-median NRMSE by implemented target identifier.}\\\\', '\\toprule Target & Spline & Tanh & Banach & $\\ell^p$ & MLP & Cheby \\\\ \\midrule \\endhead']
    for e, row in zip(fr.EQUATION_IDS, full):
        lines.append(e + ' & ' + ' & '.join((f'{v:.4f}' for v in row)) + ' \\\\')
    save('per_equation', '\n'.join(lines + ['\\bottomrule\\end{longtable}']))
    geometry = []
    for model in ('lp', 'banach'):
        runs = [r for r in populations['baseline50'] if r['model'] == model]
        ps = np.concatenate([r['p_values'] for r in runs])
        geometry.append({'model': model, 'edges': len(ps), 'median': float(np.median(ps)), 'q05': float(np.quantile(ps, 0.05)), 'q95': float(np.quantile(ps, 0.95)), 'fraction_below2': float(np.mean(ps < 2))})
    evidence['geometry'] = geometry
    table('geometry', [r['model'] for r in geometry], ['Median $p$', '5th pct.', '95th pct.', 'Fraction $p<2$'], [[r[k] for k in ('median', 'q05', 'q95', 'fraction_below2')] for r in geometry], bold=False)
    lines = ['\\begin{tabular}{lrr}', '\\toprule', 'Metric & $\\ell^p$ & Banach \\\\', '\\midrule']
    for label, key in [('Median $p$', 'median'), ('5th pct.', 'q05'), ('95th pct.', 'q95'), ('Fraction $p<2$', 'fraction_below2')]:
        lines.append(label + ' & ' + ' & '.join((f'{r[key]:.3f}' for r in geometry)) + ' \\\\')
    save('geometry_vertical', '\n'.join(lines + ['\\bottomrule', '\\end{tabular}']))
    (OUT / 'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'inventory': evidence['inventory'], 'baseline': evidence['tables']['baseline'], 'degradation': evidence['degradation'], 'paired': evidence['paired']}, indent=2))
if __name__ == '__main__':
    main()
