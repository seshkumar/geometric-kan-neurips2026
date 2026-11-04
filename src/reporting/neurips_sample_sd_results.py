import hashlib
import json
from pathlib import Path
import numpy as np
from src.result_integrity import read_records, relative_input_records
from src.reporting.neurips_camera_ready import describe, paired_ratios, summary_cells, matched_error_table, matched_per_equation_table, core_spline_comparison_table, auxiliary_paired_table
from src.reporting.neurips_public_formatting import ROSTER, LABEL, PAIR_TARGETS, curves
from src.reporting.neurips_followup_tables import fragments
SYMBOLIC = 'experiments/results/neurips_sample_sd/symbolic'
EXTRAP = 'experiments/results/neurips_followup/extrap15/extrap15_v1'
MAIN_CELLS = [('clean', 500, 0.0)] + [(f'sigma={s}', 500, s) for s in (0.1, 0.3, 0.5, 1.0)] + [(f'n={n}', n, 0.0) for n in (50, 100, 200, 1000)]
TIED_CONDITIONS = [(500, s) for s in (0.0, 0.1, 0.3, 0.5, 1.0)] + [(n, 0.3) for n in (50, 100, 200, 1000)]
CORE_ALIAS = {'banach': 'banach_c1', 'lp': 'lp_c1', 'spline_g3': 'core_spline_g3', 'spline_cv_g3': 'core_spline_cv_g3', 'banach_tanhonly': 'core_banach_tanhonly', 'spline_adam_g3': 'core_spline_adam_g3', 'spline_adam_steps10000_g3': 'core_spline_adam_steps10000_g3', 'banach_learned': 'banach_c1', 'lp_learned': 'lp_c1', 'banach_fixed_2.0': 'core_banach_fixed_2.0'}
CORE_LABEL = dict(LABEL, banach='Banach', lp='$\\ell^p$', spline_g3='Spline G=3', spline_cv_g3='CV-regularized G3', banach_tanhonly='Banach, power branch off', spline_adam_g3='Spline Adam', spline_adam_steps10000_g3='Spline Adam (10k)', lp_learned='$\\ell^p$, learned $p$', banach_learned='Banach, learned $p$', **{'banach_fixed_2.0': 'Banach, fixed $p=2$'})
CORE_LABEL.update({f'lp_fixed_{p:.1f}': f'$\\ell^p$, fixed $p={p:g}$' for p in (1.5, 2.0, 2.5, 3.0, 4.0)})

def load(root):
    root = Path(root)
    summary = json.loads((root / SYMBOLIC / 'summary.json').read_text())
    extrap = json.loads((root / EXTRAP / 'summary.json').read_text())
    for value in (summary, extrap):
        if not value['inventory']['complete_and_valid'] or not value['protocol']['performance_evidence']:
            raise ValueError('Complete performance populations required')
        if value['protocol'].get('target_sd_ddof') != 1 or value['protocol'].get('metric_ddof') != 1:
            raise ValueError('Sample-SD results required; legacy results cannot be mixed')
        for record in value['inputs']:
            if hashlib.sha256((root / record['path']).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Stale sample-SD summary: ' + record['path'])
    rows, inputs = read_records(sorted((root / SYMBOLIC).glob('eq_*/fits.jsonl')))
    index = {}
    for row in rows:
        key = (row['eq'], row['model'], row['n_train'], row['sigma'], row['seed'])
        if key in index or row.get('target_sd_ddof') != 1 or row.get('metric_ddof') != 1:
            raise ValueError('Duplicate configuration or mixed SD conventions')
        index[key] = row
    return (summary, extrap, index, relative_input_records(inputs, root))

def matrix(index, equations, models, n=500, sigma=0.0):
    canonical = [CORE_ALIAS.get(m, 'core_' + m if m.startswith('lp_fixed_') else m) for m in models]
    values = []
    for eq in equations:
        values.append([np.median([index[eq, m, n, sigma, seed]['nrmse'] for seed in range(1729, 1734)]) for m in canonical])
    return np.asarray(values)

def numeric_table(models, values, headers, labels=CORE_LABEL):
    values = np.asarray(values)
    lines = ['\\begin{tabular}{l' + 'r' * len(headers) + '}', '\\toprule', 'Model & ' + ' & '.join(headers) + ' \\\\', '\\midrule']
    for i, m in enumerate(models):
        cells = ['\\textbf{' + f'{v:.3f}' + '}' if v == values[:, j].min() else f'{v:.3f}' for j, v in enumerate(values[i])]
        lines.append(labels[m] + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule', '\\end{tabular}']) + '\n'

def ranks_table(summary):
    best = {k: min((v[m]['rank'] for m in ROSTER)) for k, v in summary['cells'].items()}
    lines = ['\\begin{tabular}{l' + 'r' * 9 + '}', '\\toprule', '& Clean & \\multicolumn{4}{c}{Noise $\\sigma$} & \\multicolumn{4}{c}{Training size $n$} \\\\', '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}', 'Model & 500 & 0.1 & 0.3 & 0.5 & 1.0 & 50 & 100 & 200 & 1000 \\\\', '\\midrule']
    for m in ROSTER:
        cells = []
        for k, _, _ in MAIN_CELLS:
            v = summary['cells'][k][m]['rank']
            t = f'{v:.1f}'
            cells.append('\\textbf{' + t + '}' if v == best[k] else t)
        lines.append(LABEL[m] + ' & ' + ' & '.join(cells) + ' \\\\')
        if m == 'lp_c1':
            lines.append('\\midrule')
    return '\n'.join(lines + ['\\bottomrule', '\\end{tabular}']) + '\n'

def main_summary(index, equations, inputs, protocol, extrap=None):
    main = dict(command='python experiments/export_neurips_sample_sd.py', roster=ROSTER, cells={}, protocol=protocol, inputs=inputs)
    if extrap is not None:
        main['extrapolation'] = dict(median=dict(zip(extrap['models'], extrap['cells']['n=500/sigma=0']['scorecard']['median'])), paired=extrap['cells']['n=500/sigma=0']['paired'])
    comparisons = []
    for key, n, sigma in MAIN_CELLS:
        x = matrix(index, equations, ROSTER, n, sigma)
        score = describe(x)
        main['cells'][key] = {m: {field: score[field][i] for field in ('median', 'rank', 'wins')} for i, m in enumerate(ROSTER)}
        for a in ('banach_c1', 'lp_c1'):
            for b in PAIR_TARGETS:
                comparisons.append((f'{key}:{a}/{b}', x[:, ROSTER.index(a)], x[:, ROSTER.index(b)]))
    main['paired'] = paired_ratios(comparisons)
    return main

def analyze(root):
    verified, extrap, index, inputs = load(root)
    equations = verified['protocol']['equations']
    core = equations[:18]
    main = main_summary(index, equations, inputs, verified['protocol'], extrap)
    six = ['spline_g3', 'tanh', 'banach', 'lp', 'mlp', 'cheby']
    full = matrix(index, equations, six)
    small = matrix(index, core, six)
    evidence = dict(command=main['command'], protocol=verified['protocol'], inputs=inputs, inventory={'sample_sd_symbolic': verified['inventory']}, c1_inventory={'sample_sd_symbolic': verified['inventory']}, tables={'baseline': dict(models=six, full=describe(full), core=describe(small))})
    groups = []
    paired = [('clean50:Banach/Spline', full[:, six.index('banach')], full[:, 0])]
    seven = ['spline_g3', 'spline_cv_g3', 'tanh', 'banach', 'lp', 'mlp', 'cheby']
    clean7 = describe(matrix(index, core, seven))
    clean7['models'] = seven
    for name, conditions in [('noise', [(500, s) for s in (0.0, 0.1, 0.3, 0.5, 1.0)]), ('scarcity', [(n, 0.3) for n in (50, 100, 200, 500, 1000)])]:
        matrices = [matrix(index, core, seven, n, s) for n, s in conditions]
        values = np.asarray([np.median(x, axis=0) for x in matrices]).T
        columns = [str(s) if name == 'noise' else str(n) for n, s in conditions]
        evidence['tables'][name] = dict(models=seven, columns=columns, values=values.tolist())
        for (n, s), x in zip(conditions, matrices):
            for a in ('banach', 'lp'):
                paired.append((f"{name}/{(s if name == 'noise' else n)}:{a}/SplineCV", x[:, seven.index(a)], x[:, 1]))
    fixed = [f'lp_fixed_{p:.1f}' for p in (1.5, 2.0, 2.5, 3.0, 4.0)] + ['lp_learned', 'banach_fixed_2.0', 'banach_learned']
    tables = {}
    for sigma in (0.0, 0.3):
        x = matrix(index, core, fixed, 500, sigma)
        med = np.median(x, axis=0)
        suffix = str(sigma).replace('.', '')
        tables['ablation_' + suffix] = numeric_table(fixed, med[:, None], ['Median NRMSE'])
        best = int(np.argmin(med[:5]))
        keep = [best, 5, 6, 7]
        tables['ablation_compact_' + suffix] = numeric_table([fixed[i] for i in keep], med[keep, None], ['Median NRMSE'])
        evidence['tables']['ablation_' + suffix] = dict(models=fixed, columns=['Median NRMSE'], values=med[:, None].tolist())
        paired.append((f'fixed_p/{sigma}:learned/fixed2.5', x[:, 5], x[:, 2]))
    branch = ['banach', 'banach_tanhonly', 'tanh', 'lp', 'spline_g3', 'spline_adam_g3', 'spline_adam_steps10000_g3']
    bmatrix = matrix(index, core, branch)
    bmedian = np.median(bmatrix, axis=0)
    tables['branch_optimizer'] = numeric_table(branch, bmedian[:, None], ['Clean NRMSE'])
    keep = [0, 1, 4, 5]
    tables['branch_optimizer_compact'] = numeric_table([branch[i] for i in keep], bmedian[keep, None], ['Clean NRMSE'], dict(CORE_LABEL, banach_tanhonly='Banach, power off'))
    evidence['tables']['branch_optimizer'] = dict(models=branch, columns=['Clean NRMSE'], values=bmedian[:, None].tolist())
    paired.append(('branch:Banach/power_off', bmatrix[:, 0], bmatrix[:, 1]))
    evidence['paired'] = paired_ratios(paired)
    evidence['bootstrap'] = dict(draws=100000, seed=20260929, family_size=len(paired), unit='equation after within-equation seed median')
    tables['paired'] = auxiliary_paired_table(evidence['paired']) + '\n'
    tables['core_spline_comparison'] = core_spline_comparison_table(clean7, evidence['tables']['noise'], evidence['tables']['scarcity'], CORE_LABEL) + '\n'
    for name in ('noise', 'scarcity'):
        t = evidence['tables'][name]
        tables[name] = numeric_table(t['models'], t['values'], t['columns'])
    geometry = []
    for display, model in [('lp', 'lp_c1'), ('banach', 'banach_c1')]:
        ps = np.concatenate([index[e, model, 500, 0.0, s]['p_values'] for e in equations for s in range(1729, 1734)])
        if not np.isfinite(ps).all() or (ps <= 1).any():
            raise ValueError('Invalid retained exponents')
        geometry.append(dict(model=display, edges=len(ps), median=float(np.median(ps)), q05=float(np.quantile(ps, 0.05)), q95=float(np.quantile(ps, 0.95)), fraction_below2=float(np.mean(ps < 2))))
    evidence['geometry'] = geometry
    tied_models = ['independent_c1', 'tied_c1_matched', 'tied_c1', 'tied_c1_p2_init']
    tied = dict(protocol=verified['protocol'] | {'performance_evidence': True, 'comparison_family': 'Three comparisons separately within each condition'}, inventory=verified['inventory'], models=tied_models, equations=core, cells={}, inputs=inputs, canonical_reference='banach_c1; same fit identities as learned Banach controls')
    global_tied = []
    for n, sigma in TIED_CONDITIONS:
        key = f'n={n}/sigma={sigma:g}'
        x = matrix(index, core, ['banach_c1'] + tied_models[1:], n, sigma)
        pairs = [(f'{m}/independent_c1', x[:, i], x[:, 0]) for i, m in enumerate(tied_models) if i]
        tied['cells'][key] = dict(values=x.tolist(), scorecard=describe(x), paired=paired_ratios(pairs))
        global_tied += [(key + ':' + label, a, b) for label, a, b in pairs]
    tied['across_condition_sensitivity'] = paired_ratios(global_tied)
    reference = []
    for eq in core:
        for seed in range(1729, 1734):
            r = index[eq, 'banach_c1', 500, 0.3, seed]
            reference.append(dict(eq=eq, seed=seed, reference_id=r['reference_id'], nrmse=r['nrmse']))
    control = evidence['tables']['ablation_03']['values'][-1][0]
    affine = tied['cells']['n=500/sigma=0.3']['scorecard']['median'][0]
    if control != affine:
        raise ValueError('Repeated Banach references differ')
    return dict(main=main, core=evidence, tied=tied, extrap=extrap, tables=tables, clean_matrix=matrix(index, equations, ROSTER).tolist(), references=dict(identical=True, median=control, fits=reference), inputs=inputs)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def result_values(data):
    main, core, tied, extrap = (data[k] for k in ('main', 'core', 'tied', 'extrap'))
    values = {}
    for prefix, model in [('Spline', 'spline'), ('Banach', 'banach_c1'), ('Lp', 'lp_c1'), ('Additive', 'pow')]:
        for field, ending, fmt in [('median', 'Median', '.3f'), ('rank', 'Rank', '.2f'), ('wins', 'Wins', 'g')]:
            values['Clean' + prefix + ending] = format(main['cells']['clean'][model][field], fmt)
        for condition, tag in [('sigma=0.3', 'Moderate'), ('sigma=1.0', 'Heavy'), ('n=50', 'Scarce')]:
            values[tag + prefix + 'Median'] = format(main['cells'][condition][model]['median'], '.3f')
    look = {p['comparison']: p for p in main['paired']}
    for name, key in [('CleanTanhRatio', 'clean:banach_c1/tanh'), ('CleanSlopeRatio', 'clean:banach_c1/adaslope'), ('CleanAdditiveRatio', 'clean:banach_c1/pow'), ('HeavyPowerRatio', 'sigma=1.0:lp_c1/pow'), ('ScarceSplineRatio', 'n=50:banach_c1/spline'), ('LargeSplineRatio', 'n=1000:banach_c1/spline')]:
        values[name] = format(look[key]['ratio'], '.2f')
    values['ScarceBanachRank'] = format(main['cells']['n=50']['banach_c1']['rank'], '.1f')
    for table, model, name in [('ablation_03', 'lp_learned', 'CoreLearnedLp'), ('ablation_03', 'lp_fixed_2.5', 'CoreFixedLp'), ('ablation_03', 'banach_learned', 'CoreLearnedBanach'), ('ablation_03', 'banach_fixed_2.0', 'CoreFixedBanach'), ('branch_optimizer', 'banach', 'CoreCleanBanach'), ('branch_optimizer', 'banach_tanhonly', 'CorePowerOff'), ('branch_optimizer', 'spline_g3', 'CoreSpline'), ('branch_optimizer', 'spline_adam_g3', 'CoreSplineAdam'), ('branch_optimizer', 'spline_adam_steps10000_g3', 'CoreSplineLong')]:
        t = core['tables'][table]
        values[name] = format(t['values'][t['models'].index(model)][0], '.3f')
    for model, prefix in [('lp', 'Lp'), ('banach', 'Banach')]:
        g = next((g for g in core['geometry'] if g['model'] == model))
        values[prefix + 'ExponentMedian'] = format(g['median'], '.2f')
        values[prefix + 'BelowTwoPercent'] = format(100 * g['fraction_below2'], '.0f') + '\\%'
    es = extrap['cells']['n=500/sigma=0']
    for model, name in [('banach_c1', 'Banach'), ('spline', 'Spline'), ('tanh', 'Tanh')]:
        values['Extrap' + name + 'Median'] = format(es['scorecard']['median'][extrap['models'].index(model)], '.3f')
    el = {p['comparison']: p for p in es['paired']}
    for model, name in [('spline', 'Spline'), ('tanh', 'Tanh')]:
        values['Extrap' + name + 'Ratio'] = format(el['banach_c1/' + model]['ratio'], '.2f')
    for table, column, models in [('noise', 4, [('spline_g3', 'CoreHeavySpline'), ('spline_cv_g3', 'CoreHeavyCV'), ('lp', 'CoreHeavyLp')]), ('scarcity', 0, [('lp', 'CoreScarceLp'), ('spline_cv_g3', 'CoreScarceCV'), ('banach', 'CoreScarceBanach')])]:
        t = core['tables'][table]
        for model, name in models:
            values[name] = format(t['values'][t['models'].index(model)][column], '.3f')
    for condition, name in [('n=500/sigma=0', 'TiedClean'), ('n=500/sigma=0.5', 'TiedNoise')]:
        q = tied['cells'][condition]['paired'][0]
        values[name + 'Ratio'] = format(q['ratio'], '.3f')
        values[name + 'Interval'] = '[' + ', '.join((format(v, '.3f') for v in q['family_ci'])) + ']'
    return values

def audit(root):
    from src.neurips_sample_sd import cases, plan, FIELDS
    from src.result_integrity import grid_inventory
    root = Path(root)
    result = {}
    symbolic_rows = None
    symbolic_inputs = None
    for profile, directory in [('symbolic', SYMBOLIC), ('extrapolation', EXTRAP)]:
        out = root / directory
        manifest = json.loads((out / 'protocol.json').read_text())
        if manifest['protocol'] != plan(profile):
            raise ValueError('Unexpected protocol')
        rows, inputs = read_records(sorted(out.glob('eq_*/fits.jsonl')))
        equations = manifest['protocol']['equations']
        expected = [dict(eq=e, model=m, n_train=n, sigma=s, seed=seed) for i, e in enumerate(equations) for m, n, s in cases(profile, i) for seed in plan(profile)['seeds']]
        check = grid_inventory(rows, FIELDS, expected, metrics=('nrmse', 'train_time_s'))
        input_hashes = {}
        for r in rows:
            if r.get('source_fingerprint') != manifest['source_fingerprint'] or r.get('target_sd_ddof') != 1 or r.get('metric_ddof') != 1:
                raise ValueError('Mixed source or SD metadata')
            i = equations.index(r['eq'])
            path = out / f'eq_{i:02d}' / r['input_file']
            if path not in input_hashes:
                input_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
            if input_hashes[path] != r['input_sha256']:
                raise ValueError('Changed retained input')
        result[profile] = dict(inventory=check, inputs=relative_input_records(inputs, root), normalization_verified=True, input_hashes_verified=True, failed_records=[r for r in rows if r.get('diverged') or r.get('nrmse') is None])
        if profile == 'symbolic':
            symbolic_rows, symbolic_inputs = (rows, relative_input_records(inputs, root))
    index = {(r['eq'], r['model'], r['n_train'], r['sigma'], r['seed']): r for r in symbolic_rows}
    expected_main = [dict(eq=e, model=m, n_train=n, sigma=s, seed=seed) for e in plan('symbolic')['equations'] for m in ROSTER for _, n, s in MAIN_CELLS for seed in plan('symbolic')['seeds']]
    selected_main = [r for r in symbolic_rows if r['model'] in ROSTER and any(((r['n_train'], r['sigma']) == (n, s) for _, n, s in MAIN_CELLS))]
    main_inventory = grid_inventory(selected_main, FIELDS, expected_main, metrics=('nrmse', 'train_time_s'))
    if not main_inventory['complete_and_valid']:
        raise ValueError('The main comparison itself is incomplete or invalid')
    main = main_summary(index, plan('symbolic')['equations'], symbolic_inputs, plan('symbolic') | {'performance_evidence': True, 'scope': 'Main comparison only'})
    main['inventory'] = main_inventory
    out = root / 'experiments/results/neurips_sample_sd_results'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'inventory.json', result)
    write_json(out / 'main_complete.json', main)
    return result

def export(root):
    root = Path(root)
    data = analyze(root)
    out = root / 'experiments/results/neurips_sample_sd_results'
    out.mkdir(parents=True, exist_ok=True)
    for key in ('main', 'core', 'tied', 'references'):
        write_json(out / (key + '.json'), data[key])
    values = result_values(data)
    write_json(out / 'results_values.json', values)
    write_json(out / 'evidence.json', dict(command='python experiments/export_neurips_sample_sd.py', inputs=data['inputs'], protocol=data['main']['protocol'], repeated_references_identical=data['references']['identical'], active_populations=[SYMBOLIC, EXTRAP, 'experiments/results/neurips_pmlb15/pmlb15_pbs_v1']))
    core_out = root / 'experiments/results/neurips_camera_ready'
    core_out.mkdir(parents=True, exist_ok=True)
    active = set(data['tables'])
    for path in core_out.glob('*.tex'):
        if path.stem not in active:
            path.unlink()
    write_json(core_out / 'evidence.json', data['core'])
    for name, text in data['tables'].items():
        (core_out / (name + '.tex')).write_text(text)
    (core_out / 'results_values.tex').write_text(''.join(('\\newcommand{\\' + name + '}{' + value + '}\n' for name, value in values.items())))
    matched = root / 'experiments/results/neurips_public_matched'
    matched.mkdir(parents=True, exist_ok=True)
    write_json(matched / 'summary.json', data['main'])
    (matched / 'tab_matched_nrmse.tex').write_text(matched_error_table(data['main']) + '\n')
    (matched / 'tab_matched_ranks.tex').write_text(ranks_table(data['main']))
    (matched / 'tab_matched_per_equation.tex').write_text(matched_per_equation_table(data['clean_matrix'], data['main']['protocol']['equations']) + '\n')
    curves(data['main'], matched)
    ft = root / 'experiments/results/neurips_followup_tables'
    ft.mkdir(parents=True, exist_ok=True)
    for name, text in fragments(data['core']['geometry'], data['main'], data['extrap'], data['tied']).items():
        (ft / (name + '.tex')).write_text(text)
    write_json(ft / 'evidence.json', dict(command='python experiments/export_neurips_sample_sd.py', inputs=data['inputs'], populations={'symbolic': data['core']['inventory']['sample_sd_symbolic'], 'extrap15': data['extrap']['inventory']}, extrapolation_comparison_family=26, tied_comparison_family_per_condition=3))
    comparison = root / 'experiments/results/neurips_comparison_table'
    comparison.mkdir(parents=True, exist_ok=True)
    for name in ('comparison', 'matched_nrmse'):
        (comparison / (name + '.tex')).write_text(matched_error_table(data['main']) + '\n')
    (comparison / 'geometry_extrapolation.tex').write_bytes((ft / 'geometry_extrapolation.tex').read_bytes())
    write_json(comparison / 'evidence.json', dict(command='python experiments/export_neurips_sample_sd.py', inputs=data['inputs'], roster=ROSTER, cells=data['main']['cells'], scope='Complete canonical sample-SD population; references fitted once'))
    doc = root / 'docs/neurips2026/camera_ready_tables'
    if doc.exists():
        for path in core_out.glob('*.tex'):
            (doc / path.name).write_bytes(path.read_bytes())
        for path in ft.glob('*.tex'):
            (doc / path.name).write_bytes(path.read_bytes())
        for source, destination in [('tab_matched_nrmse.tex', 'comparison.tex'), ('tab_matched_ranks.tex', 'matched_ranks.tex'), ('tab_matched_per_equation.tex', 'matched_per_equation.tex')]:
            (doc / destination).write_bytes((matched / source).read_bytes())
        (root / 'docs/neurips2026/figs/matched_curves.pdf').write_bytes((matched / 'matched_curves.pdf').read_bytes())
    return out
