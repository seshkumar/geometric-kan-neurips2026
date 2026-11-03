import os
import json
import glob
import statistics
import collections
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'feynman')
MODELS = ['spline_g3', 'tanh', 'banach', 'lp', 'mlp', 'cheby']
MODEL_LABELS = {'spline_g3': 'Spline G=3', 'tanh': 'Tanh', 'banach': 'Banach', 'lp': 'L^p', 'mlp': 'MLP', 'cheby': 'Cheby'}
EQ_ID_CANON = {'I.34.1': 'I.34.10'}
_IDENTITY_FIELDS = ('eq_id', 'model', 'seed', 'sigma', 'n_train', 'condition')

def _identity(r):
    return tuple((r.get(f) for f in _IDENTITY_FIELDS))

def deduplicate(rows):
    seen, out, conflicts = ({}, [], [])
    for r in rows:
        if not all((k in r for k in ('eq_id', 'model', 'seed', 'nrmse'))):
            out.append(r)
            continue
        key = _identity(r)
        if key in seen:
            if abs(seen[key] - r['nrmse']) > 1e-09 * (1 + abs(seen[key])):
                conflicts.append((key, seen[key], r['nrmse']))
            continue
        seen[key] = r['nrmse']
        out.append(r)
    if conflicts:
        head = '; '.join((f'{k} -> {a} vs {b}' for k, a, b in conflicts[:5]))
        raise ValueError(f'aggregate_numbers.deduplicate: {len(conflicts)} conflicting duplicate run(s) -- identical (eq_id,model,seed,sigma,n_train,condition) but different nrmse (e.g. {head}). Rebuild the source runs to a canonical manifest before aggregating; refusing to pick an arbitrary winner.')
    return out

def load_jsonl(*patterns):
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith('{'):
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'eq_id' in r:
                        r['eq_id'] = EQ_ID_CANON.get(r['eq_id'], r['eq_id'])
                    rows.append(r)
    return deduplicate(rows)

def median_by(rows, key_fn):
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r['nrmse'])
    return {k: statistics.median(v) for k, v in buckets.items()}

def per_equation_medians(rows, models=MODELS):
    out = collections.defaultdict(dict)
    bucket = collections.defaultdict(list)
    for r in rows:
        if 'eq_id' not in r or 'nrmse' not in r or 'model' not in r:
            continue
        if r['model'] not in models:
            continue
        bucket[r['eq_id'], r['model']].append(r['nrmse'])
    for (eq, m), vals in bucket.items():
        out[eq][m] = statistics.median(vals)
    return dict(out)

def win_counts(per_eq, models=MODELS):
    wins = {m: 0 for m in models}
    for eq, d in per_eq.items():
        vals = {m: d[m] for m in models if m in d}
        if not vals:
            continue
        winner = min(vals, key=lambda m: vals[m])
        wins[winner] += 1
    return wins

def within_5pct(per_eq, models=MODELS):
    out = {m: 0 for m in models}
    for eq, d in per_eq.items():
        vals = {m: d[m] for m in models if m in d}
        if not vals:
            continue
        best = min(vals.values())
        for m, v in vals.items():
            if v <= 1.05 * best:
                out[m] += 1
    return out

def avg_rank(per_eq, models=MODELS):
    sums = {m: 0.0 for m in models}
    counts = {m: 0 for m in models}
    for eq, d in per_eq.items():
        present = [m for m in models if m in d]
        if len(present) < 2:
            continue
        order = sorted(present, key=lambda m: d[m])
        for rank, m in enumerate(order, 1):
            sums[m] += rank
            counts[m] += 1
    return {m: sums[m] / counts[m] if counts[m] else float('nan') for m in models}

def print_model_row(label, fn_result, fmt='.3f'):
    print(f'  {label:20s}', end='')
    for m in MODELS:
        v = fn_result.get(m, float('nan'))
        if isinstance(v, int):
            print(f' {v:>7d}', end='')
        else:
            print(f' {v:>7{fmt}}', end='')
    print()

def main():
    print('=' * 80)
    print('BASELINE50 (50 eq, n=500, clean, 6 models)')
    print('=' * 80)
    rows = load_jsonl('results_baseline50_eq*.jsonl', 'results_baseline50_cheby_eq*.jsonl', 'results_baseline50_banach_eq*.jsonl')
    per_eq = per_equation_medians(rows)
    print(f'  Equations found: {len(per_eq)}')
    all_medians = collections.defaultdict(list)
    for eq, d in per_eq.items():
        for m in MODELS:
            if m in d:
                all_medians[m].append(d[m])
    med_per_model = {m: statistics.median(all_medians[m]) for m in MODELS}
    print_model_row('Median NRMSE', med_per_model)
    print_model_row('Wins (/50)', win_counts(per_eq))
    print_model_row('Within 5% of best', within_5pct(per_eq))
    print_model_row('Avg rank', avg_rank(per_eq))
    eq_dim = {}
    for r in rows:
        if 'eq_id' in r and 'n_vars' in r:
            eq_dim[r['eq_id']] = r['n_vars']
    dim_counts = collections.Counter(eq_dim.values())
    print(f'\n  Equations by dimension: {dict(sorted(dim_counts.items()))}')
    print('\n  Dimension wins:')
    for d in sorted(dim_counts):
        eq_at_d = {eq: per_eq[eq] for eq in per_eq if eq_dim.get(eq) == d}
        w = win_counts(eq_at_d)
        print(f'  {d}D ({dim_counts[d]} eq)', end='')
        for m in MODELS:
            print(f' {w[m]:>4d}', end='')
        print()
    print('\n' + '=' * 80)
    print('NOISE (18 eq core, σ×model medians)')
    print('=' * 80)
    rows_noise = load_jsonl('results_noise_eq*.jsonl', 'results_noise_cheby_eq*.jsonl')
    sigmas = sorted(set((r['sigma'] for r in rows_noise if 'sigma' in r)))
    for s in sigmas:
        sub = [r for r in rows_noise if r.get('sigma') == s]
        per_eq_s = per_equation_medians(sub)
        medians = {m: statistics.median([d[m] for d in per_eq_s.values() if m in d]) for m in MODELS}
        print(f'  σ={s:<4}', end='')
        for m in MODELS:
            print(f' {medians[m]:>7.3f}', end='')
        print()
    per_eq_0 = per_equation_medians([r for r in rows_noise if r.get('sigma') == 0.0])
    per_eq_10 = per_equation_medians([r for r in rows_noise if r.get('sigma') == 1.0])
    print('\n  Degradation σ=1.0 / σ=0.0:')
    for m in MODELS:
        v0 = statistics.median([d[m] for d in per_eq_0.values() if m in d])
        v1 = statistics.median([d[m] for d in per_eq_10.values() if m in d])
        print(f'  {m:20s} {v1 / v0:.2f}x   (0: {v0:.4f} → 1.0: {v1:.4f})')
    rows_sreg = load_jsonl('results_noise_spline_reg_eq*.jsonl')
    if rows_sreg:
        print('\n  Regularized spline (18 eq core) medians:')
        for s in sorted(set((r['sigma'] for r in rows_sreg if 'sigma' in r))):
            print(f'  σ={s:<4}', end='')
            for lam_model in sorted(set((r['model'] for r in rows_sreg))):
                sub = [r for r in rows_sreg if r.get('sigma') == s and r['model'] == lam_model]
                if not sub:
                    continue
                per = per_equation_medians(sub, models=[lam_model])
                m_med = statistics.median([d[lam_model] for d in per.values() if lam_model in d])
                print(f'  {lam_model}:{m_med:.3f}', end='')
            print()
    print('\n' + '=' * 80)
    print('SCARCITY (18 eq core, σ=0.3, n×model medians)')
    print('=' * 80)
    rows_scar = load_jsonl('results_scarcity_eq*.jsonl', 'results_scarcity_cheby_eq*.jsonl')
    ns = sorted(set((r['n_train'] for r in rows_scar if 'n_train' in r)))
    for n in ns:
        sub = [r for r in rows_scar if r.get('n_train') == n]
        per_eq_n = per_equation_medians(sub)
        medians = {m: statistics.median([d[m] for d in per_eq_n.values() if m in d]) for m in MODELS}
        wins = win_counts(per_eq_n)
        print(f'  n={n:<5}', end='')
        for m in MODELS:
            print(f' {medians[m]:>7.3f}', end='')
        print(f'    wins:', end='')
        for m in MODELS:
            print(f' {wins[m]:>3d}', end='')
        print()
    race_wins = {m: 0 for m in MODELS}
    for n in ns:
        sub = [r for r in rows_scar if r.get('n_train') == n]
        per = per_equation_medians(sub)
        w = win_counts(per)
        for m in MODELS:
            race_wins[m] += w[m]
    print(f'\n  90-race totals (18 eq × 5 n):', race_wins)
    print('\n  3-way scarcity race (Lp vs Banach vs best reg spline, 90 races):')
    rows_sreg_scar = load_jsonl('results_scarcity_spline_reg_eq*.jsonl')
    ns = sorted(set((r['n_train'] for r in rows_scar if 'n_train' in r)))
    lp_wins, banach_wins, spline_wins = (0, 0, 0)
    for n in ns:
        sub = [r for r in rows_scar if r.get('n_train') == n]
        per_eq_n = per_equation_medians(sub, models=['lp', 'banach'])
        sub_sreg = [r for r in rows_sreg_scar if r.get('n_train') == n]
        per_eq_sreg = per_equation_medians(sub_sreg, models=['spline_reg_g3_l0', 'spline_reg_g3_l0.001', 'spline_reg_g3_l0.01', 'spline_reg_g3_l0.1'])
        for eq in per_eq_n:
            lp_v = per_eq_n[eq].get('lp')
            banach_v = per_eq_n[eq].get('banach')
            sreg_best = min(per_eq_sreg.get(eq, {}).values()) if per_eq_sreg.get(eq) else None
            cands = {}
            if lp_v is not None:
                cands['lp'] = lp_v
            if banach_v is not None:
                cands['banach'] = banach_v
            if sreg_best is not None:
                cands['sreg'] = sreg_best
            if not cands:
                continue
            winner = min(cands, key=lambda k: cands[k])
            if winner == 'lp':
                lp_wins += 1
            elif winner == 'banach':
                banach_wins += 1
            else:
                spline_wins += 1
    print(f'    Lp: {lp_wins}, Banach: {banach_wins}, best-reg-spline: {spline_wins}')
    print(f'    Geometry-constrained (Lp+Banach): {lp_wins + banach_wins}/90')
    print('\n  3-way noise race (Lp vs Banach vs best reg spline, 90 races):')
    rows_sreg_noise = load_jsonl('results_noise_spline_reg_eq*.jsonl')
    sigmas = sorted(set((r['sigma'] for r in rows_noise if 'sigma' in r)))
    lp_w, banach_w, spline_w = (0, 0, 0)
    for s in sigmas:
        sub = [r for r in rows_noise if r.get('sigma') == s]
        per_eq_s = per_equation_medians(sub, models=['lp', 'banach'])
        sub_sreg = [r for r in rows_sreg_noise if r.get('sigma') == s]
        per_eq_sreg = per_equation_medians(sub_sreg, models=['spline_reg_g3_l0', 'spline_reg_g3_l0.001', 'spline_reg_g3_l0.01', 'spline_reg_g3_l0.1'])
        for eq in per_eq_s:
            cands = {}
            if 'lp' in per_eq_s[eq]:
                cands['lp'] = per_eq_s[eq]['lp']
            if 'banach' in per_eq_s[eq]:
                cands['banach'] = per_eq_s[eq]['banach']
            if per_eq_sreg.get(eq):
                cands['sreg'] = min(per_eq_sreg[eq].values())
            if not cands:
                continue
            winner = min(cands, key=lambda k: cands[k])
            if winner == 'lp':
                lp_w += 1
            elif winner == 'banach':
                banach_w += 1
            else:
                spline_w += 1
    print(f'    Lp: {lp_w}, Banach: {banach_w}, best-reg-spline: {spline_w}')
    print(f'    Geometry-constrained (Lp+Banach): {lp_w + banach_w}/90')
    print(f'\n  COMBINED (noise+scarcity, 180 races):')
    print(f'    Geometry-constrained (Lp+Banach): {lp_w + banach_w + lp_wins + banach_wins}/180')
    print(f'    best-reg-spline: {spline_w + spline_wins}/180')
    print('\n' + '=' * 80)
    print('DISSECTION (3-way race: Tanh, Lp, Banach)')
    print('=' * 80)
    TANH_LP_BANACH = ['tanh', 'lp', 'banach']
    rows_base = load_jsonl('results_baseline50_eq*.jsonl', 'results_baseline50_banach_eq*.jsonl')
    per_eq_b = per_equation_medians(rows_base, models=TANH_LP_BANACH)
    w = win_counts(per_eq_b, models=TANH_LP_BANACH)
    print(f"  Clean (50 eq): Tanh {w['tanh']}, Lp {w['lp']}, Banach {w['banach']}")
    sub = [r for r in rows_noise if r.get('sigma') == 0.3]
    per_eq_n = per_equation_medians(sub, models=TANH_LP_BANACH)
    w = win_counts(per_eq_n, models=TANH_LP_BANACH)
    print(f"  Noisy σ=0.3 (18 eq): Tanh {w['tanh']}, Lp {w['lp']}, Banach {w['banach']}")
    sub = [r for r in rows_scar if r.get('n_train') == 50]
    per_eq_s = per_equation_medians(sub, models=TANH_LP_BANACH)
    w = win_counts(per_eq_s, models=TANH_LP_BANACH)
    print(f"  Scarce n=50 (18 eq): Tanh {w['tanh']}, Lp {w['lp']}, Banach {w['banach']}")
    print('\n' + '=' * 80)
    print('ABLATION_P (18 eq core, learned vs fixed p)')
    print('=' * 80)
    rows_abl = load_jsonl('results_ablation_p_eq*.jsonl')
    abl_models = ['lp_fixed_1.5', 'lp_fixed_2.0', 'lp_fixed_2.5', 'lp_fixed_3.0', 'lp_fixed_4.0', 'lp_learned']
    per_eq = per_equation_medians(rows_abl, models=abl_models)
    w = win_counts(per_eq, models=abl_models)
    print(f'  Win counts (18 eq):', w)
    print('\n' + '=' * 80)
    print('LEARNED P BY TYPE (Banach-KAN, 50 eq)')
    print('=' * 80)
    banach_rows = [r for r in rows_base if r['model'] == 'banach' and 'p_values' in r]
    by_type = collections.defaultdict(list)
    for r in banach_rows:
        t = r.get('type') or r.get('eq_type', '?')
        for p in r.get('p_values', []):
            by_type[t].append(p)
    for t, ps in sorted(by_type.items()):
        if ps:
            print(f'  {t:15s} mean={statistics.mean(ps):.3f} std={(statistics.stdev(ps) if len(ps) > 1 else 0):.3f} range=[{min(ps):.2f}, {max(ps):.2f}]  n={len(ps)}')
    print('\n' + '=' * 80)
    print('TIME TABLE (median wall-clock per equation, 50 eq)')
    print('=' * 80)
    time_by_model = collections.defaultdict(list)
    rows_base_all = load_jsonl('results_baseline50_eq*.jsonl', 'results_baseline50_banach_eq*.jsonl', 'results_baseline50_cheby_eq*.jsonl')
    for r in rows_base_all:
        t = r.get('train_time') or r.get('train_time_s')
        if r.get('model') in MODELS and t is not None:
            time_by_model[r['model']].append(t)
    for m in MODELS:
        if time_by_model[m]:
            print(f'  {m:20s} median={statistics.median(time_by_model[m]):.1f}s  n={len(time_by_model[m])}')
if __name__ == '__main__':
    main()
