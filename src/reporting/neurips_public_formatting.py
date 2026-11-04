C1 = ['banach_c1', 'lp_c1']
ROSTER = C1 + ['tanh', 'pow', 'adaslope', 'mixture', 'rational', 'spline', 'spline_lip', 'cheby', 'rbf', 'fourier', 'wavelet', 'jacobi', 'mlp']
LABEL = {'banach_c1': 'Banach', 'lp_c1': '$\\ell^p$', 'tanh': 'Tanh', 'pow': 'Learned power', 'adaslope': 'Adaptive-slope Tanh', 'mixture': 'Activation mixture', 'rational': 'Rational', 'spline': 'Spline (Adam)', 'spline_lip': 'Spline (Adam, Lipschitz-reg.)', 'cheby': 'Chebyshev', 'rbf': 'RBF', 'fourier': 'Fourier', 'wavelet': 'Wavelet', 'jacobi': 'Jacobi', 'mlp': 'MLP', 'banach': 'Banach (additive-$\\epsilon$)', 'lp': '$\\ell^p$ (additive-$\\epsilon$)'}
CELLS = [('clean', 'baseline', lambda r: True)] + [(f'sigma={s}', 'noisy', lambda r, s=s: r['sigma'] == s) for s in (0.1, 0.3, 0.5, 1.0)] + [(f'n={n}', 'small_sample', lambda r, n=n: r['n_train'] == n) for n in (50, 100, 200, 1000)]
STEM = {'baseline': 'baseline_eq50', 'noisy': 'noisy', 'small_sample': 'small_sample'}
PAIR_TARGETS = ['tanh', 'adaslope', 'pow', 'spline', 'spline_lip']
EXTRAP_NEURIPS = ['banach_c1', 'lp_c1', 'tanh', 'spline', 'cheby', 'rbf', 'fourier']
HEADLINE = [('clean', 'banach_c1', 'tanh'), ('clean', 'banach_c1', 'adaslope'), ('clean', 'banach_c1', 'pow'), ('clean', 'banach_c1', 'spline'), ('clean', 'banach_c1', 'spline_lip'), ('sigma=1.0', 'lp_c1', 'pow'), ('sigma=1.0', 'lp_c1', 'tanh'), ('sigma=1.0', 'lp_c1', 'spline_lip'), ('n=50', 'banach_c1', 'spline'), ('n=50', 'banach_c1', 'tanh'), ('n=50', 'banach_c1', 'adaslope'), ('n=1000', 'banach_c1', 'spline')]
EXTRAP_HEADLINE = [('banach_c1', 'tanh'), ('banach_c1', 'spline'), ('banach_c1', 'banach'), ('lp_c1', 'lp')]
CELL_TEX = {'clean': 'clean', 'sigma=1.0': '$\\sigma=1$', 'n=50': '$n=50$', 'n=1000': '$n=1000$'}
CURVE_MODELS = ['banach_c1', 'lp_c1', 'tanh', 'adaslope', 'pow', 'spline', 'spline_lip', 'rbf']
CURVE_STYLE = {'banach_c1': ('#1f4e9c', 'o', '-'), 'lp_c1': ('#c0392b', 's', '-'), 'tanh': ('#7f7f7f', '^', '--'), 'adaslope': ('#9467bd', 'v', '--'), 'pow': ('#e67e22', 'D', '--'), 'spline': ('#2ca02c', 'P', ':'), 'spline_lip': ('#17becf', 'X', ':'), 'rbf': ('#8c564b', '*', ':')}

def paired_table(summary, out):
    look = {p['comparison']: p for p in summary['paired']}
    ext = {p['comparison']: p for p in summary['extrapolation']['paired']}
    iv = lambda v: f'[{v[0]:.2f}, {v[1]:.2f}]'
    tex = ['\\begin{tabular}{llccc}', '\\toprule', 'setting & comparison & ratio & 95\\% interval & family-wise \\\\', '\\midrule']
    for cell, a, b in HEADLINE:
        q = look[f'{cell}:{a}/{b}']
        tex.append(f"{CELL_TEX[cell]} & {LABEL[a]} / {LABEL[b]} & {q['ratio']:.2f} & {iv(q['ci95'])} & {iv(q['family_ci'])} \\\\")
    tex.append('\\midrule')
    for a, b in EXTRAP_HEADLINE:
        q = ext[f'{a}/{b}']
        la = LABEL[a] + (' (C$^1$)' if a in C1 and b in ('banach', 'lp') else '')
        tex.append(f"extrapolation & {la} / {LABEL[b]} & {q['ratio']:.2f} & {iv(q['ci95'])} & {iv(q['family_ci'])} \\\\")
    tex += ['\\bottomrule', '\\end{tabular}']
    (out / 'tab_matched_paired.tex').write_text('\n'.join(tex) + '\n')

def curves(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.hashsalt': 'c1', 'path.simplify': False})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), constrained_layout=True)
    panels = [(axes[0], [0.0, 0.1, 0.3, 0.5, 1.0], ['clean'] + [f'sigma={s}' for s in (0.1, 0.3, 0.5, 1.0)], 'label noise $\\sigma$ ($n=500$)'), (axes[1], [50, 100, 200, 500, 1000], [f'n={n}' for n in (50, 100, 200)] + ['clean', 'n=1000'], 'clean training size $n$')]
    for ax, xs, keys, xlabel in panels:
        for m in CURVE_MODELS:
            colour, marker, line = CURVE_STYLE[m]
            ax.plot(xs, [summary['cells'][k][m]['median'] for k in keys], color=colour, marker=marker, linestyle=line, linewidth=1.2, markersize=3.5, label=LABEL[m])
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[1].set_xscale('log')
    axes[1].set_xticks([50, 100, 200, 500, 1000])
    axes[1].set_xticklabels(['50', '100', '200', '500', '1000'])
    axes[0].set_ylabel('median NRMSE')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)
    fig.savefig(out / 'matched_curves.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
