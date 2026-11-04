from .neurips_camera_ready import geometry_extrapolation_table, summary_cells
from .neurips_public_formatting import LABEL, ROSTER, HEADLINE, CELL_TEX
TIED_LABELS = ['Independent', 'Tied, matched gain', 'Tied', 'Tied, $p_0=2$']

def checked(summary, study):
    if not summary['inventory']['complete_and_valid']:
        raise ValueError('Incomplete follow-up population')
    expected = ROSTER if study == 'extrap15' else ['independent_c1', 'tied_c1_matched', 'tied_c1', 'tied_c1_p2_init']
    if summary['models'] != expected or not summary['protocol']['performance_evidence']:
        raise ValueError('Unexpected follow-up roster or non-performance run')
    return summary

def tied_table(summary, full=False):
    keys = list(summary['cells']) if full else ['n=500/sigma=0', 'n=500/sigma=0.3', 'n=500/sigma=1']
    headers = ['$\\sigma=' + k.split('sigma=')[1] + '$' if k.startswith('n=500/') else '$n=' + k.split('/')[0][2:] + '$' for k in keys]
    lines = ['\\begin{tabular}{l' + 'r' * len(keys) + '}', '\\toprule', 'Affine variant & ' + ' & '.join(headers) + ' \\\\', '\\midrule']
    for i, label in enumerate(TIED_LABELS):
        cells = []
        for key in keys:
            values = summary['cells'][key]['scorecard']['median']
            text = f'{values[i]:.3f}'
            cells.append('\\textbf{' + text + '}' if values[i] == min(values) else text)
        lines.append(label + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule\\end{tabular}']) + '\n'

def extrapolation_table(summary):
    score = summary['cells']['n=500/sigma=0']['scorecard']
    lines = ['\\begin{tabular}{lrrr}', '\\toprule', 'Model & NRMSE & Wins & Mean rank \\\\', '\\midrule']
    for i, model in enumerate(summary['models']):
        lines.append(LABEL[model] + ' & ' + ' & '.join(summary_cells(score, i)) + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule\\end{tabular}']) + '\n'

def paired_table(matched, extrap):
    look = {q['comparison']: q for q in matched['paired']}
    ext = {q['comparison']: q for q in extrap['cells']['n=500/sigma=0']['paired']}
    iv = lambda x: f'[{x[0]:.2f}, {x[1]:.2f}]'
    lines = ['\\begin{tabular}{llccc}', '\\toprule', 'Setting & Comparison & Ratio & 95\\% interval & Family-wise \\\\', '\\midrule']
    rows = [(CELL_TEX[c], a, b, look[f'{c}:{a}/{b}']) for c, a, b in HEADLINE]
    for a, b in [('banach_c1', 'tanh'), ('banach_c1', 'spline'), ('banach_c1', 'pow')]:
        rows.append(('Extrapolation', a, b, ext[f'{a}/{b}']))
    for setting, a, b, q in rows:
        lines.append(f"{setting} & {LABEL[a]} / {LABEL[b]} & {q['ratio']:.2f} & {iv(q['ci95'])} & {iv(q['family_ci'])}" + ' \\\\')
    return '\n'.join(lines + ['\\bottomrule\\end{tabular}']) + '\n'

def fragments(geometry, matched, extrap, tied):
    checked(extrap, 'extrap15')
    checked(tied, 'tied_affine')
    score = extrap['cells']['n=500/sigma=0']['scorecard']
    medians = dict(zip(extrap['models'], score['median']))
    main = geometry_extrapolation_table(geometry, medians)
    main += '\\par\\medskip' + '\n' + tied_table(tied) + '\\par\\smallskip (c)' + '\n'
    return {'geometry_extrapolation': main, 'extrapolation': extrapolation_table(extrap), 'tied_full': tied_table(tied, full=True), 'matched_paired': paired_table(matched, extrap)}
