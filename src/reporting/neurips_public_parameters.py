ROSTER = ['banach_c1', 'lp_c1', 'tanh', 'pow', 'adaslope', 'mixture', 'rational', 'spline', 'spline_lip', 'cheby', 'rbf', 'fourier', 'wavelet', 'jacobi', 'mlp']

def parameter_counts(rows):
    result = {}
    for r in rows:
        if r['model'] not in ROSTER:
            continue
        key = (r['model'], r['n_vars'])
        count = int(r['n_params'])
        if count <= 0 or (key in result and result[key] != count):
            raise ValueError('Inconsistent recorded parameter count')
        result[key] = count
    dimensions = sorted({d for _, d in result})
    if any(((m, d) not in result for m in ROSTER for d in dimensions)):
        raise ValueError('Incomplete parameter roster')
    return (result, dimensions)
