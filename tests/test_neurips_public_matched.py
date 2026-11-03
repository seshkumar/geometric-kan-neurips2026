import numpy as np
import pytest

from src.reporting.neurips_camera_ready import paired_ratios
from src.reporting.neurips_public_matched import (
    EXTRAP_MODELS, EXTRAP_SLOTS, extrapolation_pairs, seed_median_matrix,
)


def test_reserved_bootstrap_slots_preserve_full_family_and_rng_positions():
    rng = np.random.default_rng(1729)
    matrix = np.exp(rng.normal(size=(7, len(EXTRAP_MODELS))))
    original = []
    for i, pair in enumerate(EXTRAP_SLOTS):
        if pair is None:
            original.append((f'excluded-{i}', np.arange(1, 8), np.ones(7)))
        else:
            a, b = pair
            original.append((f'{a}/{b}', matrix[:, EXTRAP_MODELS.index(a)],
                             matrix[:, EXTRAP_MODELS.index(b)]))
    expected = [p for p in paired_ratios(original, draws=150)
                if not p['comparison'].startswith('excluded-')]
    assert extrapolation_pairs(matrix, draws=150) == expected


def test_matched_cells_require_declared_seeds_without_duplicates():
    rows = [dict(eq_id='e', model='m', seed=s, nrmse=float(s)) for s in (1, 2, 3)]
    np.testing.assert_array_equal(seed_median_matrix(rows, ['m'], ['e'], lambda r: True,
                                                    seeds=[1, 2, 3]), [[2]])
    with pytest.raises(ValueError, match='Incomplete'):
        seed_median_matrix(rows[:-1], ['m'], ['e'], lambda r: True, seeds=[1, 2, 3])
    with pytest.raises(ValueError, match='Duplicate'):
        seed_median_matrix(rows+[rows[0]], ['m'], ['e'], lambda r: True, seeds=[1, 2, 3])
