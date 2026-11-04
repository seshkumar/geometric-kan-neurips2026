import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import torch
from experiments import feynman_runner as core
from src import matched_models as rb
from src.neurips_normalization import standard_deviation
from src.neurips_rerun import declare
from src import neurips_sample_sd as study
from src.edge_models import BanachC1KANLayer
from src.kan_layer import KANetwork
ROOT = Path(__file__).resolve().parents[1]

def test_sample_sd_is_explicit_for_numpy_and_torch():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert standard_deviation(a) == 1.0
    assert standard_deviation(torch.from_numpy(a)) == 1.0
    assert standard_deviation(a, ddof=0) != 1.0
    with pytest.raises(ValueError):
        standard_deviation(a, ddof=2)

def test_sample_inputs_match_core_exactly_across_entire_equation_roster():
    for index, e in enumerate(core.EQUATION_IDS):
        for n, sigma, seed in [(500, 0.3, 1729), (50, 0.3, 1733), (1000, 0.0, 1730)]:
            x, y = core.generate_data(e, n + 1000, seed)
            train, test, _, scale = core.normalize_targets(y[:n], y[n:])
            if sigma:
                torch.manual_seed(seed + int(sigma * 1000))
                train = train + torch.randn(n) * sigma
            a, b, c, d, s = rb.generate_data(rb.FEYNMAN_50[index], n, sigma, seed, sd_ddof=1)
            for actual, expected in [(a, x[:n]), (b, train), (c, x[n:]), (d, test)]:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert s == float(scale)

def test_shared_references_are_not_independently_refitted():
    for i in range(50):
        cases = study.cases('symbolic', i)
        assert len(cases) == len(set(cases))
        assert ('banach_c1', 500, 0.3) in cases
        assert not any((m in ('independent_c1', 'banach_learned') for m, n, s in cases))
        if i < 18:
            assert all(((m, 500, 0.3) in cases for m in study.followup.TIED_MODELS[1:]))
    assert study.plan('symbolic')['target_sd_ddof'] == 1
    assert study.plan('extrapolation')['seeds'] == list(range(5))

def test_core_and_matched_initialization_split_and_short_training_match():
    torch.set_num_threads(1)
    for index in (0, 8, 11, 15, 17, 40):
        eq = rb.FEYNMAN_50[index]
        x, y, xt, yt, _ = rb.generate_data(eq, 50, 0.3, 1729, sd_ddof=1)
        torch.manual_seed(1729)
        a = KANetwork(core.get_architecture(x.shape[1]), BanachC1KANLayer)
        torch.manual_seed(1729)
        b = rb.MODEL_BUILDERS['banach_c1'](x.shape[1])
        for u, v in zip(a.state_dict().values(), b.state_dict().values()):
            torch.testing.assert_close(u, v, rtol=0, atol=0)
        core.train_adam_model(core.SqueezeWrapper(a), x, y, xt, yt, steps=8)
        rb.train_one(b, x, y, steps=8)
        for u, v in zip(a.state_dict().values(), b.state_dict().values()):
            torch.testing.assert_close(u, v, rtol=0, atol=0)

def test_pmlb_uses_sample_sd_for_features_and_targets(monkeypatch):
    import pandas as pd
    import pmlb
    from src import pmlb_data as pb
    data = pd.DataFrame({'a': np.arange(100.0), 'b': np.arange(100.0) ** 2, 'target': np.sin(np.arange(100.0))})
    monkeypatch.setattr(pmlb, 'fetch_data', lambda *a, **k: data.copy())
    x, y, xt, yt, mu, scale, original = pb.load_pmlb_dataset('fixture', 1729, sd_ddof=1)
    torch.testing.assert_close(x.std(dim=0), torch.ones(2), rtol=1e-06, atol=1e-06)
    torch.testing.assert_close(y.std(), torch.tensor(1.0), rtol=1e-06, atol=1e-06)

def test_replacement_is_explicit_and_invalidates_stale_summaries(tmp_path):
    old = dict(protocol={'ddof': 0}, source_fingerprint='old')
    new = dict(protocol={'ddof': 1}, source_fingerprint='new')
    declare(tmp_path, old)
    (tmp_path / 'summary.json').write_text('old summary')
    (tmp_path / 'fits.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='preserved'):
        declare(tmp_path, new)
    assert (tmp_path / 'summary.json').read_text() == 'old summary'
    declare(tmp_path, new, True)
    assert not (tmp_path / 'summary.json').exists()
    replacement = json.loads((tmp_path / 'replacement.json').read_text())
    assert replacement['previous'] == old and 'fits.jsonl' in replacement['overwritten_files']

@pytest.mark.parametrize('profile', ['symbolic', 'extrapolation'])
def test_sample_pbs_dry_run(profile):
    for index in (0, 49):
        env = os.environ | {'PBS_O_WORKDIR': str(ROOT), 'PBS_ARRAY_INDEX': str(index), 'SAMPLE_SD_PROFILE': profile}
        command = subprocess.check_output(['bash', 'pbs/run_neurips_sample_sd.pbs', '--dry-run'], cwd=ROOT, env=env, text=True)
        assert f'--case-index {index}' in command and f'--profile {profile}' in command
        assert '--replace-existing' in command

def test_pmlb_sample_pbs_replaces_canonical_outputs():
    env = os.environ | {'PBS_O_WORKDIR': str(ROOT), 'PBS_ARRAY_INDEX': '11'}
    command = subprocess.check_output(['bash', 'pbs/run_neurips_sample_sd_pmlb.pbs', '--dry-run'], cwd=ROOT, env=env, text=True)
    assert 'pmlb15_pbs_v1' in command and '--replace-existing' in command
