"""Portable orchestration must reuse fitting drivers and protect paper inputs."""
from types import SimpleNamespace
import sys

import pytest

from src.reporting import neurips_public_driver as driver


def args(tmp_path,**kwargs):
    data=dict(study='symbolic',workers=2,out=tmp_path/'fresh',all_cases=False,
              case_index=None,aggregate=False,dry_run=False)
    data.update(kwargs)
    return SimpleNamespace(**data)


@pytest.fixture
def calls(monkeypatch):
    recorded=[]
    monkeypatch.setitem(sys.modules,'src.neurips_sample_sd',SimpleNamespace(plan=lambda profile:dict(equations=['a','b'])))
    monkeypatch.setitem(sys.modules,'src.neurips_pmlb',SimpleNamespace(protocol=lambda:dict(datasets=['a','b','c'])))
    monkeypatch.setattr(driver.subprocess,'run',lambda command,**kw:recorded.append((command,kw)))
    return recorded


def test_full_local_study_reuses_existing_driver_without_replacement(tmp_path,calls):
    result=driver.run_local(args(tmp_path,all_cases=True,aggregate=True))
    assert result['cases']==[0,1] and len(calls)==3
    assert all('neurips_sample_sd.py' in c[0][1] for c in calls)
    assert [c[0][2] for c in calls]==['run','run','summarize']
    assert all('--replace-existing' not in c[0] for c in calls)
    assert calls[0][1]['env']['OMP_NUM_THREADS']=='1'
    assert calls[0][0][calls[0][0].index('--workers')+1]=='2'


def test_aggregate_alone_does_not_fit(tmp_path,calls):
    driver.run_local(args(tmp_path,aggregate=True))
    assert len(calls)==1 and calls[0][0][2]=='summarize'


def test_dry_run_does_not_launch_training(tmp_path,calls):
    driver.run_local(args(tmp_path,all_cases=True,dry_run=True))
    assert not calls


def test_pmlb_uses_declared_dataset_count(tmp_path,calls):
    result=driver.run_local(args(tmp_path,study='pmlb',all_cases=True))
    assert result['cases']==[0,1,2]
    assert all('neurips_pmlb15.py' in c[0][1] and '--dataset-idx' in c[0] for c in calls)


@pytest.mark.parametrize('kwargs',[dict(workers=0),dict(case_index=-1),dict(case_index=2)])
def test_invalid_settings_fail_before_launch(tmp_path,calls,kwargs):
    with pytest.raises(ValueError):driver.run_local(args(tmp_path,**kwargs))
    assert not calls


def test_bundled_results_cannot_be_training_destination(tmp_path,calls):
    with pytest.raises(ValueError,match='bundled evidence'):
        driver.run_local(args(tmp_path,out=driver.ROOT/'experiments/results/neurips_sample_sd/symbolic'))
    assert not calls
