"""Publishing must reject mismatched controller sources and remove retired artifacts."""
import json
from pathlib import Path
import pytest
from dual_gtk_drs_racing import results_web


def fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(results_web,'implementation_signature',lambda:'current')
    results=tmp_path/'inputs';report=results/'mppi-report';report.mkdir(parents=True)
    (report/'manifest.json').write_text(json.dumps({'implementation':'current'}))
    (report/'validation_safety.csv').write_text('scenario,method,trials,completed,violations\ntraction,proposed,8,8,0\n')
    destination=tmp_path/'site';old=destination/'results/launch-stress';old.mkdir(parents=True)
    (old/'obsolete.csv').write_text('old planner results')
    return results,destination


def test_old_reference_source_cannot_replace_export(tmp_path,monkeypatch):
    results,dest=fixtures(tmp_path,monkeypatch)
    (results/'mppi-report/manifest.json').write_text(json.dumps({'implementation':'old'}))
    with pytest.raises(ValueError,match='Reference results'):
        results_web.build_results_portal(results,dest)
    assert (dest/'results/launch-stress/obsolete.csv').exists()


def test_old_stress_source_cannot_replace_export(tmp_path,monkeypatch):
    results,dest=fixtures(tmp_path,monkeypatch);stress=results/'mppi-stress';stress.mkdir()
    (stress/'manifest.json').write_text(json.dumps({'implementation':'old'}))
    with pytest.raises(ValueError,match='Stress records'):
        results_web.build_results_portal(results,dest)
    assert (dest/'results/launch-stress/obsolete.csv').exists()


def test_current_export_removes_all_old_results(tmp_path,monkeypatch):
    results,dest=fixtures(tmp_path,monkeypatch)
    page=results_web.build_results_portal(results,dest)
    assert not (dest/'results/launch-stress').exists()
    assert 'launch-stress/' not in page.read_text()
    assert 'MPPI racing results' in page.read_text()


def test_old_risk_probe_cannot_replace_export(tmp_path,monkeypatch):
    results,dest=fixtures(tmp_path,monkeypatch);risk=results/'mppi-risk-report';risk.mkdir()
    (risk/'summary.json').write_text(json.dumps({'implementation':'old'}))
    with pytest.raises(ValueError,match='Risk probes'):
        results_web.build_results_portal(results,dest)
    assert (dest/'results/launch-stress/obsolete.csv').exists()
