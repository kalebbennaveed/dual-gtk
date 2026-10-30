"""The replay must preserve arrival times and show learning at its event time."""
import json
from pathlib import Path
import re

import numpy as np

from scripts.export_web_demo import export


def test_web_export_preserves_common_clock_and_posterior_events(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'results' / 'focused_demo'
    output = tmp_path / 'index.html'
    export(source, output)
    html = output.read_text()
    match = re.search(r'<script id="simulation-data" type="application/json">(.*?)</script>', html)
    data = json.loads(match.group(1))
    runs = {run['method']: run for run in data['runs']}
    expected_times = {method: json.loads((source / f'informative_demo_{method}.json').read_text())['samples'][-1]['time']
                      for method in runs}
    assert {method: run['samples'][-1]['time'] for method, run in runs.items()} == expected_times
    assert len(set(round(t, 5) for t in expected_times.values())) > 1
    for method, run in runs.items():
        raw = json.loads((source / f'informative_demo_{method}.json').read_text())
        assert len(run['samples']) == len(raw['samples'])
        assert [s['time'] for s in run['samples']] == [s['time'] for s in raw['samples']]
        assert np.array_equal([s['state'] for s in run['samples']],
                              [s['state'] for s in raw['samples']])
        for epoch in raw['epochs']:
            endpoint = epoch['time'] + epoch['duration']
            samples = [s for s in run['samples'] if abs(s['time'] - endpoint) < 1e-8]
            assert samples
            assert np.allclose(samples[-1]['interval'], epoch['interval_after'])
    first = runs['proposed']['epochs'][0]
    sample = next(s for s in runs['proposed']['samples'] if abs(s['time'] - first['duration']) < 1e-8)
    assert np.allclose(sample['interval'], first['interval_after'])
    assert '__APP_SCRIPT__' not in html and '__IMPORT_MAP__' not in html
    assert 'data:text/javascript;base64,' in html


def test_focused_scenario_has_no_altitude_constraint():
    from drone_dual_gatekeeper.config import load_config
    from drone_dual_gatekeeper.planning.optimizer import TrajectoryOptimizer
    config, scenario = load_config('informative_demo')
    assert config.nominal_altitude is None
    # Only initial and terminal position/velocity equalities are present.
    assert TrajectoryOptimizer(config, scenario)._build(None, False)[2] == 12


def test_swept_audit_detects_collision_between_safe_endpoints():
    from scripts.audit_drone_demo import sweep_clearance
    from drone_dual_gatekeeper.config import Scenario
    obstacle = {'kind': 'cylinder', 'center': [0., 0., 1.], 'radius': 1., 'height': 2.}
    scenario = Scenario('audit', (0.,)*6, (0.,)*3, (-3.,)*3, (3.,)*3, (obstacle,), 'test')
    points = np.array([[-2., 0., 1.], [2., 0., 1.]])
    assert np.isclose(sweep_clearance(points, scenario, .1), -1.1)


def test_spatial_course_cannot_be_flown_at_constant_altitude():
    from drone_dual_gatekeeper.config import load_config
    from scripts.audit_drone_demo import sweep_clearance
    config,scenario=load_config('spatial_learning_course')
    assert config.nominal_altitude is None
    assert scenario.start[2]!=scenario.goal[2]
    assert sweep_clearance(np.array([scenario.start[:3],scenario.goal]),scenario,config.drone_radius)<0
    # The low and overhead walls overlap in height and span the allowed width.
    # Every fixed-height route along the corridor intersects at least one wall.
    for z in np.linspace(scenario.lower[2],scenario.upper[2],25):
        assert sweep_clearance(np.array([[0.,0.,z],[22.,0.,z]]),scenario,config.drone_radius)<0


def test_export_supports_second_scenario_without_counterfactual(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'results' / 'focused_demo'
    # Reuse the validated records to test the exporter contract, without solving.
    for method in ('proposed', 'passive', 'weighted', 'robust'):
        record = json.loads((source / f'informative_demo_{method}.json').read_text())
        record['scenario']['name'] = 'spatial_learning_course'
        record.pop('learning_comparison', None)
        (tmp_path / f'spatial_learning_course_{method}.json').write_text(json.dumps(record))
    output = tmp_path / 'complex.html'
    export(tmp_path, output, 'spatial_learning_course')
    match = re.search(r'<script id="simulation-data" type="application/json">(.*?)</script>', output.read_text())
    data = json.loads(match.group(1))
    assert data['counterfactual'] is None
    assert data['scenario']['name'] == 'spatial_learning_course'
    assert len(data['runs']) == 4


def test_slalom_geometry_requires_alternating_lateral_passages():
    from drone_dual_gatekeeper.config import load_config
    from scripts.audit_drone_demo import sweep_clearance
    config, scenario = load_config('slalom_course')
    assert config.nominal_altitude is None
    assert scenario.start[2] != scenario.goal[2]
    assert sweep_clearance(np.array([scenario.start[:3], scenario.goal]), scenario, config.drone_radius) < 0
    for index, obstacle in enumerate(scenario.obstacles):
        assert obstacle['lower'][2] <= scenario.lower[2]
        assert obstacle['upper'][2] >= scenario.upper[2]
        if index % 2 == 0:
            assert obstacle['lower'][1] == scenario.lower[1]
            assert obstacle['upper'][1] > 0
        else:
            assert obstacle['upper'][1] == scenario.upper[1]
            assert obstacle['lower'][1] < 0
