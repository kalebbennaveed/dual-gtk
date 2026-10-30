"""Independently audit saved runs and refine execution of their first segment.

No trajectory optimization or access to truth occurs in reconstruct_plan.
Truth is used only by this validation script and the true simulator.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from drone_dual_gatekeeper.config import ControlConfig, Scenario
from drone_dual_gatekeeper.control.tube_propagation import build_tube
from drone_dual_gatekeeper.environment.collision import certify_plan
from drone_dual_gatekeeper.planning.curve import Curve
from drone_dual_gatekeeper.planning.dual_gatekeeper import reconnection
from drone_dual_gatekeeper.simulation.logging import save_result
from drone_dual_gatekeeper.simulation.simulator import SimulationTruth, TrueDroneSimulator
from drone_dual_gatekeeper.types import Interval, RobustPlan


def reconstruct_plan(record, config):
    interval = Interval(*record['interval'])
    curve = Curve(record['coefficients'], record['horizon'], config.subdivisions)
    tube = build_tube(curve, interval, config)
    np.testing.assert_allclose(tube.boundary_layer, record['phi'], atol=1e-10)
    np.testing.assert_allclose(tube.position_radius, record['rho'], atol=1e-10)
    np.testing.assert_allclose(tube.drag_forcing, record['drag_forcing'], atol=1e-10)
    return RobustPlan(np.empty((0, 6)), np.empty((0, 3)), np.array([]),
                      tube.boundary_layer, tube.position_radius, record['cost'],
                      curve.horizon, curve, tube, interval.midpoint,
                      record['solver_status'], record['residual'],
                      record['planning_time'], 0., record['certificate'])


def audit(result):
    config = ControlConfig(**result['config'])
    scenario = Scenario(**result['scenario'])
    truth = result['truth']['true_drag_coefficient']
    assert result['metrics']['success'], 'Incomplete mission'
    assert result['metrics']['violations'] == 0
    if all('planned_duration' in e for e in result['epochs']):
        assert abs(result['metrics']['completion_time']-result['samples'][-1]['time'])<1e-8
        reached=lambda row: (np.linalg.norm(np.array(row['state'][:3])-scenario.goal)<=config.goal_tolerance
                             and np.linalg.norm(row['state'][3:])<=config.goal_velocity_tolerance)
        assert reached(result['samples'][-1])
        assert not any(reached(row) for row in result['samples'][:-1]),'Arrival was detected late'
    previous = Interval(*config.initial_interval)
    spent = 0.
    for epoch in result['epochs']:
        before = Interval(*epoch['interval_before'])
        after = Interval(*epoch['interval_after'])
        assert before == previous and before.contains(truth) and after.contains(truth)
        assert before.lower <= after.lower <= after.upper <= before.upper
        backup = reconstruct_plan(epoch['backup'], config)
        committed = reconstruct_plan(epoch['committed'], config)
        for plan in (backup, committed):
            assert certify_plan(plan.curve, plan.tube, before, config, scenario)['safe']
            assert plan.residual <= 3e-6
        selected = [row for row in epoch['candidates'] if row['selected']]
        assert len(selected) == int(epoch['informative_committed'])
        if selected:
            row = selected[0]
            assert row['safe'] and row['reconnects'] and row['budget_feasible']
            assert reconnection(committed, backup, row['horizon'])
            feasible = [v for v in epoch['candidates'] if v['status'] in ('feasible', 'committed')]
            assert row['score'] == max(v['score'] for v in feasible)
            spent += row['exploration_cost']
        elif result['method']=='proposed':
            assert np.allclose(committed.curve.coefficients,backup.curve.coefficients,atol=1e-10,rtol=0)
            assert abs(epoch.get('planned_duration',epoch['duration'])-min(config.candidate_increment,backup.horizon))<1e-8
        assert abs(spent-epoch['exploration_spent']) < 1e-8
        assert spent <= epoch['budget']+1e-8
        # The independently reconstructed tube must contain every executed sample.
        for sample in result['samples']:
            t = sample['time']-epoch['time']
            # Boundary samples belong to the previous epoch in the saved stream.
            if t <= 1e-8 and epoch['index'] > 0:
                continue
            if -1e-8 <= t <= epoch['duration']+1e-8:
                error = np.array(sample['state'])-committed.state(max(0., t))
                assert committed.tube.utilization(error, max(0., t)) <= 1+2e-5
                assert np.all(np.abs(sample['control']) <= np.array(config.input_limits)+1e-6)
        previous = after
    if result['method'] == 'robust':
        assert previous == Interval(*config.initial_interval)
    comparison=result.get('learning_comparison')
    if comparison:
        for key in ['prior_backup','learned_backup']:
            plan=reconstruct_plan(comparison[key],config)
            interval=Interval(*comparison[key]['interval'])
            assert certify_plan(plan.curve,plan.tube,interval,config,scenario)['safe']
            assert np.max(abs(plan.state(0)-comparison['state']))<=3e-6
            assert np.max(abs(plan.state(plan.horizon)-np.r_[scenario.goal,np.zeros(3)]))<=3e-6
            phi,rho=plan.tube.at(plan.horizon)
            assert np.all(rho<=config.goal_tolerance/np.sqrt(3)+3e-6)
            assert np.all(phi+config.sliding_lambda*rho<=config.goal_velocity_tolerance/np.sqrt(3)+3e-6)
    return {'scenario': scenario.name, 'method': result['method'],
            'epochs_checked': len(result['epochs']),
            'samples_checked': len(result['samples'])}


def refine(result):
    config = ControlConfig(**result['config'])
    scenario = Scenario(**result['scenario'])
    epoch = result['epochs'][0]
    plan = reconstruct_plan(epoch['committed'], config)
    truth = SimulationTruth(**result['truth'])
    state = np.array(scenario.start)
    endpoints = []
    utilizations = []
    steps = [config.simulation_step, config.simulation_step/2, config.simulation_step/4]
    for step in steps:
        simulator = TrueDroneSimulator(truth, replace(config, simulation_step=step), scenario)
        endpoint, samples, _ = simulator.execute(plan, Interval(*epoch['interval_before']),
                                                state, 0., epoch['duration'])
        endpoints.append(endpoint)
        utilizations.append(max(v['tube_utilization'] for v in samples))
    differences = [float(np.linalg.norm(v-endpoints[-1], ord=np.inf)) for v in endpoints[:-1]]
    # Empirical integration consistency, not a proof of an RK4 truncation bound.
    assert max(differences) < .01, f'Step refinement discrepancy: {differences}'
    return {'scenario': scenario.name, 'method': result['method'], 'steps': steps,
            'duration': epoch['duration'], 'endpoint_max_abs_errors_vs_finest': differences,
            'maximum_tube_utilizations': utilizations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    files = sorted(args.directory.glob('raw/*.json')) if (args.directory/'raw').exists() else sorted(args.directory.glob('*.json'))
    records = [json.loads(path.read_text()) for path in files if not path.name.endswith('_failure.json')]
    records = [r for r in records if 'epochs' in r and 'samples' in r]
    assert records, 'No saved simulation runs found'
    audits = [audit(r) for r in records]
    refinement = []
    if args.refine:
        seen = set()
        for result in records:
            key = (result['scenario']['name'], result['method'])
            if key not in seen:
                refinement.append(refine(result)); seen.add(key)
    save_result({'audited_runs': len(audits), 'audits': audits, 'refinement': refinement},
                args.directory/'independent_validation.json')
    print(f'{len(audits)} saved runs independently verified; {len(refinement)} segments refined')


if __name__ == '__main__':
    main()
