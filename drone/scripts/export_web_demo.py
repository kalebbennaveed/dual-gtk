"""Export recorded, validated runs as a dependency-free JavaScript webpage.

No optimization happens in the browser. Exact spline/tube samples are exported
from the recorded coefficients; browser interpolation is for display only.
"""
import argparse
import base64
import html as html_module
import json
from pathlib import Path

import numpy as np

from drone_dual_gatekeeper.config import ControlConfig
from drone_dual_gatekeeper.visualization.meshcat_visualizer import reconstruct
from web_demo.scenario_previews import scenario_previews, scenario_links, SCENARIO_ROUTES


def plan_display(record, config):
    curve, tube = reconstruct(record, config)
    times = np.linspace(0, curve.horizon, 181)
    return {
        'horizon': curve.horizon,
        'cost': record['cost'],
        'points': curve.value(times).tolist(),
        'rho': [tube.at(t)[1].tolist() for t in times],
    }


def export(source, output, scenario_name='informative_demo'):
    runs = []
    for method in ('proposed', 'passive', 'weighted', 'robust'):
        result = json.loads((source / f'{scenario_name}_{method}.json').read_text())
        config = ControlConfig(**result['config'])
        epochs = []
        for epoch in result['epochs']:
            epochs.append({
                'time': epoch['time'], 'duration': epoch['duration'],
                'index': epoch['index'], 'interval': epoch['interval_before'],
                'interval_after': epoch['interval_after'],
                'informative': epoch['informative_committed'],
                'passive_fallback': epoch.get('passive_fallback',False),
                'weighted_fallback': any(row['epoch'] == epoch['index'] for row in result.get('weighted_failures', [])),
                'solver_status': epoch['committed']['solver_status'],
                'solver_converged': epoch['committed']['certificate'].get('solver_converged', True),
                'backup': plan_display(epoch['backup'], config),
                'committed': plan_display(epoch['committed'], config),
            })
        # Retain every integration sample and identification window. Live charts
        # use actual measured intervals, not the planner's information surrogate.
        samples = [{key: sample[key] for key in (
            'time', 'state', 'nominal', 'interval', 'control', 'rho',
            'mission_cost', 'tube_utilization', 'clearance',
        )} for sample in result['samples']]
        # Execution logs store the endpoint before processing the just-completed
        # measurement batch. At the exact replan timestamp show the posterior
        # event, rather than delaying the identification jump by one RK4 sample.
        for epoch in result['epochs']:
            event_time = epoch['time'] + epoch['duration']
            for sample in samples:
                if abs(sample['time'] - event_time) < 1e-8:
                    sample['interval'] = epoch['interval_after']
        metrics=dict(result['metrics'])
        # Historical recordings checked mission termination only at epoch ends.
        # Derive goal entry from their real 20 ms states without changing clocks
        # or trajectories; retain the original execution end separately.
        arrivals=[s['time'] for s in samples
                  if np.linalg.norm(np.array(s['state'][:3])-result['scenario']['goal'])<=config.goal_tolerance
                  and np.linalg.norm(s['state'][3:])<=config.goal_velocity_tolerance]
        if arrivals:
            metrics['execution_end_time']=metrics['completion_time']
            metrics['completion_time']=arrivals[0]
        runs.append({'method': method, 'metrics': metrics,
                     'samples': samples, 'epochs': epochs,
                     'weighted_gamma':config.weighted_gamma if method=='weighted' else None,
                     'information_discount':result['config'].get('weighted_information_discount',0.) if method=='weighted' else None})
        if method == 'proposed':
            scenario = result['scenario']
            comparison = result.get('learning_comparison')
            counterfactual = None
            if comparison:
                counterfactual = {key: comparison[key] for key in (
                    'time', 'prior_cost', 'learned_cost', 'prior_horizon', 'learned_horizon')}
                counterfactual['prior_backup'] = plan_display(comparison['prior_backup'], config)
    recovery_path=source/'passive_recovery.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    payload = json.dumps({'scenario': scenario, 'scenario_links':scenario_links(scenario_name), 'runs': runs, 'drone_radius': config.drone_radius,
                          'counterfactual': counterfactual,'recovery':recovery,
                          'information_advantage_threshold':config.minimum_information_advantage}, separators=(',', ':'), allow_nan=False)
    template = (Path(__file__).resolve().parents[1] / 'web_demo' / 'index.template.html').read_text()
    # Inline data makes the demo work on static hosts and directly from a file.
    # Escape HTML delimiters even though the source here is local simulation data.
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    vendor = Path(__file__).resolve().parents[1] / 'web_demo' / 'vendor'
    imports = {'three': 'three.module.js',
               'three/addons/controls/OrbitControls.js': 'OrbitControls.js'}
    import_map = {'imports': {name: 'data:text/javascript;base64,' +
                             base64.b64encode((vendor / filename).read_bytes()).decode()
                             for name, filename in imports.items()}}
    import_map['imports']['drone-geometry'] = 'data:text/javascript;base64,' + base64.b64encode(
        (vendor.parent / 'drone_geometry.js').read_bytes()).decode()
    html = template.replace('__SIMULATION_DATA__', payload).replace(
        '__IMPORT_MAP__', json.dumps(import_map))
    html = html.replace('__SCENARIO_PREVIEWS__', scenario_previews(scenario_name))
    html = html.replace('__SCENARIO_NUMBER__',str(list(SCENARIO_ROUTES).index(scenario_name)+1))
    html = html.replace('__STYLES__', (vendor.parent / 'styles.css').read_text())
    html = html.replace('__APP_SCRIPT__', (vendor.parent / 'app.js').read_text())
    license_text = html_module.escape((vendor / 'LICENSE-three.txt').read_text())
    html = html.replace('</body>', '<script type="text/plain" id="three-license">' +
                        license_text + '</script></body>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f'JavaScript demo: {output.resolve()} ({len(html):,} bytes)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/focused_demo'))
    parser.add_argument('--output', type=Path, default=Path('results/focused_demo/index.html'))
    parser.add_argument('--scenario', default='informative_demo')
    args = parser.parse_args()
    export(args.source, args.output, args.scenario)


if __name__ == '__main__':
    main()
