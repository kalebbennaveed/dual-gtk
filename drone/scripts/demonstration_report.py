"""Show the causal chain from early excitation to a cheaper, faster backup.

The same-state counterfactual uses identical starting states and warm guesses.
It changes only the drag interval/estimate. It is never executed as a policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from drone_dual_gatekeeper.config import ControlConfig, Scenario, CONFIG_DIR
from drone_dual_gatekeeper.estimation.smid import update_interval
from drone_dual_gatekeeper.methods.registry import METHODS
from drone_dual_gatekeeper.planning.dual_gatekeeper import mission_cost
from drone_dual_gatekeeper.planning.optimizer import TrajectoryOptimizer
from drone_dual_gatekeeper.simulation.logging import plan_record, save_result
from drone_dual_gatekeeper.simulation.reporting import summarize, plots, plt
from drone_dual_gatekeeper.simulation.runner import shifted_backup_guess
from drone_dual_gatekeeper.simulation.simulator import SimulationTruth, TrueDroneSimulator
from drone_dual_gatekeeper.types import Interval
from scripts.validate_results import reconstruct_plan

COLORS = {'robust': '#8656a0', 'passive': '#2875b8', 'weighted': '#c4861a', 'proposed': '#1a9b55'}


def width_at(result, time):
    width = result['config']['initial_interval'][1]-result['config']['initial_interval'][0]
    for epoch in result['epochs']:
        if epoch['time']+epoch['duration'] <= time+1e-7:
            width = epoch['interval_after'][1]-epoch['interval_after'][0]
    return width


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/focused_demo'))
    parser.add_argument('--reuse-counterfactual', action='store_true')
    parser.add_argument('--scenario',default='informative_demo')
    args = parser.parse_args(); out = args.output
    protocol=yaml.safe_load((CONFIG_DIR/'demonstration_protocol.yaml').read_text())
    width_target=protocol['measurement_width_target']
    results = [json.loads((out/f'{args.scenario}_{name}.json').read_text()) for name in METHODS]
    proposed = next(r for r in results if r['method'] == 'proposed')
    config = ControlConfig(**proposed['config']); scenario = Scenario(**proposed['scenario'])
    first = proposed['epochs'][0]
    assert first['informative_committed'], 'The demonstration did not commit early excitation'
    time = first['duration']
    sample = min(proposed['samples'], key=lambda v: abs(v['time']-time))
    assert abs(sample['time']-time) < 1e-8
    state = np.array(sample['state']); prior = Interval(*first['interval_before'])
    posterior = Interval(*first['interval_after'])
    initial_backup = reconstruct_plan(first['backup'], config)
    warm = shifted_backup_guess(initial_backup, time, config)
    cached = proposed.get('learning_comparison') if args.reuse_counterfactual else None
    if cached:
        np.testing.assert_allclose(cached['state'], state, atol=1e-10)
        assert cached['prior_interval'] == [prior.lower, prior.upper]
        assert cached['posterior_interval'] == [posterior.lower, posterior.upper]
        old = reconstruct_plan(cached['prior_backup'], config)
        new = reconstruct_plan(cached['learned_backup'], config)
    else:
        old = TrajectoryOptimizer(config, scenario).solve(state, prior, warm=warm)
        new = TrajectoryOptimizer(config, scenario).solve(state, posterior, warm=warm)
    old_cost = mission_cost(old, config, np.array(scenario.goal))
    new_cost = mission_cost(new, config, np.array(scenario.goal))
    comparison = {'time': time, 'state': state.tolist(), 'prior_interval': [prior.lower, prior.upper],
                  'posterior_interval': [posterior.lower, posterior.upper],
                  'prior_cost': old_cost, 'learned_cost': new_cost,
                  'prior_horizon': old.horizon, 'learned_horizon': new.horizon,
                  'prior_backup': plan_record(old, prior), 'learned_backup': plan_record(new, posterior)}
    proposed['learning_comparison'] = comparison
    save_result(proposed, out/f'{args.scenario}_proposed.json')
    summarize(results, out); plots(results, out)

    # Isolate the information content of the first motion, before any replanning.
    # Both motions run for exactly the same duration, under the same prior,
    # true coefficient, bounded wind and physical-clock sensor noise.
    active = reconstruct_plan(first['committed'], config)
    ablation = []
    for seed in protocol['excitation_ablation_seeds']:
        row = {'seed': seed}
        for label, plan in [('mission_backup', initial_backup), ('informative', active)]:
            truth = SimulationTruth(proposed['truth']['true_drag_coefficient'], seed,
                                    disturbance_profile=proposed['truth']['disturbance_profile'])
            simulator = TrueDroneSimulator(truth, config, scenario)
            _, samples, observations = simulator.execute(plan, prior, np.array(scenario.start), 0., time)
            learned = update_interval(prior, observations, config.disturbance_bound)
            assert learned.contains(truth.true_drag_coefficient)
            row[label+'_width'] = learned.width
            row[label+'_maximum_tube_utilization'] = max(v['tube_utilization'] for v in samples)
        ablation.append(row)

    metrics = []
    for result in results:
        reached = [e['time']+e['duration'] for e in result['epochs']
                   if e['interval_after'][1]-e['interval_after'][0] <= width_target]
        metrics.append({'method': result['method'], 'width_after_informative_horizon': width_at(result, time),
                        'time_to_target_width': min(reached) if reached else None, **result['metrics']})
    selected = next(c for c in first['candidates'] if c['selected'])
    report = {'same_state_backup_comparison': comparison, 'methods': metrics,
              'first_selected_candidate': selected, 'matched_duration_excitation_ablation': ablation,
              'protocol': {'scenario': scenario.name, 'truth': proposed['truth'],
                           'measurement_noise_bound': config.measurement_noise,
                           'alpha': config.alpha, 'bandwidth_optimized': False, 'width_target':width_target,
                           'purpose': 'Designed mechanism demonstration, not a statistical superiority claim'}}
    save_result(report, out/'demonstration.json')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for obstacle in scenario.obstacles:
        if obstacle['kind']=='box':
            lo=obstacle['lower'];hi=obstacle['upper']
            axes[0,0].add_patch(plt.Rectangle(lo[:2],hi[0]-lo[0],hi[1]-lo[1],color='#999999',alpha=.6))
        else:
            axes[0, 0].add_patch(plt.Circle(obstacle['center'][:2], obstacle['radius'], color='#999999', alpha=.6))
    for result in results:
        method = result['method']; samples = result['samples']
        t = np.array([v['time'] for v in samples]); states = np.array([v['state'] for v in samples])
        label = METHODS[method].label; color = COLORS[method]
        axes[0, 0].plot(states[:, 0], states[:, 1], color=color, label=label)
        endpoints = [0]+[e['time']+e['duration'] for e in result['epochs']]
        widths = [config.initial_interval[1]-config.initial_interval[0]]+[e['interval_after'][1]-e['interval_after'][0] for e in result['epochs']]
        axes[0, 1].step(endpoints, widths, where='post', color=color, label=label)
        axes[1, 0].plot(t, [v['mission_cost'] for v in samples], color=color, label=label)
        axes[1, 1].plot(t, np.linalg.norm(states[:, 3:], axis=1), color=color, label=label)
    actual = np.array([v['state'][:3] for v in proposed['samples'] if v['time'] <= time+1e-8])
    axes[0, 0].plot(actual[:, 0], actual[:, 1], color=COLORS['proposed'], lw=4)
    axes[0, 0].scatter([scenario.start[0], scenario.goal[0]], [scenario.start[1], scenario.goal[1]], c=['#222222', '#e0a315'], marker='*', s=90)
    axes[0, 0].set_aspect('equal'); axes[0, 0].set_title('Actual paths; thick green = early informative motion')
    axes[0, 0].set_xlabel('x (m)'); axes[0, 0].set_ylabel('y (m)'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title('Intervals actually used at replanning'); axes[0, 1].set_ylabel('Drag interval width')
    axes[0, 1].axhline(width_target, color='#777777', ls='--', lw=1)
    axes[1, 0].set_title('Actual accumulated mission cost'); axes[1, 0].set_ylabel('Cost')
    axes[1, 1].set_title('Actual speed'); axes[1, 1].set_ylabel('Speed (m/s)')
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.axvline(time, color='#1a9b55', ls=':', label='Learning replan'); ax.set_xlabel('Physical time (s)')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.savefig(out/'mechanism_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    t = np.linspace(0, time, 301)
    for label, plan, color in [('Initial mission backup', initial_backup, '#2875b8'), ('Selected informative motion', active, '#1a9b55')]:
        position = plan.state(t)
        axes[0].plot(position[:, 0], position[:, 1], color=color, label=label)
        axes[1].plot(t, position[:, 3], color=color, label=label)
    axes[0].set_aspect('equal'); axes[0].set_xlabel('x (m)'); axes[0].set_ylabel('y (m)')
    axes[0].set_title(f'First {time:g} seconds: nominal excitation'); axes[0].legend(fontsize=8)
    axes[1].axhline(0, color='#777777', lw=1); axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('x velocity (m/s)')
    axes[1].set_title('Optimizer-selected velocity excitation')
    for label, plan, color in [('Prior uncertainty', old, '#777777'), ('After informative motion', new, '#1a9b55')]:
        t = np.linspace(0, plan.horizon, 301)
        axes[2].plot(t, np.linalg.norm(plan.state(t)[:, :3]-scenario.goal, axis=1), color=color, label=label)
    axes[2].set_title(f'Remaining backups from the same state at t={time:g}s')
    axes[2].set_xlabel('Planned time to go (s)'); axes[2].set_ylabel('Nominal distance to goal (m)'); axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(out/'excitation_and_replanning.png', dpi=170); plt.close(fig)

    lines = ['# Informative trajectory demonstration', '',
             f'The proposed method commits a {time:g}-second informative segment, then replans with interval width {posterior.width:.6f} (initial width {prior.width:g}).', '',
             f'From the identical measured state at t={time:g}s, the prior-uncertainty backup costs {old_cost:.3f} with horizon {old.horizon:.3f}s; the updated backup costs {new_cost:.3f} with horizon {new.horizon:.3f}s. Both pass full-tube certificates and use alpha={config.alpha:g}.', '',
             f'| Method | Width at learning replan | Time to width ≤ {width_target:g} | Arrival | Actual cost | Violations |',
             '|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        target = 'not reached' if row['time_to_target_width'] is None else f"{row['time_to_target_width']:.2f}s"
        lines.append(f"| {METHODS[row['method']].label} | {row['width_after_informative_horizon']:.6f} | {target} | {row['completion_time']:.2f}s | {row['mission_cost']:.3f} | {row['violations']} |")
    lines += ['', 'The excitation ablation repeats the two initial motions for equal durations using five sensor-noise seeds. It measures information content without subsequent replanning; it is not five complete mission trials.', '',
              'This is a deliberately designed example with constant bounded wind, a broad drag prior, and bounded velocity-measurement noise. It demonstrates the mechanism under these assumptions, not general superiority or exact paper reproduction.']
    (out/'demonstration.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
