"""Run or replay the complete focused informative-motion demonstration."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml
from drone_dual_gatekeeper.config import CONFIG_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/focused_demo')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol=yaml.safe_load((CONFIG_DIR/'demonstration_protocol.yaml').read_text())
    common = ['--scenario', protocol['scenario'], '--true-drag', str(protocol['true_drag_coefficient']),
              '--seed', str(protocol['seed']), '--disturbance-profile', protocol['disturbance_profile'], '--output', args.output]
    def run(script, parameters):
        subprocess.run([sys.executable, '-m', 'scripts.'+Path(script).stem, *parameters], cwd=root, check=True)
    run('compare_methods.py', common+(['--replay'] if args.replay else []))
    run('demonstration_report.py', ['--output', args.output]+(['--reuse-counterfactual'] if args.replay else []))
    run('validate_results.py', [args.output, '--refine'])
    run('export_web_demo.py', ['--source', args.output,
                              '--output', str(Path(args.output)/'index.html')])
    if args.visualize:
        run('compare_methods.py', common+['--replay', '--visualize'])
        run('run_simulation.py', ['--replay', str(Path(args.output)/'informative_demo_proposed.json'),
                                 '--output', args.output, '--visualize'])
    run('write_source_manifest.py', ['--output', args.output])


if __name__ == '__main__':
    main()
