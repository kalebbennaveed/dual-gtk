import argparse,json
from pathlib import Path
from dataclasses import replace
from drone_dual_gatekeeper.config import load_config
from drone_dual_gatekeeper.methods.registry import METHODS
from drone_dual_gatekeeper.simulation.runner import run_method
from drone_dual_gatekeeper.simulation.simulator import SimulationTruth
from drone_dual_gatekeeper.simulation.logging import save_result

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='informative_demo');p.add_argument('--method',choices=METHODS,default='proposed')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--true-drag',type=float,default=.1/.68)
    p.add_argument('--gamma',type=float);p.add_argument('--visualize',action='store_true');p.add_argument('--output',default='results/demo');p.add_argument('--quiet',action='store_true')
    p.add_argument('--replay',type=Path);p.add_argument('--no-tubes',action='store_true');p.add_argument('--no-candidates',action='store_true')
    p.add_argument('--disturbance-profile',choices=['seeded','constant_positive'],default='seeded')
    a=p.parse_args();c,s=load_config(a.scenario)
    if a.gamma is not None:c=replace(c,weighted_gamma=a.gamma)
    if a.replay:r=json.loads(a.replay.read_text())
    else:
        r=run_method(c,s,a.method,SimulationTruth(a.true_drag,a.seed,disturbance_profile=a.disturbance_profile),a.quiet)
        save_result(r,Path(a.output)/f'{a.scenario}_{a.method}.json')
    print(json.dumps(r['metrics'],indent=2))
    if a.visualize:
        from drone_dual_gatekeeper.visualization.meshcat_visualizer import visualize
        visualize([r],Path(a.output)/f"{r['scenario']['name']}_{r['method']}.html",tubes=not a.no_tubes,candidates=not a.no_candidates)
    if not r['metrics']['success']:raise SystemExit('Mission did not complete')
if __name__=='__main__':main()
