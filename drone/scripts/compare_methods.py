import argparse,json
from pathlib import Path
from dataclasses import replace
from drone_dual_gatekeeper.config import load_config
from drone_dual_gatekeeper.methods.registry import METHODS
from drone_dual_gatekeeper.simulation.runner import run_method
from drone_dual_gatekeeper.simulation.simulator import SimulationTruth
from drone_dual_gatekeeper.simulation.logging import save_result
from drone_dual_gatekeeper.simulation.reporting import summarize,plots

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenario',default='informative_demo');p.add_argument('--visualize',action='store_true');p.add_argument('--overlay',action='store_true')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--true-drag',type=float,default=.1/.68);p.add_argument('--output',default='results/demo');p.add_argument('--replay',action='store_true');p.add_argument('--quiet',action='store_true');p.add_argument('--gamma',type=float)
    p.add_argument('--disturbance-profile',choices=['seeded','constant_positive'],default='seeded')
    a=p.parse_args();c,s=load_config(a.scenario)
    if a.gamma is not None:c=replace(c,weighted_gamma=a.gamma)
    results=[]
    for method in METHODS:
        path=Path(a.output)/f'{a.scenario}_{method}.json'
        if a.replay:r=json.loads(path.read_text())
        else:r=run_method(c,s,method,SimulationTruth(a.true_drag,a.seed,disturbance_profile=a.disturbance_profile),a.quiet);save_result(r,path)
        results.append(r)
    summarize(results,Path(a.output)/a.scenario);plots(results,Path(a.output)/a.scenario)
    print((Path(a.output)/a.scenario/'comparison.md').read_text())
    if a.visualize:
        from drone_dual_gatekeeper.visualization.meshcat_visualizer import visualize
        visualize(results,Path(a.output)/f'{a.scenario}_comparison.html',comparison=True,overlay=a.overlay)
    if not all(r['metrics']['success'] for r in results):raise SystemExit('At least one mission did not complete')
if __name__=='__main__':main()
