"""Prespecified fresh trials after an explicitly exploratory scenario search."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import time
import numpy as np
from .experiments import METHODS, NUMERICAL_FILES, implementation_signature, run_case, atomic_json
from .simulation import simulate
from .validation import audit_record
from .scenarios import get_scenario


def design(launch_trials=1000, lap_trials=100, seed=20261009):
    cases=[]
    # Independent RNG streams for the two cohorts, and separate certification
    # seeds. Fix the whole manifest before seeing any confirmatory outcomes.
    for group,n,seconds,methods,stream in (
        ('launch',launch_trials,8.,('proposed',),0),
        ('lap',lap_trials,360.,METHODS,1)):
        rng=np.random.default_rng(np.random.SeedSequence([seed,stream]))
        for i,mu in enumerate(rng.uniform(.76,1.05,n)):
            cert_seed=int(np.random.SeedSequence([seed,stream,i,9703]).generate_state(1)[0])
            for scenario in ('traction','traction_relaxed'):
                for method in methods:
                    cases.append(dict(id=f'{group}-{i:04d}-{scenario}-{method}',group=group,trial=i,
                        method=method,true_mu=float(mu),seed=cert_seed,seconds=seconds,scenario=scenario,
                        initial_mu=[.76,1.05],information_weight=4.,max_cells=256,start=0.))
    return cases


def run(directory,workers=4,launch_trials=1000,lap_trials=100,seed=20261009):
    directory=Path(directory).resolve();cases=design(launch_trials,lap_trials,seed)
    signature=implementation_signature();package=Path(__file__).parent
    manifest=dict(protocol='sampled_launch_stress_v1',implementation=signature,seed=seed,
        launch_trials=launch_trials,lap_trials=lap_trials,cases=cases,
        distribution='IID Uniform[0.76,1.05], one fixed true coefficient per episode',
        selection='Exploratory screen: 120 circuit start positions with the original prior found no initial-prefix failures; six lower bounds in [0.74,0.765] were then screened. The [0.76,1.05] launch setup was frozen before these independent trials. Discovery results are separate, with no inferred random-trial rate.',
        pairing='Matched true coefficient and certification seed across methods and tolerances within each cohort. The launch and full-lap cohorts use independent RNG streams.',
        safety_scope='Same plant, candidate generator and sampled certifier as the original demos. Only initial bounds change. Beta=0, assumed uniform conditional friction, alpha=0 assumed invariant backup. Epsilon/delta are per decision, not per episode.',
        failure_definition='Physical state or tire-force constraint violated by the actually selected control at the true coefficient. Certificate unavailable and audit errors are separate. Tire-force infeasibility is not a simulated crash.',
        scenarios=[asdict(get_scenario(s)) for s in ('traction','traction_relaxed')],
        sources={f:hashlib.sha256((package/f).read_bytes()).hexdigest() for f in ('stress_study.py','experiments.py','validation.py')})
    for sub in ('runs','records','sources','discovery'):(directory/sub).mkdir(parents=True,exist_ok=True)
    path=directory/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if any(old[k]!=manifest[k] for k in ('protocol','implementation','cases','sources')):
            raise ValueError('Existing stress study has a different protocol or implementation')
    else:
        atomic_json(path,manifest)
        for name in (*NUMERICAL_FILES,'stress_study.py','experiments.py','validation.py'):
            target=directory/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(package/name,target)
        for p in Path('output/stress-discovery').glob('*.json'):shutil.copy2(p,directory/'discovery'/p.name)
        shutil.copy2(package.parent/'tools/explore_safety_stress.py',directory/'discovery/explore_safety_stress.py')
    todo=[]
    for case in cases:
        p=directory/'runs'/(case['id']+'.json')
        if p.exists():
            old=json.loads(p.read_text())
            if old['case']!=case or old['implementation']!=signature:raise ValueError('Cached run differs from protocol')
        else:todo.append(case)
    print(f'{len(cases)} episodes; {len(cases)-len(todo)} cached; {workers} workers',flush=True)
    if todo:
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
        audit_record(simulate('proposed',seconds=.1,scenario='traction_relaxed',initial_mu=(.76,1.05)))
        begin=time.monotonic();done=len(cases)-len(todo);violations=0;errors=0
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=[pool.submit(run_case,c,str(directory),signature) for c in todo]
            for job in as_completed(jobs):
                row=job.result();done+=1;errors+=row['status']!='ok'
                violations+=row.get('metrics',{}).get('constraint_violated',False)
                if done%40==0 or row['status']!='ok' or done==len(cases):
                    print(f'{done}/{len(cases)} · {time.monotonic()-begin:.0f}s · violations {violations} · errors {errors}',flush=True)
    return directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/launch-stress'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--launch-trials',type=int,default=1000)
    p.add_argument('--lap-trials',type=int,default=100)
    p.add_argument('--seed',type=int,default=20261009)
    args=p.parse_args();print(run(args.output,args.workers,args.launch_trials,args.lap_trials,args.seed))


if __name__=='__main__':main()
