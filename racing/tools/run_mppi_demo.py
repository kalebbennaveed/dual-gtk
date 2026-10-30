"""Paired MPPI replays, preserved records, independent audits, and summary CSV."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,csv,time,sys
import numpy as np
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record
from dual_gtk_drs_racing.experiments import implementation_signature

VARIANTS=[('proposed','proposed',2.),('passive','passive',0.),('robust','robust',0.),
          ('weighted_zero','weighted',0.),('weighted_low','weighted',.1),('weighted','weighted',.4),('weighted_high','weighted',2.)]

def run_case(case):
    directory,key,method,weight,seconds,scenario,mu,seed,signature=case
    path=Path(directory)/f'{key}.json'
    if path.exists():
        previous=json.loads(path.read_text())
        expected=dict(seconds=seconds,scenario=scenario,true_mu=mu,information_weight=weight)
        if previous.get('implementation')==signature and previous['seed']==seed and previous['summary']['method']==method and all(previous['config'].get(k)==v for k,v in expected.items()):
            return key,previous['summary'],audit_record(previous),'cached'
    begin=time.perf_counter()
    result=simulate(method,seconds=seconds,scenario=scenario,information_weight=weight,true_mu=mu,seed=seed)
    if result['implementation']!=signature:raise RuntimeError('Source changed during run; rerun this case')
    audit=audit_record(result)
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(result,allow_nan=False));temp.replace(path)
    return key,result['summary'],audit,time.perf_counter()-begin


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('output/mppi'))
    p.add_argument('--seconds',type=float,default=360.);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--mu',type=float,default=.85);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--quick',action='store_true');p.add_argument('--only',nargs='*');p.add_argument('--risk',choices=['both','01','05'],default='both')
    a=p.parse_args();signature=implementation_signature();cases=[];rows=[]
    variants=VARIANTS[:3]+[VARIANTS[5]] if a.quick else VARIANTS
    if a.only:variants=[v for v in VARIANTS if v[0] in a.only]
    for suffix,scenario in [('01','traction'),('05','traction_relaxed')]:
        if a.risk!='both' and a.risk!=suffix:continue
        folder=a.output/f'risk-{suffix}';folder.mkdir(parents=True,exist_ok=True)
        cases.extend((str(folder),key,method,weight,a.seconds,scenario,a.mu,a.seed,signature) for key,method,weight in variants)
    manifest=dict(implementation=signature,protocol='local_mppi_paired_demonstrations',true_mu=a.mu,seed=a.seed,
        initial_mu=[.45,1.05],seconds=a.seconds,cases=cases,description='Fixed-friction matched demonstrations, not Monte Carlo confidence estimates.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run_case,c):c for c in cases}
        for task in as_completed(pending):
            c=pending[task]
            try:
                key,s,audit,wall=task.result();row=dict(scenario=c[5],variant=key,weight=c[3],**s,audit_passed=audit['passed'])
                rows.append(row);print(c[5],key,s['termination'],round(s['time'],2),round(s['distance'],1),round(s['cost'],2),'wall',wall,flush=True)
            except Exception as exc:
                rows.append(dict(scenario=c[5],variant=c[1],error=str(exc)))
                print('ERROR',c[5],c[1],repr(exc),flush=True)
            (a.output/'summary.json').write_text(json.dumps(rows,indent=2))
    if any('error' in r for r in rows):raise SystemExit('Experiment errors preserved in summary.json')
    with (a.output/'summary.csv').open('w') as f:
        fields=['scenario','variant','weight','lap_complete','termination','time','distance','cost','final_mu','informative_seconds','budget_charged','wall_seconds','audit_passed']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
