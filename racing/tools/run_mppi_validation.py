"""Prespecified paired full-lap validation draws; all failures are retained."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,gzip,time,os
import numpy as np
from scipy.stats import beta
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record
from dual_gtk_drs_racing.experiments import implementation_signature


def run(case,root,signature):
    name=case['id'];path=Path(root)/'records'/f'{name}.json.gz'
    if path.exists():
        with gzip.open(path,'rt') as f:r=json.load(f)
        expected={k:case[k] for k in ('seconds','scenario','true_mu','information_weight')}
        if r['implementation']==signature and r['seed']==case['seed'] and r['summary']['method']==case['method'] and all(r['config'].get(k)==v for k,v in expected.items()):
            return dict(case=case,summary=r['summary'],audit=audit_record(r))
    args={k:v for k,v in case.items() if k not in ('id','trial')}
    r=simulate(**args)
    if r['implementation']!=signature:raise RuntimeError('Source changed during validation')
    temp=path.with_name(f'.{name}.{os.getpid()}.tmp')
    with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(r,f,allow_nan=False)
    temp.replace(path)
    return dict(case=case,summary=r['summary'],audit=audit_record(r))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results/mppi-validation'))
    p.add_argument('--trials',type=int,default=8);p.add_argument('--workers',type=int,default=3);p.add_argument('--seconds',type=float,default=360.)
    a=p.parse_args();(a.output/'records').mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261013);signature=implementation_signature();cases=[]
    for trial,mu in enumerate(rng.uniform(.45,1.05,a.trials)):
        for scenario in ('traction','traction_relaxed'):
            for method in ('proposed','passive','weighted','robust'):
                cases.append(dict(id=f'{trial:03d}-{scenario}-{method}',trial=trial,method=method,scenario=scenario,
                    true_mu=float(mu),seed=3000+trial,seconds=a.seconds,information_weight=2. if method=='proposed' else .4 if method=='weighted' else 0.))
    manifest=dict(implementation=signature,seed=20261013,cases=cases,
        description='Fresh paired IID friction draws, fixed within an episode. Same realization and controller seed across methods/tolerances. Proposed uses the fixed gamma=2 informative objective; only the weighted baseline has a weight sweep in the separate demonstrations. All planned trials counted.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run,c,str(a.output),signature):c for c in cases}
        for task in as_completed(pending):
            c=pending[task]
            try:
                row=task.result();s=row['summary'];print(c['id'],round(c['true_mu'],4),s['termination'],s['time'],flush=True)
            except Exception as exc:row=dict(case=c,error=repr(exc));print('ERROR',c['id'],repr(exc),flush=True)
            rows.append(row);(a.output/'outcomes.json').write_text(json.dumps(rows,indent=2))
    aggregates=[]
    for scenario in ('traction','traction_relaxed'):
        for method in ('proposed','passive','weighted','robust'):
            group=[r for r in rows if r['case']['scenario']==scenario and r['case']['method']==method]
            errors=sum('error' in r for r in group)
            violations=sum(r.get('summary',{}).get('constraint_violated',False) for r in group)
            unresolved=sum(r.get('summary',{}).get('certificate_unavailable',False) for r in group)+errors
            completed=[r['summary'] for r in group if r.get('summary',{}).get('lap_complete')]
            k=violations+unresolved;n=len(group)
            aggregates.append(dict(scenario=scenario,method=method,trials=n,completed=len(completed),violations=violations,
                unresolved=unresolved,audit_errors=errors,unsafe_or_unresolved_upper95=float(beta.isf(.05,k+1,n-k)) if k<n else 1.,
                mean_completed_lap=float(np.mean([s['time'] for s in completed])) if completed else None,
                mean_completed_cost=float(np.mean([s['cost'] for s in completed])) if completed else None))
    (a.output/'aggregate.json').write_text(json.dumps(aggregates,indent=2))
    if any('error' in r for r in rows):raise SystemExit('Errors preserved in outcomes.json')
if __name__=='__main__':main()
