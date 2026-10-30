"""Targeted replication around the low-friction failure found by probe_mppi_risk."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib,json
from probe_mppi_risk import worker
from run_mppi_stress import write_json
from dual_gtk_drs_racing.experiments import implementation_signature


if __name__=='__main__':
    root=Path('results/mppi-risk-refinement')
    for folder in ('records','outcomes'):(root/folder).mkdir(parents=True,exist_ok=True)
    signature=implementation_signature();cases=[]
    for i,mu in enumerate((.4501,.4505,.451,.4515,.452,.4525,.453,.4535,.454,.455)):
        for percent in (1,5,10):
            cases.append(dict(id=f'near-boundary-{i:02d}-{percent:02d}',trial=i,method='proposed',
                scenario=f'probe_{percent}',true_mu=mu,seed=20100,seconds=8.,initial_mu=[.45,1.05],information_weight=2.,start=0.))
    manifest=dict(implementation=signature,cases=cases,protocol='targeted_low_friction_failure_replication',
        description='Post-discovery sensitivity analysis around mu=0.45, seed=20100, start=0. Not IID and not a failure-rate estimate. Only scenario epsilon and true friction vary; controller logic unchanged.',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==manifest
    write_json(root/'manifest.json',manifest);rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for job in as_completed([pool.submit(worker,c,str(root),signature) for c in cases]):
            rows.append(job.result())
            print(f'{len(rows)}/{len(cases)} violations={sum(r.get("summary",{}).get("constraint_violated",False) for r in rows)}',flush=True)
    rows.sort(key=lambda r:r['case']['id']);write_json(root/'all_outcomes.json',rows)
    assert implementation_signature()==signature
