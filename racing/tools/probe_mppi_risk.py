"""Exploratory risk-tolerance probes; change scenario parameters, never controller logic."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import argparse, csv, gzip, hashlib, json, time
import numpy as np
from dual_gtk_drs_racing.experiments import implementation_signature
from dual_gtk_drs_racing.scenarios import SCENARIOS
from dual_gtk_drs_racing.track import Track
from run_mppi_stress import run, write_json


def worker(case, root, signature):
    # Existing Scenario configuration API: only epsilon differs from traction.
    for percent in (1, 5, 10):
        name=f'probe_{percent}'
        SCENARIOS[name]=replace(SCENARIOS['traction'],name=name,epsilon=percent/100,
                               title=f'Exploratory {percent}% tolerance')
    return run(case, root, signature)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/mppi-risk-probe'))
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();root=a.output
    for folder in ('records','outcomes'):(root/folder).mkdir(parents=True,exist_ok=True)
    signature=implementation_signature();cases=[]
    # First 64 frozen friction draws from the completed launch study, now at 10%.
    rng=np.random.default_rng(2026101301)
    for trial,mu in enumerate(rng.uniform(.76,1.05,64)):
        cases.append(dict(id=f'paired-{trial:03d}',trial=trial,method='proposed',scenario='probe_10',
            true_mu=float(mu),seed=10000+trial,seconds=8.,initial_mu=[.76,1.05],information_weight=2.,start=0.))
    # Deliberately oversample low-friction boundaries: these are stress cases, not IID estimates.
    length=Track('technical').length
    for start_index,start in enumerate(np.arange(4)*length/4):
        for mu_index,mu in enumerate((.45,.455,.46,.47,.48,.50,.55,.65,.85,1.05)):
            trial=100+10*start_index+mu_index
            for percent in (1,5,10):
                cases.append(dict(id=f'boundary-{start_index}-{mu_index:02d}-{percent:02d}',trial=trial,
                    method='proposed',scenario=f'probe_{percent}',true_mu=mu,seed=20000+trial,
                    seconds=8.,initial_mu=[.45,1.05],information_weight=2.,start=float(start)))
    manifest=dict(implementation=signature,cases=cases,
        protocol='exploratory_current_mppi_risk_tolerance_probe',
        description='64 paired original-distribution launches at 10%; 40 targeted low-friction/start-position cases at each of 1%, 5%, 10%. Fixed gamma=2. All simulation and audit outcomes retained. Boundary counts are not estimates of IID failure probability.',
        configuration='Register temporary Scenario copies in each worker, differing only in name, title and epsilon from traction. No numerical source files or controller logic changed.',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (root/'manifest.json').exists():assert json.loads((root/'manifest.json').read_text())==manifest
    write_json(root/'manifest.json',manifest);rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(worker,c,str(root),signature) for c in cases]
        for job in as_completed(jobs):
            rows.append(job.result())
            if len(rows)%8==0 or len(rows)==len(cases):
                count=sum(r.get('summary',{}).get('constraint_violated',False) for r in rows)
                errors=sum('error' in r or 'audit_error' in r for r in rows)
                print(f'{len(rows)}/{len(cases)} violations={count} errors={errors} elapsed={time.monotonic()-started:.1f}s',flush=True)
                write_json(root/'progress.json',dict(completed=len(rows),planned=len(cases),violations=count,errors=errors,complete=len(rows)==len(cases)))
    rows.sort(key=lambda r:r['case']['id'])
    for r in rows:
        if 'summary' not in r:continue
        record=json.load(gzip.open(root/'records'/f"{r['case']['id']}.json.gz",'rt'))
        selected=[e['selected']['certificate'] for e in record['events'] if e['status']=='committed']
        r['selected_certificates']=len(selected)
        r['selected_with_sampled_failures']=sum(c['failures']>0 for c in selected)
        r['max_selected_sampled_failures']=max((c['failures'] for c in selected),default=0)
    write_json(root/'all_outcomes.json',rows)
    fields=['id','scenario','true_mu','start','seed','termination','time','violation_kind','minimum_tire_margin','selected_certificates','selected_with_sampled_failures','max_selected_sampled_failures','audit_passed','error','audit_error']
    with (root/'all_episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:
            s=r.get('summary',{});out={k:r['case'][k] for k in fields if k in r['case']}
            out.update({k:s[k] for k in fields if k in s});out.update({k:r[k] for k in fields if k in r})
            out.update(violation_kind=(s.get('violation') or {}).get('kind',''),audit_passed=r.get('audit',{}).get('passed',False))
            w.writerow(out)
    assert implementation_signature()==signature
    print('Complete; all outcomes retained.',flush=True)


if __name__=='__main__':main()
