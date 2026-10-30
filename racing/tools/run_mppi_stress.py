"""Frozen paired launch stress study of the current MPPI controller; retain every outcome."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse, csv, gzip, hashlib, json, os, time
import numpy as np
from scipy.stats import beta
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record
from dual_gtk_drs_racing.experiments import implementation_signature


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def run(case, root, signature):
    root=Path(root);out=root/'outcomes'/f"{case['id']}.json";record=root/'records'/f"{case['id']}.json.gz"
    if out.exists():
        previous=json.loads(out.read_text())
        if previous['case']!=case or previous['implementation']!=signature:raise ValueError('Cached case mismatch')
        if 'summary' in previous and (not record.exists() or hashlib.sha256(record.read_bytes()).hexdigest()!=previous['record_sha256']):raise ValueError('Cached record missing or changed')
        return previous
    row=dict(case=case,implementation=signature)
    try:
        result=simulate(**{k:v for k,v in case.items() if k not in ('id','trial')})
        if result['implementation']!=signature:raise RuntimeError('Controller source changed during experiment')
        row['summary']=result['summary']
        temp=record.with_suffix('.tmp')
        with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(result,f,allow_nan=False,separators=(',',':'))
        temp.replace(record);row['record_sha256']=hashlib.sha256(record.read_bytes()).hexdigest()
        try:row['audit']=audit_record(result)
        except Exception as exc:row['audit_error']=repr(exc)
        row['minimum_tire_margin']=min((f['traction_margin'] for f in result['frames'] if 'traction_margin' in f),default=None)
    except Exception as exc:row['error']=repr(exc)
    write_json(out,row);return row


def aggregate(rows, cases):
    stats=[]
    for scenario in ('traction','traction_relaxed'):
        group=[r for r in rows if r['case']['scenario']==scenario];n=len(group)
        violations=sum(bool(r.get('summary',{}).get('constraint_violated')) for r in group)
        errors=sum('error' in r for r in group);audits=sum('audit_error' in r for r in group)
        unavailable=sum(bool(r.get('summary',{}).get('certificate_unavailable')) for r in group)
        complete=sum(r.get('summary',{}).get('termination')=='timeout' and r.get('summary',{}).get('time')==8. and r.get('audit',{}).get('passed',False) for r in group)
        unresolved=sum('error' in r or 'audit_error' in r or r.get('summary',{}).get('certificate_unavailable',False) for r in group)
        bad=sum(r.get('summary',{}).get('constraint_violated',False) or 'error' in r or 'audit_error' in r or r.get('summary',{}).get('certificate_unavailable',False) for r in group)
        stats.append(dict(scenario=scenario,method='proposed',planned=sum(c['scenario']==scenario for c in cases),executed=n,
            safe_horizon_completed=complete,violations=violations,certificate_unavailable=unavailable,simulation_errors=errors,audit_errors=audits,unresolved=unresolved,
            violation_rate=violations/n if n else None,ci95_lower=float(beta.ppf(.025,violations,n-violations+1)) if violations else 0.,
            ci95_upper=float(beta.isf(.025,violations+1,n-violations)) if violations<n else 1.,
            unsafe_or_unresolved_upper95=float(beta.isf(.05,bad+1,n-bad)) if bad<n else 1.))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('results/mppi-stress'))
    parser.add_argument('--trials',type=int,default=1000);parser.add_argument('--workers',type=int,default=6)
    a=parser.parse_args();root=a.output
    for folder in ('records','outcomes'):(root/folder).mkdir(parents=True,exist_ok=True)
    signature=implementation_signature();rng=np.random.default_rng(2026101301);cases=[]
    for trial,mu in enumerate(rng.uniform(.76,1.05,a.trials)):
        for scenario in ('traction','traction_relaxed'):
            cases.append(dict(id=f'{trial:04d}-{scenario}',trial=trial,method='proposed',scenario=scenario,true_mu=float(mu),seed=10000+trial,
                              seconds=8.,initial_mu=[.76,1.05],information_weight=2.,start=0.))
    manifest=dict(implementation=signature,protocol='current_mppi_paired_8_second_launch_stress',seed=2026101301,cases=cases,
        trials_per_tolerance=a.trials,seconds=8.,initial_mu=[.76,1.05],methods=['proposed'],tolerances=[.01,.05],
        description='Prespecified IID Uniform[0.76,1.05] friction draws, fixed within each eight-second episode. Same friction and controller seed paired across tolerances. Standard current MPPI settings and DRS checks; fixed proposed gamma=2. No search, tuning, discarded failures, or retries until success. These are closed-loop launches, not full laps or candidate-certification rollouts.',
        failure_definition='Any executed state or tire-force constraint violation. Certificate unavailable, simulation errors and audit errors are reported separately and included as unresolved outcomes.',
        safety_scope='Risk/confidence are per certification decision. Episode violation intervals are empirical two-sided 95% Clopper-Pearson; also report a one-sided 95% unsafe-or-unresolved upper bound.',
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:raise SystemExit('Existing frozen manifest differs; use a new output directory')
    write_json(root/'manifest.json',manifest);rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run,c,str(root),signature):c for c in cases}
        for task in as_completed(pending):
            row=task.result();rows.append(row)
            if len(rows)%10==0 or len(rows)==len(cases):
                stats=aggregate(rows,cases);write_json(root/'aggregate.json',stats)
                write_json(root/'progress.json',dict(completed=len(rows),planned=len(cases),elapsed_seconds=time.monotonic()-started,complete=len(rows)==len(cases)))
                print(f"{len(rows)}/{len(cases)} | violations={sum(s['violations'] for s in stats)} unresolved={sum(s['unresolved'] for s in stats)} elapsed={time.monotonic()-started:.1f}s",flush=True)
    rows.sort(key=lambda r:r['case']['id']);write_json(root/'all_outcomes.json',rows)
    fields=['id','trial','scenario','true_mu','seed','seconds','termination','time','constraint_violated','certificate_unavailable','violation_kind','minimum_tire_margin','audit_passed','error','audit_error','record_sha256']
    with (root/'all_episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:
            s=r.get('summary',{});row={k:r['case'][k] for k in fields if k in r['case']};row.update({k:s[k] for k in fields if k in s})
            row.update(violation_kind=(s.get('violation') or {}).get('kind',''),minimum_tire_margin=r.get('minimum_tire_margin'),audit_passed=r.get('audit',{}).get('passed',False),error=r.get('error',''),audit_error=r.get('audit_error',''),record_sha256=r.get('record_sha256',''))
            writer.writerow(row)
    if any('error' in r or 'audit_error' in r for r in rows):raise SystemExit('All outcomes retained; inspect recorded errors')

if __name__=='__main__':main()
