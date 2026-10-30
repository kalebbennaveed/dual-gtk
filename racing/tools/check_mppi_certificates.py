"""Replay accepted certification batches and inspect independent holdouts."""
from pathlib import Path
import json
import numpy as np
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.control_safety import control_rollouts

root=Path('output/mppi');rows=[];tr=Track('technical')
for risk in ('01','05'):
    for method in ('proposed','weighted','passive'):
        record=json.loads((root/f'risk-{risk}'/f'{method}.json').read_text())
        accepted=[e for e in record['events'] if e['status']=='committed']
        # Prespecified diagnostic locations: first accepted policy and highest-speed commitment.
        chosen=[accepted[0],max(accepted,key=lambda e:record['frames'][round(e['time']/.05)]['state'][3])]
        for case,event in enumerate(chosen):
            a=event['selected'];c=a['certificate'];x=np.array(record['frames'][round(event['time']/.05)]['state'])
            controls=np.array(a['controls']);refs=np.array(a['references']);tail=round((c['horizon']-a['duration'])/.05)
            mus=np.random.default_rng(c['sample_seed']).uniform(*c['support'],c['samples'])
            failures,evaluated,minimum,_=control_rollouts(x,mus,controls,refs,tail,tr.curvature,tr.ds,max(abs(tr.curvature)),True,c['samples'])
            assert evaluated==c['samples'] and failures==c['failures']
            seed=80000+len(rows);n=10000
            samples=np.random.default_rng(seed).uniform(*c['support'],n)
            failures,n,margin,_=control_rollouts(x,samples,controls,refs,tail,tr.curvature,tr.ds,max(abs(tr.curvature)),True,n)
            rows.append(dict(risk=risk,method=method,selection='initial' if case==0 else 'highest_speed',time=event['time'],
                original_failures=c['failures'],original_samples=c['samples'],original_bound=c['failure_upper'],
                replay_matches=True,holdout_seed=seed,holdout_failures=int(failures),holdout_samples=int(n),holdout_rate=failures/n))
            print(rows[-1],flush=True)
output=Path('results/mppi-report');output.mkdir(exist_ok=True)
(output/'certificate_holdouts.json').write_text(json.dumps(rows,indent=2))
