"""Replay explicit controls, measurement strips, commitments and DRS decisions."""
from pathlib import Path
import json
import numpy as np
from .track import Track
from .gatekeeper import Policy
from .dynamics import DT,plant_step,running_cost
from .belief import FrictionSet,observation
from .safety import binomial_upper,scalar_margin


def audit_record(record):
    if record.get('schema',2)<3:
        from .legacy_validation import audit_record as legacy_audit
        return legacy_audit(record)
    tr=Track(record['scenario']['track']);kmax=float(max(abs(tr.curvature)))
    summary=record['summary'];truth=summary['true_mu'];belief=FrictionSet(*summary['initial_mu'])
    frames=record['frames'];events=record['events'];policy=Policy();index=0;cost=0.;error=0.;budget=0.
    coverage=record['initial_certificate']['horizon'];rng=np.random.default_rng(record['seed'])
    def check_certificate(c):
        assert c['candidate_delta']==c['delta']/c['candidate_limit']
        assert c['beta']==0 and c['alpha']==0
        if c['complete_batch']:
            assert c['evaluated']==c['samples']
            assert abs(binomial_upper(c['failures'],c['samples'],c['candidate_delta'])-c['failure_upper'])<1e-12
        else:assert not c['accepted'] and c['failure_upper']==1.
        assert c['accepted']==(c['complete_batch'] and c['failure_upper']<=c['epsilon'])
    check_certificate(record['initial_certificate'])
    assert record['initial_certificate']['accepted']
    for e in events:
        checks=e.get('certification_checks',[])
        assert len(checks)<=record['scenario']['candidate_limit']
        assert len({c['sample_seed'] for c in checks})==len(checks)
        for c in checks:check_certificate(c)
        if e['status']=='committed':
            a=e['selected'];check_certificate(a['certificate']);assert a['certificate']['accepted']
            assert len(e['policy']['controls'])==int(round(e['policy']['duration']/DT))
            assert e['policy']['controls']==a['controls'] and e['policy']['references']==a['references']
            if summary['method']=='proposed' and a['kind']=='informative':
                pair=next(p for p in e['pairs'] if abs(p['duration']-a['duration'])<1e-9)
                assert pair['mission']['accepted'] and pair['informative']['accepted']
                charge=max(0.,pair['informative_cost']-pair['mission_cost'])
                assert abs(charge-a['charge'])<1e-9
                candidates=[v for v in e['informative_attempts'] if v['paired_safe'] and v['budget_feasible'] and v['gain']>1e-6]
                assert a['score']==max(v['score'] for v in candidates)
                for v in e['informative_attempts']:
                    assert abs(v['score']-np.exp(-record['config']['planner_config']['selection_discount']*v['duration'])*v['gain'])<1e-10
                budget+=charge
        elif e['status'] in ('retained','backup_renewal','backup_unavailable'):
            assert e['policy']==e['previous'],'Rejected plan changed commitment'
        assert abs(e['budget_charged']-budget)<1e-8 and budget<=summary['budget_limit']+1e-9
    for i,f in enumerate(frames):
        while index<len(events) and events[index]['time']<=f['t']+1e-9:
            e=events[index];policy=Policy(**e['policy']);index+=1
            if e['status']=='committed':coverage=e['time']+e['selected']['certificate']['horizon']
            if e['status']=='backup_renewal':
                check_certificate(e['certificate']);assert e['certificate']['accepted']
                coverage=e['time']+e['certificate']['horizon']
        x=np.array(f['state']);assert abs(f['certified_until']-coverage)<1e-8
        np.testing.assert_allclose(f['mu'],[belief.lower,belief.upper],atol=1e-10,rtol=0)
        assert belief.lower-1e-9<=truth<=belief.upper+1e-9
        assert abs(f['cost']-cost)<1e-7
        assert abs(f['clearance']-(5.7-abs(x[1])))<1e-10
        if i==len(frames)-1:break
        assert coverage>=f['t']+DT-1e-9
        assert scalar_margin(x,kmax)>=-1e-8
        u=policy.action(x,f['t'],tr)
        np.testing.assert_allclose(u,f['control'],atol=1e-10,rtol=0)
        nxt,phi,f0,tire=plant_step(x,truth,u,tr.curvature,tr.ds)
        if record['scenario']['traction']:
            assert min(tire)>=0 and abs(min(tire)-f['traction_margin'])<1e-10
        expected=observation((nxt-x)/DT,phi,f0,rng,record['scenario']['sensor'])
        obs=f['observation']
        for a,key in zip(expected,('residual','regressor','bound')):
            np.testing.assert_allclose(a,obs[key],atol=1e-10,rtol=0)
        assert np.all(abs(expected[0]-truth*expected[1])<=expected[2]+1e-10)
        if summary['method']!='robust':belief.update(*expected)
        err=float(np.max(abs(nxt-np.array(frames[i+1]['state']))));error=max(error,err)
        assert err<1e-8,'Executed controls differ from recorded policy'
        cost+=DT*running_cost(x,u,record['scenario']['racing_speed'])
    final=np.array(frames[-1]['state'])
    if summary['constraint_violated']:
        if summary['violation']['kind']=='state':assert scalar_margin(final,kmax)<-1e-8
        else:
            u=policy.action(final,frames[-1]['t'],tr)
            assert min(plant_step(final,truth,u,tr.curvature,tr.ds)[3])<0
    elif summary['termination']!='certificate_unavailable':assert scalar_margin(final,kmax)>=-1e-8
    return dict(method=summary['method'],passed=True,frames=len(frames),max_replay_error=error,
                minimum_clearance=min(f['clearance'] for f in frames),cost=float(cost))


def audit_directory(path,stress=False):
    records=[json.loads(p.read_text()) for p in sorted(Path(path).glob('*.json')) if p.stem in ('proposed','passive','robust') or p.stem.startswith('weighted')]
    if not records:raise ValueError('No run records found')
    results=[audit_record(r) for r in records]
    (Path(path)/'audit.json').write_text(json.dumps(results,indent=2))
    if stress:
        from .legacy_validation import backup_stress
        from .model import GRAPH,OUTPUTS
        tr=Track(records[0]['scenario']['track'])
        result=backup_stress(tr.curvature,tr.ds,GRAPH,OUTPUTS,np.linspace(.45,1.05,13))
        (Path(path)/'backup_stress.json').write_text(json.dumps(dict(type='empirical_not_invariance_proof',metrics=list(result))))
    return results
