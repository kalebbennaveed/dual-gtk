"""Independent replay audits and an explicitly empirical backup stress test."""
from pathlib import Path
import json
import numpy as np
from .track import Track
from .model import Model,GRAPH,OUTPUTS
from .expressions import evaluate,lookup
from .gatekeeper import stage_cost
from dataclasses import dataclass
from .model import commands

@dataclass
class Policy:
    kind: str
    start: float
    duration: float
    speed: float
    amplitude: float = 0.
    def action_reference(self,t):
        return commands(max(0.,t-self.start),self.duration,self.speed,self.amplitude,8.)
from .belief import FrictionSet
from .safety import binomial_upper, margins
from numba import njit


def audit_record(record):
    tr=Track(record.get("scenario",{}).get("track","barcelona"));model=Model(tr);frames=record["frames"];truth=record["summary"]["true_mu"]
    belief=FrictionSet(*record["summary"]["initial_mu"])
    traction=record.get("scenario",{}).get("traction",False)
    max_error=0.;cost=0.;last_width=belief.width;event_index=-1
    policy=Policy("backup",0,0,8)
    covered_until=record["initial_certificate"].get("horizon",3.5)
    events=record["events"]
    for event in events:
        checks=event.get("certification_checks",[])
        assert len(checks)<=record['scenario']['candidate_limit']
        seeds=[c['sample_seed'] for c in checks]
        assert len(set(seeds))==len(seeds),"Certification batches reused within a decision"
        for cert in checks:
            assert cert['candidate_delta']==cert['delta']/cert['candidate_limit']
            assert cert['beta']==cert['alpha']==0
            if cert['accepted']:
                assert cert['complete_batch'] and cert['evaluated']==cert['samples']
                q=binomial_upper(cert['failures'],cert['samples'],cert['candidate_delta'])
                assert abs(q-cert['failure_upper'])<1e-12 and q<=cert['epsilon']
    for i,f in enumerate(frames):
        while event_index+1<len(events) and events[event_index+1]["time"]<=f["t"]+1e-9:
            event_index+=1;e=events[event_index]
            if e["status"] in ("retained","backup_renewal"):
                assert e["policy"]==e["previous"],"Rejected candidate replaced the commitment"
                if e["status"]=="backup_renewal":
                    assert e["certificate"]["accepted"]
                    covered_until=e["time"]+e["certificate"].get("horizon",3.5)
            elif e["status"]=="committed":
                cert=e["selected"]["certificate"]
                assert cert["accepted"] and cert["failure_upper"]<=cert["epsilon"]
                assert cert["reason"]=="sampled_drs_accepted" and cert["evaluated"]==cert["samples"]
                covered_until=e["time"]+cert.get("horizon",e["policy"]["duration"]+3.5)
                if e["selected"]["kind"]=="informative" and record["summary"]["method"]=="proposed":
                    assert any(p["duration"]==e["selected"]["duration"] and p["mission"]["accepted"] and p["informative"]["accepted"] for p in e["pairs"])
                if record["summary"]["method"]=="weighted":
                    assert any(a["duration"]==e["selected"]["duration"] and a["certificate"]["accepted"] for a in e["weighted_attempts"])
            assert e["budget_charged"]<=record["summary"]["budget_limit"]+1e-10
            policy=Policy(**e["policy"])
        lo,hi=f["mu"];width=hi-lo
        if "observation" in frames[0]:
            np.testing.assert_allclose([lo,hi],[belief.lower,belief.upper],atol=1e-10,rtol=0)
        assert lo-1e-9<=truth<=hi+1e-9,"Truth excluded"
        assert width<=last_width+1e-9,"Uncertainty set expanded"
        if i<len(frames)-1:
            assert margins(np.array(f["state"]),np.zeros(7),tr.curvature,tr.ds)>=-1e-8,"Unreported state violation"
        if i<len(frames)-1:
            assert f["certified_until"]>=f["t"]+record["dt"]-1e-9,"Execution exceeded its finite-horizon certificate"
        assert abs(f["certified_until"]-covered_until)<1e-9,"Stored coverage differs from the accepted certificate"
        assert abs(f["clearance"]-(8-model.vehicle.radius-abs(f["state"][1])))<1e-10
        assert abs(f["cost"]-cost)<1e-7,"Stored cost differs from integrated running cost"
        last_width=width
        if i==len(frames)-1: break
        x=np.array(f["state"]);refs=policy.action_reference(f["t"])
        if traction:
            h=model.traction_margin(x,truth,*refs)
            assert h>=0,"Combined tire-force limit exceeded"
            assert abs(h-f["traction_margin"])<1e-10
        nxt,u,phi,f0=model.step(x,truth,*refs)
        if "observation" in f:
            obs=f["observation"];g=np.array(obs["regressor"]);y=np.array(obs["residual"]);w=np.array(obs["bound"])
            expected=phi[5:6] if record["scenario"]["sensor"]=="quantized_yaw" else phi[3:6]
            np.testing.assert_allclose(g,expected,atol=1e-10,rtol=0)
            assert np.all(np.abs(y-truth*g)<=w+1e-10),"Measurement exceeds its declared error bound"
            if record["scenario"]["sensor"]=="quantized_yaw":
                rate=(nxt-x)/model.dt
                np.testing.assert_allclose(y,[.36*np.rint(rate[5]/.36)-f0[5]],atol=1e-10,rtol=0)
            if record["summary"]["method"]!="robust":belief.update(y,g,w)
        error=float(np.max(np.abs(nxt-frames[i+1]["state"])))
        max_error=max(max_error,error)
        assert error<1e-8,"Recorded plant differs from the executed committed policy"
        assert np.max(np.abs(u-f["control"]))<1e-8
        cost+=model.dt*stage_cost(np.array([f["state"],frames[i+1]["state"]]),u[None,:],record.get("scenario",{}).get("racing_speed",24.))[0]
    final=frames[-1];termination=record['summary'].get('termination')
    if termination=='constraint_violation':
        violation=record['summary']['violation']
        if violation['kind']=='state':assert margins(np.array(final['state']),np.zeros(7),tr.curvature,tr.ds)<-1e-8
        else:assert model.traction_margin(np.array(final['state']),truth,*policy.action_reference(final['t']))<0
    elif termination!='certificate_unavailable':
        assert margins(np.array(final['state']),np.zeros(7),tr.curvature,tr.ds)>=-1e-8
    return dict(method=record["summary"]["method"],passed=True,frames=len(frames),
                max_replay_error=max_error,minimum_clearance=min(f["clearance"] for f in frames),cost=float(cost))


def audit_directory(path,stress=False):
    records=[json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json")) if p.stem in ("proposed","passive","weighted","robust")]
    if not records: raise ValueError("No run records found")
    results=[audit_record(r) for r in records]
    (Path(path)/"audit.json").write_text(json.dumps(results,indent=2))
    if stress:
        tr=Track(records[0].get("scenario",{}).get("track","barcelona"));frictions=np.linspace(.45,1.05,13)
        clearance,lateral,heading=backup_stress(tr.curvature,tr.ds,GRAPH,OUTPUTS,frictions)
        report=dict(type="empirical_backup_stress_not_invariance_proof",frictions=frictions.tolist(),
            initial_offsets=[-1,0,1],laps=39,minimum_clearance=clearance,max_lateral_error=lateral,max_heading_error=heading)
        (Path(path)/"backup_stress.json").write_text(json.dumps(report,indent=2))
    return results


@njit(cache=True)
def backup_stress(k,ds,graph,outputs,frictions):
    minimum=1e9;max_offset=0.;max_heading=0.
    for mu in frictions:
        for offset in (-1.,0.,1.):
            x=np.array([0.,offset,0.,8.,0.,0.,0.])
            for _ in range(12000):
                inp=np.empty(11);inp[:7]=x;inp[7:]=np.array([mu,8.,0.,0.])
                x=evaluate(inp,graph,outputs,k,ds)[:7]
                minimum=min(minimum,5.7-abs(x[1]));max_offset=max(max_offset,abs(x[1]));max_heading=max(max_heading,abs(x[2]))
                if x[0]>=ds*(len(k)-1):break
    return minimum,max_offset,max_heading
