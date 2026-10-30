"""Explicit-control rollout adapter for the unchanged sampled DRS test.

The fixed-N design, confidence allocation, safety margins and terminal set
are imported from safety.py. Certification/execution share tracking_control.
"""
from dataclasses import asdict
import hashlib
import numpy as np
from numba import njit
from .dynamics import DT,plant_step,tracking_control,fallback_control
from .safety import DRSCertifier,Certificate,scalar_margin,terminal_margin

@njit(cache=True)
def control_rollouts(x0,mus,controls,references,tail_steps,k,ds,kmax,traction,allowed_failures):
    failures=0;minimum=1e100;steps=0
    for i in range(len(mus)):
        x=x0.copy();h=scalar_margin(x,kmax)
        for j in range(len(controls)+tail_steps):
            if h<=0:break
            u=tracking_control(x,references[j],controls[j]) if j<len(controls) else fallback_control(x,k,ds)
            x,_,_,tire=plant_step(x,mus[i],u,k,ds);steps+=1
            if traction:h=min(h,tire[0],tire[1])
            if not np.all(np.isfinite(x)):
                h=-1e100;break
            h=min(h,scalar_margin(x,kmax))
        h=min(h,terminal_margin(x,k,ds));minimum=min(minimum,h)
        failures+=int(h<=0)
        if failures>allowed_failures:return failures,i+1,minimum,steps
    return failures,len(mus),minimum,steps

class ControlCertifier(DRSCertifier):
    def check_plan(self,x,belief,controls,references):
        controls=np.ascontiguousarray(controls,dtype=float).reshape(-1,3)
        references=np.ascontiguousarray(references,dtype=float).reshape(-1,7)
        if len(controls)!=len(references):raise ValueError('One reference state per control is required')
        if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(references)):raise ValueError('Nonfinite policy')
        if len(controls) and (np.any(controls[:,0]<0) or np.any(controls[:,0]>2880) or np.any(controls[:,1]>0) or np.any(controls[:,1]<-5040) or np.any(abs(controls[:,2])>.65)):
            raise ValueError('Control outside actuator limits')
        digest=hashlib.sha256(controls.tobytes()+references.tobytes()).hexdigest()
        key=(digest,tuple(x),belief.lower,belief.upper)
        if key in self.cache:return self.cache[key]
        slot=len(self.checks)
        if slot>=self.candidate_limit:raise RuntimeError('Candidate confidence budget exceeded')
        duration=len(controls)*DT
        # Preserve the existing tail bound, using actuator-limited peak speed.
        speed=max(float(x[3]),float(np.max(references[:,3])) if len(references) else 8.)
        tail=int(np.ceil(self.tail_duration(x,speed,duration)/DT-1e-9))
        seed=int(np.random.SeedSequence([self.seed,self.decision_index,slot]).generate_state(1)[0])
        failures=evaluated=steps=0;minimum=-1e100
        if np.all(np.isfinite(x)) and self.allowed_failures>=0:
            mus=np.random.default_rng(seed).uniform(belief.lower,belief.upper,self.samples)
            failures,evaluated,minimum,steps=control_rollouts(np.asarray(x),mus,controls,references,tail,
                self.track.curvature,self.track.ds,self.kmax,self.traction,self.allowed_failures)
        complete=evaluated==self.samples;upper=float(self.upper[failures]) if complete else 1.
        accepted=complete and upper<=self.epsilon
        reason='sampled_drs_accepted' if accepted else ('insufficient_sample_budget' if self.allowed_failures<0 else 'sample_failure_limit')
        cert=Certificate(accepted,upper,float(minimum),0,reason,(belief.lower,belief.upper),(len(controls)+tail)*DT,
            self.samples,int(evaluated),int(failures),self.epsilon,self.delta,self.candidate_delta,self.candidate_limit,
            0.,0.,seed,int(steps),bool(complete))
        self.checks.append(dict(policy_hash=digest,duration=duration,tail_steps=tail,**asdict(cert)))
        self.cache[key]=cert
        return cert
