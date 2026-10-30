"""Paired MPPI prefixes, sampled DRS certification, and budgeted selection."""
from dataclasses import dataclass,asdict,field
import numpy as np
from .belief import predicted_quantized_width,predicted_width
from .control_safety import ControlCertifier
from .dynamics import DT,tracking_control,fallback_control,plant_step,running_cost
from .mppi import MPPI,MPPIConfig
from .scenarios import get_scenario

# Fixed generation objective for the proposed method, not a selection weight.
PROPOSED_INFORMATION_WEIGHT = 2.

@dataclass
class Policy:
    kind: str = 'backup'
    start: float = 0.
    duration: float = 0.
    controls: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def action(self,x,t,track):
        index=int(round((t-self.start)/DT))
        if 0<=index<len(self.controls):
            return tracking_control(x,np.asarray(self.references[index]),np.asarray(self.controls[index]))
        return fallback_control(x,track.curvature,track.ds)


def stage_cost(xs,us,target_speed=24.):
    return np.array([running_cost(x,u,target_speed) for x,u in zip(xs[:-1],us)])

class Gatekeeper:
    def __init__(self,model,method='proposed',budget=None,max_cells=256,scenario='barcelona',information_weight=.4,seed=7,planner_config=None):
        if method not in ('proposed','passive','weighted','robust'):raise ValueError('Unknown method')
        self.model=model;self.method=method;self.scenario=get_scenario(scenario)
        self.config=planner_config or MPPIConfig()
        if any(n>self.config.horizon or n<1 for n in self.config.candidate_steps):raise ValueError('Candidate exceeds planner horizon')
        self.certifier=ControlCertifier(model.track,backup_duration=self.scenario.backup_duration,traction=self.scenario.traction,
            epsilon=self.scenario.epsilon,delta=self.scenario.delta,samples=self.scenario.samples,candidate_limit=self.scenario.candidate_limit,seed=seed)
        args=(model.track,self.scenario.racing_speed,self.scenario.sensor,self.scenario.traction)
        self.nominal=MPPI(*args,seed=seed+101,config=self.config)
        self.informative=MPPI(*args,seed=seed+202,config=self.config)
        self.committed=Policy();self.charged=0.;self.budget=self.config.budget if budget is None else float(budget)
        if not np.isfinite(information_weight) or information_weight<0:raise ValueError('information_weight must be finite and nonnegative')
        self.information_weight=PROPOSED_INFORMATION_WEIGHT if method=='proposed' else float(information_weight) if method=='weighted' else 0.
        self.events=[];self.last_action=fallback_control(np.array([0.,0,0,8.,0,0,0]),model.track.curvature,model.track.ds)

    def candidate(self,plan,n,belief):
        states=plan['states'][:n+1];controls=plan['controls'][:n];phis=plan['phis'][:n]
        remaining=predicted_quantized_width(belief,states,phis) if self.scenario.sensor=='quantized_yaw' else predicted_width(belief.width,phis)
        return dict(duration=n*DT,steps=n,cost=float(DT*stage_cost(states,controls,self.scenario.racing_speed).sum()),
            gain=max(0.,belief.width-remaining),predicted_width=float(remaining),progress=float(states[-1,0]-states[0,0]),
            controls=controls.tolist(),references=states[:-1].tolist())

    def certify(self,x,belief,a):
        cert=self.certifier.check_plan(x,belief,a['controls'],a['references'])
        a['certificate']=asdict(cert)
        return cert.accepted

    def replan(self,x,belief,t):
        # Honor a paid informative prefix in full; its phase and certified policy stay fixed.
        if self.method=='proposed' and self.committed.kind=='informative' and t<self.committed.start+self.committed.duration-1e-8:return
        self.certifier.begin_decision(t)
        event=dict(time=float(t),support=[belief.lower,belief.upper],previous=asdict(self.committed),pairs=[],
                   nominal_attempts=[],informative_attempts=[],weighted_attempts=[],budget_charged=self.charged,
                   information_weight=self.information_weight,planner='local_mppi')
        selected=None;plans={}
        if self.method=='weighted':
            plan=self.nominal.optimize(x,belief.center,t,self.last_action,gamma=self.information_weight)
            plans['weighted']=plan
            for n in self.config.candidate_steps:
                a=self.candidate(plan,n,belief);self.certify(x,belief,a)
                a.update(kind='informative' if self.information_weight>0 else 'mission',charge=0.,score=0.)
                event['weighted_attempts'].append(a)
            valid=[a for a in event['weighted_attempts'] if a['certificate']['accepted']]
            if valid:selected=max(valid,key=lambda a:a['duration'])
        else:
            nominal=self.nominal.optimize(x,belief.center,t,self.last_action)
            plans['nominal']=nominal
            for n in self.config.candidate_steps:
                a=self.candidate(nominal,n,belief);self.certify(x,belief,a)
                a.update(kind='mission',charge=0.,score=0.);event['nominal_attempts'].append(a)
            if self.method=='proposed':
                info=self.informative.optimize(x,belief.center,t,self.last_action,gamma=self.information_weight,initial=nominal['coordinates'])
                plans['informative']=info
                for n,mission in zip(self.config.candidate_steps,event['nominal_attempts']):
                    a=self.candidate(info,n,belief);self.certify(x,belief,a)
                    charge=max(0.,a['cost']-mission['cost'])
                    score=float(np.exp(-self.config.selection_discount*a['duration'])*a['gain'])
                    a.update(kind='informative',charge=charge,score=score)
                    a['paired_safe']=bool(mission['certificate']['accepted'] and a['certificate']['accepted'])
                    a['budget_feasible']=self.charged+charge<=self.budget+1e-12
                    event['informative_attempts'].append(a)
                    event['pairs'].append(dict(duration=a['duration'],mission=mission['certificate'],informative=a['certificate'],
                        mission_cost=mission['cost'],informative_cost=a['cost'],charge=charge,gain=a['gain'],score=score,budget_feasible=a['budget_feasible']))
                eligible=[a for a in event['informative_attempts'] if a['paired_safe'] and a['budget_feasible'] and a['gain']>1e-6]
                if eligible:selected=max(eligible,key=lambda a:a['score'])
            if selected is None:
                valid=[a for a in event['nominal_attempts'] if a['certificate']['accepted']]
                if valid:selected=max(valid,key=lambda a:a['duration'])
        event['planner_diagnostics']={name:p['diagnostics'] for name,p in plans.items()}
        event['candidate_plans']={name:self.model.track.pose(p['states'])[::2].tolist() for name,p in plans.items()}
        if selected is not None:
            self.committed=Policy(selected['kind'],t,selected['duration'],selected['controls'],selected['references'])
            self.charged+=selected['charge']
            # Display the same feedback policy and fallback used in verification.
            state=x.copy();states=[state.copy()]
            for j in range(int(round(selected['certificate']['horizon']/DT))):
                u=self.committed.action(state,t+j*DT,self.model.track)
                state,_,_,_=plant_step(state,belief.center,u,self.model.track.curvature,self.model.track.ds)
                states.append(state.copy())
            event.update(status='committed',selected=selected,plan=self.model.track.pose(np.array(states))[::3].tolist())
        else:event.update(status='retained')
        event.update(policy=asdict(self.committed),budget_charged=self.charged,certification_checks=self.certifier.checks.copy())
        self.events.append(event)
