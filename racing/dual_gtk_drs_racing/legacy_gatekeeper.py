"""Mission/information candidates, paired safety checks, budget, and commitment."""
from dataclasses import dataclass,asdict
import numpy as np
from .belief import predicted_width,excitation_score,predicted_quantized_width
from .safety import DRSCertifier
from .model import commands
from .scenarios import get_scenario


@dataclass
class Policy:
    kind: str
    start: float
    duration: float
    speed: float
    amplitude: float = 0.

    def action_reference(self,t):
        return commands(max(0.,t-self.start),self.duration,self.speed,self.amplitude,8.)


def stage_cost(xs,us,target_speed=24.):
    return 1.+.004*(target_speed-xs[:-1,3])**2+.08*xs[:-1,1]**2+.2*xs[:-1,2]**2+ .002*((us[:,0]+us[:,1])/720.)**2+.08*us[:,2]**2


class Gatekeeper:
    def __init__(self,model,method="proposed",budget=8.,max_cells=256,scenario="barcelona",information_weight=4.,seed=7):
        if method not in ("proposed","passive","weighted","robust"):
            raise ValueError("Unknown method")
        self.model,self.method=model,method
        self.scenario=get_scenario(scenario)
        self.certifier=DRSCertifier(model.track,max_cells=max_cells,backup_duration=self.scenario.backup_duration,traction=self.scenario.traction,
            epsilon=self.scenario.epsilon,delta=self.scenario.delta,samples=self.scenario.samples,
            candidate_limit=self.scenario.candidate_limit,seed=seed)
        self.committed=Policy("backup",0.,0.,8.)
        self.budget=float(budget);self.charged=0.
        if information_weight<0:raise ValueError("information_weight must be nonnegative")
        self.information_weight=float(information_weight)
        self.events=[]

    def candidate(self,x,belief,speed,amplitude,duration,t):
        tail=np.ceil(self.certifier.tail_duration(x,speed,duration)/.05-1e-9)*.05
        xs,us,phi=self.model.rollout(x,belief.center,speed,amplitude,duration,tail)
        n=int(round(duration/.05))
        cost=float(.05*stage_cost(xs[:n+1],us[:n],self.scenario.racing_speed).sum())
        remaining=(predicted_quantized_width(belief,xs[:n+1],phi[:n]) if self.scenario.sensor=="quantized_yaw"
                   else predicted_width(belief.width,phi[:n],self.scenario.sensor))
        gain=belief.width-remaining
        excitation=excitation_score(phi[:n],self.scenario.sensor)
        progress=float(xs[n,0]-x[0])
        return dict(speed=float(speed),amplitude=float(amplitude),duration=float(duration),
                    cost=cost,gain=gain,excitation=excitation,progress=progress,xs=xs)

    def replan(self,x,belief,t):
        self.certifier.begin_decision(t)
        if self.method=="weighted":
            self.replan_weighted(x,belief,t)
            return
        # Finish an accepted informative segment; its phase and controls are
        # immutable. Measurements still update the support on every plant step.
        if self.committed.kind=="informative" and t<self.committed.start+self.committed.duration-1e-8:
            return
        tr=self.model.track
        ahead=x[0]+np.linspace(0,max(35.,x[3]*2.5),50)
        maxk=max(.0001,float(np.max(np.abs(tr.kappa(ahead)))))
        desired=float(min(self.scenario.racing_speed,np.sqrt(7.0/maxk)))
        candidates=[];pairs=[];attempts=[]
        for duration in (.6,1.2,1.8):
            # Finite policy-space MPC: optimize progress and tracking cost at the
            # current point estimate, then certify using an independent IID rollout batch.
            speeds=(desired,) if self.scenario.traction else (desired,.85*desired,max(8.,.7*desired))
            options=[self.candidate(x,belief,v,0.,duration,t) for v in speeds]
            options.sort(key=lambda a: -(a["progress"]-.25*a["cost"])/duration)
            mission=None
            for a in options:
                cert=self.certifier.check(x,belief,a["speed"],0.,duration)
                a["certificate"]=asdict(cert)
                attempts.append(dict(speed=a["speed"],duration=duration,certificate=asdict(cert)))
                if cert.accepted:
                    mission=a;break
            if mission is not None:
                mission.update(kind="mission",charge=0.,score=0.)
                candidates.append(mission)
            if self.method!="proposed" or belief.width<=.003: continue
            if mission is None:
                if not self.scenario.traction: continue
                # When racing is infeasible, pair a slow steering probe with
                # the centerline backup itself. Both complete policies must
                # pass the same sampled probabilistic certificate.
                mission=self.candidate(x,belief,8.,0.,duration,t)
                cert=self.certifier.check(x,belief,8.,0.,duration)
                if not cert.accepted: continue
                mission["certificate"]=asdict(cert)
            # Deterministic shooting search for informative reference excursions.
            infos=[self.candidate(x,belief,mission["speed"],a,duration,t) for a in (-3.,-1.5,1.5,3.)]
            infos.sort(key=lambda a: -(a["gain"]-.08*max(0.,a["cost"]-mission["cost"])))
            for a in infos[:2]:
                charge=max(0.,a["cost"]-mission["cost"])
                extra_gain=max(0.,a["gain"]-mission["gain"])
                if extra_gain<=.003 or self.charged+charge>self.budget: continue
                cert=self.certifier.check(x,belief,a["speed"],a["amplitude"],duration)
                a["certificate"]=asdict(cert)
                if not cert.accepted: continue
                score=extra_gain*np.exp(-.5*duration)
                a.update(kind="informative",charge=charge,score=float(score))
                candidates.append(a)
                pairs.append(dict(duration=duration,speed=mission["speed"],mission=mission["certificate"],informative=asdict(cert)))
        informative=[a for a in candidates if a["kind"]=="informative" and a["score"]>0]
        if informative:
            selected=max(informative,key=lambda a:a["score"])
        else:
            missions=[a for a in candidates if a["kind"]=="mission"]
            selected=max(missions,key=lambda a:(a["progress"]-.25*a["cost"])/a["duration"],default=None)
        event=dict(time=float(t),support=[belief.lower,belief.upper],feasible=len(candidates),pairs=pairs,nominal_attempts=attempts,
                   previous=asdict(self.committed))
        if selected is not None:
            self.committed=Policy(selected["kind"],t,selected["duration"],selected["speed"],selected["amplitude"])
            # Charge the entire committed prefix up front. Preemption cannot
            # spend unreserved exploration budget or refund fictitious savings.
            self.charged+=selected["charge"]
            event.update(selected={k:v for k,v in selected.items() if k!="xs"},policy=asdict(self.committed),
                         plan=self.model.track.pose(selected["xs"])[::3].tolist(),status="committed")
        else:
            event.update(policy=asdict(self.committed),status="retained")
        event["budget_charged"]=self.charged
        event["certification_checks"]=self.certifier.checks.copy()
        self.events.append(event)

    def replan_weighted(self,x,belief,t):
        """Fixed-weight persistent excitation, optimized over verified policies.

        Search past rejected candidates so weighted control can learn too.
        There is no proposed-method budget, shrinkage cutoff or paired-policy
        selection rule. Progress and running cost remain in its objective.
        """
        tr=self.model.track
        ahead=x[0]+np.linspace(0,max(35.,x[3]*2.5),50)
        desired=float(min(self.scenario.racing_speed,np.sqrt(7./max(.0001,float(np.max(np.abs(tr.kappa(ahead))))))))
        attempts=[];options=[]
        for duration in (.6,1.2,1.8):
            # Same fast and slow policy primitives as the proposed planner.
            speeds=(desired,8.) if self.scenario.traction else (desired,.85*desired,max(8.,.7*desired),8.)
            options.extend(self.candidate(x,belief,v,a,duration,t) for v in speeds for a in (0.,-3.,-1.5,1.5,3.))
        for a in options:
            a["objective"]=(a["cost"]-.25*a["progress"])/a["duration"]-self.information_weight*a["excitation"]
        options.sort(key=lambda a:a["objective"])
        selected=None
        for a in options:
            duration=a["duration"]
            cert=self.certifier.check(x,belief,a["speed"],a["amplitude"],duration)
            kind="informative" if a["amplitude"] else ("backup" if a["speed"]<=8.+1e-9 else "mission")
            a.update(certificate=asdict(cert),kind=kind,charge=0.,score=-a["objective"])
            attempts.append({k:v for k,v in a.items() if k!="xs"})
            if cert.accepted:
                selected=a;break
        event=dict(time=float(t),support=[belief.lower,belief.upper],feasible=int(selected is not None),pairs=[],
                   weighted_attempts=attempts,information_weight=self.information_weight,
                   previous=asdict(self.committed),budget_charged=0.)
        if selected is not None:
            self.committed=Policy(selected["kind"],t,selected["duration"],selected["speed"],selected["amplitude"])
            event.update(selected={k:v for k,v in selected.items() if k!="xs"},policy=asdict(self.committed),
                         plan=tr.pose(selected["xs"])[::3].tolist(),status="committed")
        else:
            event.update(policy=asdict(self.committed),status="retained")
        event["certification_checks"]=self.certifier.checks.copy()
        self.events.append(event)
