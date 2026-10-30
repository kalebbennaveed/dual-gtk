"""Local MPPI: Gaussian control-sequence rollouts and exponential weighting.

Algorithm reference: supplied car-racing build_mppi/mppi_update. This module
has no imports from another racing repository. Force uses a signed coordinate
to exclude simultaneous drive and brake; output is three physical controls.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
from numba import njit
from .dynamics import DT, plant_step, fallback_control, running_cost, rollout_controls
from .expressions import lookup
from .safety import scalar_margin

@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 60
    samples: int = 384
    iterations: int = 4
    temperature: float = .35
    force_noise: float = 500.
    steering_noise: float = .055
    noise_correlation: float = .8
    force_slew: float = 150.
    steering_slew: float = .12
    information_regularizer: float = 1.
    selection_discount: float = .5
    budget: float = 8.
    replan_steps: int = 10
    candidate_steps: tuple = (6,12,24,36,48,60)

@njit(cache=True)
def physical_controls(v):
    out=np.zeros((len(v),3))
    for j in range(len(v)):
        out[j,0]=max(0.,v[j,0]);out[j,1]=min(0.,v[j,0]);out[j,2]=v[j,1]
    return out

@njit(cache=True)
def project(v, previous, force_slew, steering_slew):
    out=v.copy();last_force=previous[0]+previous[1];last_rate=previous[2]
    for j in range(len(v)):
        out[j,0]=min(2880.,max(-5040.,min(last_force+force_slew,max(last_force-force_slew,v[j,0]))))
        out[j,1]=min(.65,max(-.65,min(last_rate+steering_slew,max(last_rate-steering_slew,v[j,1]))))
        last_force=out[j,0];last_rate=out[j,1]
    return out

@njit(cache=True)
def warm_start(x,mu,horizon,k,ds,target,previous,force_slew,steering_slew):
    v=np.empty((horizon,2));state=x.copy();last=previous.copy()
    for j in range(horizon):
        # Curvature preview only initializes optimization. Every returned input
        # is subsequently a decision variable in MPPI.
        limit=target
        for d in range(1,16):
            distance=d*max(2.,state[3]*.18)
            curve=abs(lookup(state[0]+distance,k,ds))
            corner=np.sqrt(max(.1,.65*mu*9.81)/max(curve,1e-4))
            limit=min(limit,np.sqrt(corner*corner+2.*3.*distance))
        u=fallback_control(state,k,ds,limit)
        # Nominal traction-aware initial guess, evaluated at the estimate only.
        u[0]=min(u[0],.85*2.*1678.9573770491806*mu)
        raw=np.array([[u[0]+u[1],u[2]]])
        row=project(raw,last,force_slew,steering_slew)[0]
        v[j]=row;last=physical_controls(row.reshape(1,2))[0]
        state,_,_,_=plant_step(state,mu,last,k,ds)
    return v

@njit(cache=True)
def objective(x,mu,v,k,ds,kmax,target,gamma,regularizer,sensor,traction,previous):
    state=x.copy();cost=0.;gramian=0.;last=previous.copy()
    for j in range(len(v)):
        if not np.all(np.isfinite(state)) or scalar_margin(state,kmax)<=0:
            return 1e8+(len(v)-j)*1e5
        u=np.array([max(0.,v[j,0]),min(0.,v[j,0]),v[j,1]])
        nxt,phi,_,tire=plant_step(state,mu,u,k,ds)
        cost+=DT*running_cost(state,u,target)
        if traction:
            cost+=2e4*max(0.,-min(tire[0],tire[1]))**2+1e3*max(0.,-min(tire[0],tire[1]))
        if sensor==1:gramian+=DT*(phi[5]/.18)**2
        else:gramian+=DT*((phi[3]/.6)**2+(phi[4]/.6)**2+(phi[5]/.18)**2)
        state=nxt;last=u
    if not np.all(np.isfinite(state)) or scalar_margin(state,kmax)<=0:return 1e8
    # Terminal mission cost is zero in this instantiation. Reported mission
    # cost, paired budget cost and optimizer running cost use the same ell.
    return cost-gamma*np.log(regularizer+gramian)

@njit(cache=True)
def batch_cost(x,mu,plans,k,ds,kmax,target,gamma,regularizer,sensor,traction,previous):
    costs=np.empty(len(plans))
    for i in range(len(plans)):
        costs[i]=objective(x,mu,plans[i],k,ds,kmax,target,gamma,regularizer,sensor,traction,previous)
    return costs

class MPPI:
    def __init__(self,track,target,sensor,traction,seed=0,config=None):
        self.config=config or MPPIConfig();self.track=track;self.target=target
        self.sensor=int(sensor=='quantized_yaw');self.traction=traction
        self.rng=np.random.default_rng(seed);self.plan=None;self.last_time=None
        self.kmax=float(np.max(np.abs(track.curvature)))

    def optimize(self,x,mu,t,previous,gamma=0.,initial=None):
        cfg=self.config;tr=self.track;begin=time.perf_counter()
        args=(tr.curvature,tr.ds,self.kmax,self.target,gamma,cfg.information_regularizer,self.sensor,self.traction,previous)
        warm=warm_start(x,mu,cfg.horizon,tr.curvature,tr.ds,self.target,previous,cfg.force_slew,cfg.steering_slew)
        base=warm
        alternatives=[]
        if self.plan is not None:
            shift=min(cfg.horizon,max(0,int(round((t-self.last_time)/DT))))
            alternatives.append(np.concatenate((self.plan[shift:],warm[-shift:])) if shift else self.plan.copy())
        if initial is not None:alternatives.append(initial.copy())
        best_cost=objective(x,mu,base,*args)
        for a in alternatives:
            a=project(a,previous,cfg.force_slew,cfg.steering_slew);c=objective(x,mu,a,*args)
            if c<best_cost:base=a;best_cost=c
        initial_cost=float(best_cost);history=[initial_cost]
        sigma=np.array([cfg.force_noise,cfg.steering_noise])
        for iteration in range(cfg.iterations):
            noise=self.rng.normal(size=(cfg.samples,cfg.horizon,2))*sigma
            for j in range(1,cfg.horizon):
                noise[:,j]=cfg.noise_correlation*noise[:,j-1]+np.sqrt(1-cfg.noise_correlation**2)*noise[:,j]
            samples=np.array([project(base+n,previous,cfg.force_slew,cfg.steering_slew) for n in noise])
            samples[0]=base
            costs=batch_cost(x,mu,samples,*args)
            weights=np.exp(-(costs-costs.min())/cfg.temperature);weights/=weights.sum()
            update=project(np.sum(samples*weights[:,None,None],axis=0),previous,cfg.force_slew,cfg.steering_slew)
            update_cost=objective(x,mu,update,*args)
            # Retain an incumbent if finite-sample averaging worsens the objective.
            if update_cost<best_cost:base=update;best_cost=update_cost
            idx=int(np.argmin(costs))
            if costs[idx]<best_cost:base=samples[idx].copy();best_cost=costs[idx]
            history.append(float(best_cost))
        self.plan=base.copy();self.last_time=t
        controls=physical_controls(base);states,phis=rollout_controls(x,mu,controls,tr.curvature,tr.ds)
        return dict(controls=controls,states=states,phis=phis,coordinates=base,
                    diagnostics=dict(gamma=gamma,initial_objective=initial_cost,objective=float(best_cost),
                        iteration_costs=history,samples=cfg.samples,iterations=cfg.iterations,
                        solve_seconds=time.perf_counter()-begin,config=asdict(cfg)))
