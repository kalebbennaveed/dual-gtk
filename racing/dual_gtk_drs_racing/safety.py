"""Sampled DRS gatekeeper, beta=0 (nominal probabilistic specialization).

The backup set is assumed robustly invariant (alpha=0). A fixed N IID
friction sample batch is drawn per candidate; each sample is constant over
its rollout. Bonferroni allocation across a fixed maximum of candidate
checks protects adaptive selection with fresh independent batches.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from numba import njit
from scipy.stats import beta as beta_distribution
from .expressions import lookup, evaluate
from .model import GRAPH, OUTPUTS, commands
from .fast_model import step_and_tire

@njit(cache=True)
def margins(c,rad,k,ds,terminal=False,kmax=-1.,k_lipschitz=-1.):
    # The track is its Frenet ribbon, eroded by a circumscribed vehicle disk.
    h=min(5.7-abs(c[1])-rad[1], .65-abs(c[2])-rad[2],
          c[3]-rad[3]-2.,65.-c[3]-rad[3], .32-abs(c[6])-rad[6])
    # Keep the Frenet chart regular throughout the enclosure.
    if kmax<0: kmax=max(abs(k.min()),abs(k.max()))
    h=min(h,1.-(abs(c[1])+rad[1])*kmax-.2)
    if terminal:
        if k_lipschitz<0: k_lipschitz=np.max(np.abs(np.diff(k)))/ds
        h=min(h,1.5-abs(c[1])-rad[1], .12-abs(c[2])-rad[2],
              1.2-abs(c[3]-8.)-rad[3], 1.-abs(c[4])-rad[4],
              .20-abs(c[5]-8.*lookup(c[0],k,ds))-rad[5]-8*k_lipschitz*rad[0])
    return h



@njit(cache=True, inline="always")
def scalar_margin(x, kmax):
    return min(5.7-abs(x[1]), .65-abs(x[2]), x[3]-2.,
               65.-x[3], .32-abs(x[6]), .8-abs(x[1])*kmax)


@njit(cache=True, inline="always")
def terminal_margin(x, k, ds):
    return min(1.5-abs(x[1]), .12-abs(x[2]), 1.2-abs(x[3]-8.),
               1.-abs(x[4]), .20-abs(x[5]-8.*lookup(x[0],k,ds)))


@njit(cache=True)
def sampled_rollouts(x0, mus, references, k, ds, kmax, traction, allowed_failures):
    """Fixed-N design; early rejection only. Never accept a partial batch."""
    failures=0; minimum=1e100; steps=0
    for i in range(len(mus)):
        x=(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6])
        h=scalar_margin(x,kmax)
        for j in range(len(references)):
            if h<=0:break
            sp,off,vel=references[j]
            z=step_and_tire(x,mus[i],sp,off,vel,k,ds)
            steps+=1
            if traction:h=min(h,z[7],z[8])
            x=(z[0],z[1],z[2],z[3],z[4],z[5],z[6])
            finite=True
            for v in x:
                if not math.isfinite(v):finite=False
            if not finite:
                h=-1e100;break
            h=min(h,scalar_margin(x,kmax))
        h=min(h,terminal_margin(x,k,ds))
        minimum=min(minimum,h)
        failures+=int(h<=0)
        if failures>allowed_failures:
            return failures,i+1,minimum,steps
    return failures,len(mus),minimum,steps


def binomial_upper(failures, samples, delta):
    if not 0<delta<1 or samples<1 or not 0<=failures<=samples:
        raise ValueError("Invalid binomial certificate parameters")
    if failures==samples:return 1.
    return float(beta_distribution.isf(delta,failures+1,samples-failures))


@dataclass
class Certificate:
    accepted: bool
    failure_upper: float
    margin: float
    cells: int
    reason: str
    support: tuple[float,float]
    horizon: float = 0.
    samples: int = 0
    evaluated: int = 0
    failures: int = 0
    epsilon: float = .01
    delta: float = .05
    candidate_delta: float = .05/64
    candidate_limit: int = 64
    beta: float = 0.
    alpha: float = 0.
    sample_seed: int = 0
    rollout_steps: int = 0
    # A partial batch may reject, but has no completed binomial certificate.
    complete_batch: bool = False


class DRSCertifier:
    def __init__(self,track,max_cells=None,backup_duration=10.,traction=False,
                 epsilon=.01,delta=.05,samples=1000,candidate_limit=64,seed=7):
        if not 0<epsilon<1 or not 0<delta<1 or samples<1 or candidate_limit<1:
            raise ValueError("Invalid sampled DRS configuration")
        if backup_duration<=0:raise ValueError("backup_duration must be positive")
        self.track,self.backup_duration,self.traction=track,backup_duration,traction
        self.epsilon,self.delta,self.samples=epsilon,delta,int(samples)
        self.candidate_limit,self.candidate_delta=int(candidate_limit),delta/candidate_limit
        self.seed=int(seed);self.kmax=float(np.max(np.abs(track.curvature)))
        self.upper=np.array([binomial_upper(j,self.samples,self.candidate_delta) for j in range(self.samples+1)])
        valid=np.flatnonzero(self.upper<=epsilon)
        self.allowed_failures=int(valid[-1]) if len(valid) else -1
        self.begin_decision(0.)

    def begin_decision(self,t):
        self.decision_index=int(round(t/.05));self.checks=[];self.cache={}

    def tail_duration(self,x,speed,duration):
        if not self.traction:return self.backup_duration
        peak=max(x[3],min(speed,x[3]+3.5*duration))
        return min(self.backup_duration,max(3.5,2.+(peak-8.)/6.))

    def check(self,x,belief,speed,amplitude,duration):
        key=(float(speed),float(amplitude),float(duration))
        # Cache only an identical candidate at the current state/support/decision.
        key+=(tuple(x),belief.lower,belief.upper)
        if key in self.cache:return self.cache[key]
        slot=len(self.checks)
        if slot>=self.candidate_limit:raise RuntimeError("Candidate confidence budget exceeded")
        tail=np.ceil(self.tail_duration(x,speed,duration)/.05-1e-9)*.05
        horizon=float(duration+tail)
        seed=int(np.random.SeedSequence([self.seed,self.decision_index,slot]).generate_state(1)[0])
        failures=evaluated=steps=0;minimum=-1e100
        if np.all(np.isfinite(x)) and self.allowed_failures>=0:
            rng=np.random.default_rng(seed)
            mus=rng.uniform(belief.lower,belief.upper,self.samples)
            refs=np.array([commands(j*.05,duration,speed,amplitude,8.) for j in range(int(round(horizon/.05)))])
            failures,evaluated,minimum,steps=sampled_rollouts(x,mus,refs,self.track.curvature,
                self.track.ds,self.kmax,self.traction,self.allowed_failures)
        complete=evaluated==self.samples
        upper=float(self.upper[failures]) if complete else 1.
        accepted=complete and upper<=self.epsilon
        reason="sampled_drs_accepted" if accepted else ("insufficient_sample_budget" if self.allowed_failures<0 else "sample_failure_limit")
        cert=Certificate(accepted,upper,float(minimum),0,reason,(belief.lower,belief.upper),horizon,
            self.samples,int(evaluated),int(failures),self.epsilon,self.delta,self.candidate_delta,
            self.candidate_limit,0.,0.,seed,int(steps),bool(complete))
        self.checks.append(dict(speed=float(speed),amplitude=float(amplitude),duration=float(duration),**asdict(cert)))
        self.cache[key]=cert
        return cert

@njit(cache=True)
def point_margin(xs,k,ds):
    result=1e9
    kmax=max(abs(k.min()),abs(k.max()))
    k_lipschitz=np.max(np.abs(np.diff(k)))/ds
    for i in range(len(xs)):
        if not np.all(np.isfinite(xs[i])): return -np.inf
        result=min(result,margins(xs[i],np.zeros(7),k,ds,i==len(xs)-1,kmax,k_lipschitz))
    return result


@njit(cache=True)
def traction_point_margin(xs,mu,speed,amplitude,duration,k,ds):
    result=1e9
    for j in range(len(xs)-1):
        sp,off,vel=commands(j*.05,duration,speed,amplitude,8.)
        inp=np.empty(11);inp[:7]=xs[j];inp[7:]=np.array([mu,sp,off,vel])
        h=evaluate(inp,GRAPH,OUTPUTS[24:26],k,ds).min()
        if not np.isfinite(h):return -np.inf
        result=min(result,h)
    return result
