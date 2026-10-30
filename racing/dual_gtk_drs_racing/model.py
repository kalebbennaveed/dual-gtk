"""Appendix C dynamic bicycle model in equivalent Frenet coordinates.

State: s, e_y, e_psi, v_x, v_y, yaw_rate, steering.
Inputs: drive force [N], brake force [N, negative], steering rate [rad/s].
The reference discrete system is explicit Euler, dt=0.05 s. Its vector field
and one-step map are affine in the *fixed* friction parameter mu.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .expressions import Expr, evaluate


@dataclass(frozen=True)
class Vehicle:
    mass: float = 720.
    inertia: float = 900.
    lf: float = 1.45
    lr: float = 1.60
    B: float = 5.
    C: float = 2.28
    rolling: float = .012
    steer_limit: float = .32
    steer_rate: float = .65
    # Circumscribed disk encloses the source URDF, including wheels.
    radius: float = 2.3


def build_graph(dt=.05, p=Vehicle(), gains=(.012,.45,.025,.18)):
    Expr.nodes = []
    q = [Expr(1, a=i) for i in range(11)]
    s,e,psi,vx,vy,r,delta,mu,speed,offset,offset_rate = q
    k = s.curvature()
    kl = (s + vx*.65).curvature()
    ke,kpsi,kvy,kr=gains
    # Lateral-error feedback must soften with speed. The original constant
    # gains destabilize the discrete steering loop above about 32 m/s.
    # This schedule is shared by the plant and every verified controller.
    scale=(24./vx).clip(.3,1.)
    lateral_gain=scale*scale
    desired = ((p.lf+p.lr)*kl).atan() - ke*lateral_gain*(e-offset) - kpsi*psi - kvy*lateral_gain*(vy-offset_rate) + kr*(vx*k-r)
    desired = desired.clip(-p.steer_limit,p.steer_limit)
    steer_rate = (3.*(desired-delta)).clip(-p.steer_rate,p.steer_rate)
    force = p.mass*((speed-vx)*1.3).clip(-6.,3.5)+p.rolling*p.mass*9.81
    fd,fb = force.clip(0.,p.mass*4),force.clip(-p.mass*7,0.)
    fzf = .5*p.mass*9.81*p.lr/(p.lf+p.lr)
    fzr = .5*p.mass*9.81*p.lf/(p.lf+p.lr)
    fxf = .25*fb-p.rolling*fzf
    fxr = .5*fd+.25*fb-p.rolling*fzr
    af = delta-((p.lf*r+vy)/(vx+1e-3)).atan()
    ar = ((p.lr*r-vy)/(vx+1e-3)).atan()
    fyf = fzf*(p.C*(p.B*af).atan()).sin()
    fyr = fzr*(p.C*(p.B*ar).atan()).sin()
    phi = [0.,0.,0.,-2/p.mass*fyf*delta.sin(),
           2/p.mass*(fyr+fyf*delta.cos()),
           2/p.inertia*(-p.lr*fyr+p.lf*fyf*delta.cos()),0.]
    sd = (vx*psi.cos()-vy*psi.sin())/(1-e*k)
    rhs0 = [sd,vx*psi.sin()+vy*psi.cos(),r-k*sd,
            2/p.mass*(fxr+fxf*delta.cos())+r*vy,
            2/p.mass*fxf*delta.sin()-r*vx,
            2*p.lf/p.inertia*fxf*delta.sin(),steer_rate]
    rhs = [a+b*mu for a,b in zip(rhs0,phi)]
    out = [q[i]+dt*rhs[i] for i in range(7)]+[fd,fb,steer_rate]
    out += [Expr.cast(z) for z in phi]+[Expr.cast(z) for z in rhs0]
    # Combined longitudinal/lateral tire-force feasibility (per wheel).
    # Kept as an optional safety constraint so the reference affine dynamics
    # remain unchanged. An infeasible force request is rejected, not simulated
    # with an invented saturation model.
    out += [mu*mu-(fxf/fzf)*(fxf/fzf)-(mu*fyf/fzf)*(mu*fyf/fzf),
            mu*mu-(fxr/fzr)*(fxr/fzr)-(mu*fyr/fzr)*(mu*fyr/fzr)]
    return np.asarray(Expr.nodes,dtype=float),np.asarray([a.i for a in out],dtype=np.int64)


GRAPH, OUTPUTS = build_graph()


@njit(cache=True)
def commands(age, duration, speed, amplitude, backup_speed):
    if age >= duration-1e-9: return backup_speed,0.,0.
    # Smooth excursion; amplitude is optimized against information and cost.
    phase = np.pi*age/max(duration,1e-9)
    return speed,amplitude*np.sin(phase)**2,amplitude*np.pi/duration*np.sin(2*phase)


@njit(cache=True)
def rollout(x0, mu, speed, amplitude, duration, total, k, ds, backup_speed=8., dt=.05):
    n=int(round(total/dt))
    xs=np.empty((n+1,7)); us=np.empty((n,3));phis=np.empty((n,7))
    xs[0]=x0
    for i in range(n):
        sp,off,vel=commands(i*dt,duration,speed,amplitude,backup_speed)
        inputs=np.empty(11);inputs[:7]=xs[i];inputs[7:]=np.array([mu,sp,off,vel])
        out=evaluate(inputs,GRAPH,OUTPUTS,k,ds)
        xs[i+1],us[i],phis[i]=out[:7],out[7:10],out[10:17]
    return xs,us,phis


class Model:
    def __init__(self, track, dt=.05):
        if dt != .05: raise ValueError("The certified discrete model uses dt=0.05 s.")
        self.track,self.dt,self.vehicle=track,dt,Vehicle()

    def step(self,x,mu,speed,offset=0.,offset_rate=0.):
        out=evaluate(np.r_[x,mu,speed,offset,offset_rate],GRAPH,OUTPUTS,self.track.curvature,self.track.ds)
        return out[:7],out[7:10],out[10:17],out[17:24]

    def rollout(self,x,mu,speed,amplitude,duration,backup_duration=3.):
        return rollout(x,mu,speed,amplitude,duration,duration+backup_duration,self.track.curvature,self.track.ds)

    def traction_margin(self,x,mu,speed,offset=0.,offset_rate=0.):
        return float(evaluate(np.r_[x,mu,speed,offset,offset_rate],GRAPH,OUTPUTS[24:26],self.track.curvature,self.track.ds).min())
