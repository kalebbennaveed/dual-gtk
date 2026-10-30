"""Explicit-input bicycle plant and fixed feedback policies.

Physical parameters/discretization match model.py exactly. Controls are
[drive force (N), negative brake force (N), steering rate (rad/s)].
"""
import math
import numpy as np
from numba import njit
from .expressions import lookup

DT = .05

@njit(cache=True)
def bounded_control(u):
    return np.array([min(2880., max(0., u[0])), min(0., max(-5040., u[1])),
                     min(.65, max(-.65, u[2]))])

@njit(cache=True)
def fallback_control(x, k, ds, speed=8., offset=0., offset_rate=0.):
    s,e,psi,vx,vy,r,delta=x
    scale=min(1.,max(.3,24./max(vx,1e-3)))**2
    desired=(math.atan(3.05*lookup(s+vx*.65,k,ds))-.012*scale*(e-offset)
             -.45*psi-.025*scale*(vy-offset_rate)+.18*(vx*lookup(s,k,ds)-r))
    desired=min(.32,max(-.32,desired))
    force=720.*min(3.5,max(-6.,1.3*(speed-vx)))+84.7584
    return bounded_control(np.array([max(0.,force),min(0.,force),3.*(desired-delta)]))

@njit(cache=True)
def tracking_control(x, reference, u):
    """Fixed feedback about the optimized state/input; shared by verifier/plant."""
    scale=min(1.,max(.3,24./max(x[3],1e-3)))**2
    out=u.copy()
    force=u[0]+u[1]+720.*1.3*(reference[3]-x[3])
    out[0]=max(0.,force);out[1]=min(0.,force)
    out[2]+=3.*(-.012*scale*(x[1]-reference[1])-.45*(x[2]-reference[2])
                -.025*scale*(x[4]-reference[4])-.18*(x[5]-reference[5])-(x[6]-reference[6]))
    return bounded_control(out)

@njit(cache=True)
def plant_step(x, mu, u, k, ds):
    """No hidden controller or tire clipping. Return next state, regressor, f0, tire margins."""
    s,e,psi,vx,vy,r,delta=x
    fd,fb,steer_rate=u
    fzf=1852.6426229508202;fzr=1678.9573770491806
    fxf=.25*fb-.012*fzf;fxr=.5*fd+.25*fb-.012*fzr
    af=delta-math.atan((1.45*r+vy)/(vx+.001))
    ar=math.atan((1.6*r-vy)/(vx+.001))
    fyf=fzf*math.sin(2.28*math.atan(5.*af))
    fyr=fzr*math.sin(2.28*math.atan(5.*ar))
    curvature=lookup(s,k,ds)
    progress=(vx*math.cos(psi)-vy*math.sin(psi))/(1.-e*curvature)
    phi=np.array([0.,0.,0.,-2./720.*fyf*math.sin(delta),
                  2./720.*(fyr+fyf*math.cos(delta)),
                  2./900.*(-1.6*fyr+1.45*fyf*math.cos(delta)),0.])
    f0=np.array([progress,vx*math.sin(psi)+vy*math.cos(psi),r-curvature*progress,
                 2./720.*(fxr+fxf*math.cos(delta))+r*vy,
                 2./720.*fxf*math.sin(delta)-r*vx,
                 2.*1.45/900.*fxf*math.sin(delta),steer_rate])
    tire=np.array([mu*mu-(fxf/fzf)**2-(mu*fyf/fzf)**2,
                   mu*mu-(fxr/fzr)**2-(mu*fyr/fzr)**2])
    return x+DT*(f0+mu*phi),phi,f0,tire

@njit(cache=True)
def running_cost(x,u,target):
    return 1.+.004*(target-x[3])**2+.08*x[1]**2+.2*x[2]**2+.002*((u[0]+u[1])/720.)**2+.08*u[2]**2

@njit(cache=True)
def rollout_controls(x,mu,controls,k,ds):
    xs=np.empty((len(controls)+1,7));phis=np.empty((len(controls),7));xs[0]=x
    for j in range(len(controls)):
        xs[j+1],phis[j],_,_=plant_step(xs[j],mu,controls[j],k,ds)
    return xs,phis
