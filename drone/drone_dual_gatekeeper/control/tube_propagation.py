"""DTMPC (13),(14),(29)-(31), fixed alpha. Position error satisfies
rho_dot=-lambda*rho+Phi; |e_v|<=Phi+lambda*rho.
A Bernstein majorant of nominal drag forcing makes propagation conservative
between nodes. Exact positive LTI propagation; tube is NEVER a decision variable.
"""
from dataclasses import dataclass
import numpy as np

def delta_bound(interval,estimate,velocity):
    delta=max(abs(interval.lower-estimate),abs(interval.upper-estimate))
    v=np.asarray(velocity)
    return delta*np.linalg.norm(v,axis=-1,keepdims=True)*np.abs(v)

def propagate(phi,rho,forcing,h,alpha,lam,exp=np.exp):
    ea=exp(-alpha*h); el=exp(-lam*h); steady=forcing/alpha
    phi1=steady+(phi-steady)*ea
    if abs(alpha-lam)<1e-12:
        rho1=rho*el+steady*(1-el)/lam+(phi-steady)*h*el
    else:
        rho1=rho*el+steady*(1-el)/lam+(phi-steady)*(ea-el)/(lam-alpha)
    return phi1,rho1

@dataclass
class DynamicTube:
    edges: np.ndarray
    drag_forcing: np.ndarray
    boundary_layer: np.ndarray
    position_radius: np.ndarray
    alpha: float
    sliding_lambda: float
    disturbance_bound: float
    eta: float
    def at(self,t):
        i=min(np.searchsorted(self.edges,t,side='right')-1,len(self.drag_forcing)-1)
        i=max(i,0)
        h=max(0,min(t,self.edges[-1])-self.edges[i])
        return propagate(self.boundary_layer[i],self.position_radius[i],
                         self.drag_forcing[i]+self.disturbance_bound+self.eta,
                         h,self.alpha,self.sliding_lambda)
    def forcing_at(self,t):
        i=min(max(np.searchsorted(self.edges,t,side='right')-1,0),len(self.drag_forcing)-1)
        return self.drag_forcing[i]
    def error_bounds(self,t):
        phi,rho=self.at(t)
        return np.r_[rho,phi+self.sliding_lambda*rho]
    def utilization(self,error,t):
        phi,rho=self.at(t)
        s=error[3:]+self.sliding_lambda*error[:3]
        # At t=0, zero radius means exact initial position.
        rratio=np.divide(np.abs(error[:3]),rho,out=np.zeros(3),where=rho>1e-14)
        if np.any((rho<=1e-14)&(np.abs(error[:3])>1e-8)): return float('inf')
        return max(np.max(rratio),np.max(np.abs(s)/phi))
    def cell_envelope(self,i):
        h=self.edges[i+1]-self.edges[i]
        p=np.maximum(self.boundary_layer[i],self.boundary_layer[i+1])
        r0=self.position_radius[i]
        r=np.maximum(r0,r0*np.exp(-self.sliding_lambda*h)+p*(-np.expm1(-self.sliding_lambda*h))/self.sliding_lambda)
        return p,r,p+self.sliding_lambda*r

def build_tube(curve,interval,config,phi0=None,rho0=None):
    estimate=interval.midpoint; phi=config.initial_boundary_layer if phi0 is None else np.asarray(phi0)
    rho=np.zeros(3) if rho0 is None else np.asarray(rho0)
    ps=[phi.copy()]; rs=[rho.copy()]; ds=[]
    for i in range(len(curve.edges)-1):
        # Use precisely the optimizer's smooth conservative velocity hull.
        # Otherwise reconnection compares differently propagated tubes and a
        # zero-motion axis can spuriously make every candidate infeasible.
        vc=curve.bernstein_coefficients(i,1)
        def maximum(a,b):
            return (a+b+np.sqrt((a-b)**2+config.smooth_max_width**2))/2
        def maxall(values):
            result=values[0]
            for value in values[1:]:result=maximum(result,value)
            return result
        lo=-np.array([maxall(-vc[:,j]) for j in range(3)])
        hi=np.array([maxall(vc[:,j]) for j in range(3)])
        vmax=maximum(np.sqrt(lo*lo+config.smooth_abs_width**2),
                     np.sqrt(hi*hi+config.smooth_abs_width**2))
        drag=delta_bound(interval,estimate,vmax); ds.append(drag)
        h=(curve.edges[i+1]-curve.edges[i])*curve.horizon
        phi,rho=propagate(phi,rho,drag+config.disturbance_bound+config.eta,h,config.alpha,config.sliding_lambda)
        ps.append(phi);rs.append(rho)
    return DynamicTube(curve.edges*curve.horizon,np.array(ds),np.array(ps),np.array(rs),config.alpha,
                       config.sliding_lambda,config.disturbance_bound,config.eta)
