"""Integrated derivative-free SMID, Dual-Gatekeeper Eq.(3) and Lemma 1.
RK4 simulator accumulates integrals at the SAME stages as dynamics; no truth
passes into this interface. Exact state assumed in base runs; endpoint velocity
noise adds 2*noise to the residual. Regressor noise must be bounded separately.
"""
import numpy as np
from ..types import Interval

def intersect_regression(prior,regressor,output,residual_bound):
    lo,hi=prior.lower,prior.upper
    for f,y,eps in zip(np.ravel(regressor),np.ravel(output),np.broadcast_to(residual_bound,np.shape(output)).ravel()):
        if abs(f)<1e-12:
            if abs(y)>eps+1e-10:raise ValueError('Inconsistent zero-excitation observation')
            continue
        a,b=sorted(((y-eps)/f,(y+eps)/f));lo=max(lo,a);hi=min(hi,b)
    if lo>hi+1e-10:raise ValueError(f'Empty SMID intersection: {lo}, {hi}')
    result=Interval(lo,hi)
    assert prior.lower<=result.lower and result.upper<=prior.upper
    return result

def update_interval(prior,observations,disturbance_bound):
    result=prior
    for item in observations:
        # Numerical allowance is saved with data, independent of hidden truth.
        eps=disturbance_bound*item['duration']+item.get('integration_allowance',1e-8)+item.get('measurement_allowance',0.)
        result=intersect_regression(result,item['regressor'],item['output'],eps)
    return result
