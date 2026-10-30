"""DTMPC (29)-(31) with interval radius replacing Cbar and cell-majorized Delta.
K=max(0, Delta(actual)-Delta_cell+alpha*Phi) obeys the lower bound (9),
including when the equality gain would be negative. sat acts on s/Phi, not u.
"""
import numpy as np
from .tube_propagation import delta_bound

def ancillary_control(state,nominal,phi,drag_forcing,interval,config):
    er=state[:3]-nominal[:3]; ev=state[3:]-nominal[3:]
    s=ev+config.sliding_lambda*er
    va=state[3:];vn=nominal[3:]
    gain=np.maximum(0,delta_bound(interval,interval.midpoint,va)-drag_forcing+config.alpha*phi)
    return (interval.midpoint*(np.linalg.norm(va)*va-np.linalg.norm(vn)*vn)
            -config.sliding_lambda*ev-gain*np.clip(s/phi,-1,1))
