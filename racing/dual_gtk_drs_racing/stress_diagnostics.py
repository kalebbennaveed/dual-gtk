"""Independent classification of physical and terminal rollout failures."""
import math
import numpy as np
from numba import njit
from .fast_model import step_and_tire
from .safety import scalar_margin,terminal_margin


@njit(cache=True)
def rollout_outcomes(x0,mus,refs,k,ds):
    # Codes: 0 safe; 1 physical state; 2 tire-force; 3 terminal only; 4 nonfinite.
    times=np.full(len(mus),np.nan);kinds=np.zeros(len(mus),dtype=np.int64)
    km=np.max(np.abs(k))
    for i in range(len(mus)):
        x=(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6])
        for j in range(len(refs)):
            if not all_finite(x):times[i]=j*.05;kinds[i]=4;break
            if scalar_margin(x,km)<=0:times[i]=j*.05;kinds[i]=1;break
            sp,off,vel=refs[j];z=step_and_tire(x,mus[i],sp,off,vel,k,ds)
            if min(z[7],z[8])<=0:times[i]=j*.05;kinds[i]=2;break
            x=(z[0],z[1],z[2],z[3],z[4],z[5],z[6])
        if kinds[i]==0:
            if not all_finite(x):times[i]=len(refs)*.05;kinds[i]=4
            elif scalar_margin(x,km)<=0:times[i]=len(refs)*.05;kinds[i]=1
            elif terminal_margin(x,k,ds)<=0:times[i]=len(refs)*.05;kinds[i]=3
    return times,kinds


@njit(cache=True)
def all_finite(x):
    for v in x:
        if not math.isfinite(v):return False
    return True
