"""Exploratory screen; no claim of IID trial statistics from this grid."""
from pathlib import Path
import json
import time
import numpy as np
from numba import njit
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.model import Model, commands
from dual_gtk_drs_racing.fast_model import step_and_tire
from dual_gtk_drs_racing.safety import scalar_margin, terminal_margin
from dual_gtk_drs_racing.legacy_gatekeeper import Gatekeeper
from dual_gtk_drs_racing.belief import FrictionSet


@njit(cache=True)
def classify(x0, mus, refs, k, ds):
    """First physical failure time/kind, independently of terminal membership."""
    times=np.full(len(mus),np.nan); kinds=np.zeros(len(mus),dtype=np.int64)
    for i in range(len(mus)):
        x=(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],x0[6]); km=np.max(np.abs(k))
        for j in range(len(refs)):
            if scalar_margin(x,km)<=0:
                times[i]=j*.05;kinds[i]=1;break
            sp,off,vel=refs[j]
            z=step_and_tire(x,mus[i],sp,off,vel,k,ds)
            if min(z[7],z[8])<=0:
                times[i]=j*.05;kinds[i]=2;break
            x=(z[0],z[1],z[2],z[3],z[4],z[5],z[6])
        if kinds[i]==0:
            if scalar_margin(x,km)<=0:times[i]=len(refs)*.05;kinds[i]=1
            elif terminal_margin(x,k,ds)<=0:times[i]=len(refs)*.05;kinds[i]=3
    return times,kinds


def main():
    tr=Track('technical');model=Model(tr)
    out=Path('output/stress-discovery');out.mkdir(parents=True,exist_ok=True)
    starts=np.linspace(0,tr.length,121)[:-1];mus=np.linspace(.45,1.05,1001)
    manifest=dict(kind='exploratory deterministic grid, not IID evidence',starts=starts.tolist(),mus=mus.tolist(),scenario='traction_relaxed',seed=7)
    (out/'grid-manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];begin=time.time()
    for i,s in enumerate(starts):
        x=np.array([s,0.,0.,8.,0.,8*tr.kappa(s),np.arctan(3.05*tr.kappa(s))])
        gk=Gatekeeper(model,'proposed',scenario='traction_relaxed',seed=7)
        gk.replan(x,FrictionSet(),0.)
        e=gk.events[-1]
        row=dict(start=float(s),status=e['status'])
        if e['status']=='committed':
            a=e['selected'];c=a['certificate'];p=gk.committed
            refs=np.array([p.action_reference(j*.05) for j in range(int(round(c['horizon']/.05)))])
            t,kind=classify(x,mus,refs,tr.curvature,tr.ds)
            prefix=(kind>0)&(kind<3)&(t<p.duration-1e-9)
            row.update(kind=a['kind'],amplitude=a['amplitude'],duration=a['duration'],speed=a['speed'],
                certificate=c,prefix_unsafe_mu=mus[prefix].tolist(),full_unsafe_mu=mus[kind>0].tolist(),
                physical_failures=int(((kind>0)&(kind<3)).sum()),terminal_failures=int((kind==3).sum()))
            if prefix.any():print('PREFIX FAILURE',s,row['kind'],row['duration'],row['amplitude'],mus[prefix].min(),mus[prefix].max(),int(prefix.sum()),flush=True)
        rows.append(row)
        if i%10==0:print(i,'/',len(starts),round(time.time()-begin,1),flush=True)
        (out/'grid-results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
