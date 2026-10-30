"""Reproduce the disclosed prior screen; this is not a random-trial estimate."""
from pathlib import Path
import argparse
import json
import numpy as np
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.model import Model
from dual_gtk_drs_racing.legacy_gatekeeper import Gatekeeper
from dual_gtk_drs_racing.belief import FrictionSet
from dual_gtk_drs_racing.stress_diagnostics import rollout_outcomes


def screen():
    tr=Track('technical');model=Model(tr)
    x=np.array([0.,0.,0.,8.,0.,8*tr.kappa(0),np.arctan(3.05*tr.kappa(0))]);rows=[]
    for lower in (.74,.745,.75,.755,.76,.765):
        for scenario in ('traction','traction_relaxed'):
            g=Gatekeeper(model,scenario=scenario,seed=7);g.replan(x,FrictionSet(lower,1.05),0.)
            e=g.events[-1];p=g.committed
            r=dict(lower=lower,scenario=scenario,policy=e['policy'],status=e['status'])
            if e['status']=='committed':
                c=e['selected']['certificate'];mus=np.linspace(lower,1.05,2001)
                refs=np.array([p.action_reference(j*.05) for j in range(round(c['horizon']/.05))])
                times,kinds=rollout_outcomes(x,mus,refs,tr.curvature,tr.ds)
                fail=(kinds>0)&(kinds<3)&(times<p.duration-1e-9)
                r.update(upper=c['failure_upper'],sample_failures=c['failures'],prefix_failures=int(fail.sum()),grid_samples=len(mus),
                    lowest_failing=float(mus[fail].min()) if fail.any() else None,
                    highest_failing=float(mus[fail].max()) if fail.any() else None,
                    physical_failures=int(((kinds>0)&(kinds<3)).sum()),terminal_failures=int((kinds==3).sum()))
            rows.append(r)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/stress-discovery/prior-screen-reproduced.json'))
    args=p.parse_args();rows=screen();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2));print(args.output)
