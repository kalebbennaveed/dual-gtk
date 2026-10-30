"""Reproducible closed-loop simulation; truth is confined to plant/sensor code."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import time
import numpy as np
from .track import Track
from .model import Model
from .belief import FrictionSet,observation
from .gatekeeper import Gatekeeper,stage_cost
from .safety import margins
from .dynamics import plant_step
from .mppi import MPPIConfig
from .scenarios import get_scenario


def simulate(method="proposed",seconds=35.,seed=7,true_mu=.85,max_cells=256,start=0.,progress_callback=None,scenario="barcelona",information_weight=.4,initial_mu=(.45,1.05),planner_config=None):
    lower,upper=map(float,initial_mu)
    if not (np.isfinite(lower) and np.isfinite(upper) and 0<lower<upper):
        raise ValueError("Initial friction bounds must be finite, positive and ordered")
    if not lower<=true_mu<=upper: raise ValueError("True friction must lie in the declared initial support")
    experiment=get_scenario(scenario)
    track=Track(experiment.track);model=Model(track);belief=FrictionSet(lower,upper)
    gk=Gatekeeper(model,method,max_cells=max_cells,scenario=scenario,information_weight=information_weight,seed=seed,planner_config=planner_config)
    x=np.array([start,0.,0.,8.,0.,8.*track.kappa(start),np.arctan(3.05*track.kappa(start))])
    gk.certifier.seed=seed+1000000
    initial=gk.certifier.check_plan(x,belief,[],[])
    gk.certifier.seed=seed
    if not initial.accepted: raise RuntimeError(f"Initial backup not certified: {initial}")
    rng=np.random.default_rng(seed)
    frames=[];cost=0.;min_clearance=np.inf;start_wall=time.perf_counter()
    previous_mode="backup"
    termination="timeout";violation=None
    covered_until=initial.horizon or experiment.backup_duration
    for i in range(int(round(seconds/model.dt))+1):
        t=i*model.dt
        info_finished=previous_mode!="backup" and t>=gk.committed.start+gk.committed.duration-1e-8
        if i%gk.config.replan_steps==0 or info_finished: gk.replan(x,belief,t)
        if gk.events and gk.events[-1]["time"]==t and gk.events[-1]["status"]=="committed":
            covered_until=t+gk.events[-1]["selected"]["certificate"]["horizon"]
        coverage_failed=False
        if t+model.dt>covered_until+1e-9:
            # Never silently execute beyond the verified tail by treating the
            # recovery envelope as a proven invariant set. Renew the same backup
            # law, without changing the old committed policy or its switch time.
            if t<gk.committed.start+gk.committed.duration:
                raise RuntimeError("Finite-horizon certificate expired during a committed prefix")
            if gk.certifier.decision_index!=int(round(t/.05)):
                gk.certifier.begin_decision(t)
            renewal=gk.certifier.check_plan(x,belief,[],[])
            if not renewal.accepted:coverage_failed=True
            else:covered_until=t+renewal.horizon
            gk.events.append(dict(time=t,status="backup_renewal" if renewal.accepted else "backup_unavailable",policy=asdict(gk.committed),
                previous=asdict(gk.committed),certificate=asdict(renewal),budget_charged=gk.charged,
                support=[belief.lower,belief.upper],certification_checks=gk.certifier.checks.copy()))
        mode=gk.committed.kind if t<gk.committed.start+gk.committed.duration-1e-8 else "backup"
        clearance=track.width/2-model.vehicle.radius-abs(x[1])
        min_clearance=min(min_clearance,clearance)
        frames.append(dict(t=round(t,6),state=x.tolist(),pose=track.pose(x).tolist(),mu=[belief.lower,belief.upper],
                           mode=mode,cost=cost,clearance=float(clearance),budget=gk.charged,certified_until=covered_until))
        if margins(x,np.zeros(7),track.curvature,track.ds)<-1e-8:
            termination="constraint_violation";violation=dict(time=t,kind="state",clearance=float(clearance));break
        if coverage_failed:
            termination="certificate_unavailable";break
        if x[0]-start>=track.length:
            termination="lap_complete";break
        if i==int(round(seconds/model.dt)):break
        u=gk.committed.action(x,t,track)
        nxt,phi,f0,tire=plant_step(x,true_mu,u,track.curvature,track.ds)
        if experiment.traction:
            force_margin=float(tire.min())
            frames[-1]["traction_margin"]=force_margin
            if force_margin<0:
                termination="constraint_violation";violation=dict(time=t,kind="tire_force",margin=float(force_margin));break
        gk.last_action=u.copy()
        measured_rate=(nxt-x)/model.dt
        residual,regressor,bound=observation(measured_rate,phi,f0,rng,experiment.sensor)
        frames[-1]["observation"]=dict(residual=residual.tolist(),regressor=regressor.tolist(),bound=bound.tolist())
        if method!="robust": belief.update(residual,regressor,bound)
        if not belief.lower-1e-9<=true_mu<=belief.upper+1e-9:
            raise RuntimeError("Set membership excluded the true friction")
        cost+=float(model.dt*stage_cost(np.stack([x,nxt]),u[None,:],experiment.racing_speed)[0])
        frames[-1]["control"]=u.tolist()
        x=nxt
        previous_mode=mode
        if progress_callback and i%100==0: progress_callback(t,x,belief)
    summary=dict(method=method,time=frames[-1]["t"],distance=float(x[0]-start),
        lap_complete=termination=="lap_complete",termination=termination,violation=violation,
        constraint_violated=termination=="constraint_violation",certificate_unavailable=termination=="certificate_unavailable",cost=cost,minimum_clearance=float(min_clearance),
        final_mu=[belief.lower,belief.upper],initial_mu=[lower,upper],true_mu=true_mu,
        informative_commits=sum(e.get("selected",{}).get("kind")=="informative" for e in gk.events),
        retained_plans=sum(e["status"]=="retained" for e in gk.events),
        candidate_rejections=sum(not a["certificate"]["accepted"] for e in gk.events for a in e.get("nominal_attempts",[])+e.get("weighted_attempts",[])),
        accepted_plans=sum(e["status"]=="committed" for e in gk.events),
        mission_seconds=sum(f["mode"]=="mission" for f in frames[:-1])*model.dt,
        backup_seconds=sum(f["mode"]=="backup" for f in frames[:-1])*model.dt,
        informative_seconds=sum(f["mode"]=="informative" for f in frames[:-1])*model.dt,
        budget_charged=gk.charged,budget_limit=gk.budget,wall_seconds=time.perf_counter()-start_wall)
    package=Path(__file__).parent
    fingerprint=hashlib.sha256(b"".join((package/p).read_bytes() for p in
        ("model.py","fast_model.py","expressions.py","safety.py","gatekeeper.py","belief.py","simulation.py","track.py","scenarios.py","dynamics.py","mppi.py","control_safety.py","assets/barcelona.csv"))).hexdigest()
    return dict(schema=3,implementation=fingerprint,summary=summary,dt=model.dt,seed=seed,vehicle=asdict(model.vehicle),
                config=dict(seconds=seconds,true_mu=true_mu,max_cells=max_cells,start=start,scenario=scenario,information_weight=gk.information_weight,initial_mu=[lower,upper],planner_config=asdict(gk.config)),scenario=asdict(experiment),
                frames=frames,events=gk.events,initial_certificate=asdict(initial),
                safety_scope="Sampled DRS with beta=0, conditional uniform friction, alpha=0 assumed invariant backup; confidence and risk are per decision, not whole-lap guarantees.")
