"""Checks that the gap comes from admissible observations and tire physics."""
import numpy as np
import pytest
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.model import Model
from dual_gtk_drs_racing.belief import FrictionSet,observation
from dual_gtk_drs_racing.gatekeeper import Gatekeeper
from dual_gtk_drs_racing.safety import DRSCertifier,traction_point_margin,point_margin
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record


def initial(model):
    k=model.track.kappa(0.)
    return np.array([0.,0.,0.,8.,0.,8.*k,np.arctan(3.05*k)])


def test_combined_force_constraint_matches_independent_wheel_calculation():
    m=Model(Track("technical"));x=initial(m);mu=.45
    _,u,_,_=m.step(x,mu,36.)
    p=m.vehicle;fzf=.5*p.mass*9.81*p.lr/(p.lf+p.lr);fzr=.5*p.mass*9.81*p.lf/(p.lf+p.lr)
    fx=[.25*u[1]-p.rolling*fzf,.5*u[0]+.25*u[1]-p.rolling*fzr]
    slip=[x[6]-np.arctan((p.lf*x[5]+x[4])/(x[3]+.001)),np.arctan((p.lr*x[5]-x[4])/(x[3]+.001))]
    fy=[mu*fz*np.sin(p.C*np.arctan(p.B*a)) for fz,a in zip([fzf,fzr],slip)]
    expected=min(mu**2-(a*a+b*b)/fz**2 for a,b,fz in zip(fx,fy,[fzf,fzr]))
    assert m.traction_margin(x,mu,36.)==pytest.approx(expected)
    assert expected<0 and m.traction_margin(x,.85,36.)>0


def test_slow_probe_sampled_acceptance_and_independent_dense_holdout():
    m=Model(Track("technical"));x=initial(m);b=FrictionSet()
    cert=DRSCertifier(m.track,backup_duration=6.5,traction=True)
    assert cert.check(x,b,36.,0.,.6).reason=="sample_failure_limit"
    accepted=cert.check(x,b,8.,3.,.6)
    assert accepted.accepted and accepted.evaluated==1000 and accepted.failure_upper<=.01
    for mu in np.linspace(.45,1.05,73):
        xs,_,_=m.rollout(x,mu,8.,3.,.6,accepted.horizon-.6)
        assert point_margin(xs,m.track.curvature,m.track.ds)>=0
        assert traction_point_margin(xs,mu,8.,3.,.6,m.track.curvature,m.track.ds)>=0


def test_quantization_bound_does_not_depend_on_seed_or_ground_truth():
    phi=np.array([0.,0.,0.,0.,0.,2.,0.]);f0=np.zeros(7)
    for rate_value in np.linspace(-3,3,301):
        rate=f0.copy();rate[5]=rate_value
        y,g,w=observation(rate,phi,f0,np.random.default_rng(3),"quantized_yaw")
        assert abs(y[0]-rate_value)<=w[0]
        np.testing.assert_array_equal(g,[2.])


def test_zero_budget_only_allows_information_without_extra_mission_cost():
    m=Model(Track("technical"));x=initial(m);b=FrictionSet()
    g=Gatekeeper(m,"proposed",budget=0.,scenario="traction")
    g.replan(x,b,0.)
    assert g.charged==0.
    if g.committed.kind=='informative':
        selected=g.events[-1]['selected']
        nominal=next(a for a in g.events[-1]['nominal_attempts'] if a['steps']==selected['steps'])
        assert selected['charge']==0. and selected['cost']<=nominal['cost']
        assert selected['certificate']['accepted'] and nominal['certificate']['accepted']


def test_learning_unlocks_racing_without_giving_passive_different_rules():
    p=simulate("proposed",seconds=3.,scenario="traction")
    b=simulate("passive",seconds=3.,scenario="traction")
    n=simulate("robust",seconds=3.,scenario="traction")
    for r in (p,b,n):assert audit_record(r)["passed"]
    assert p["summary"]["informative_commits"]>=1
    assert p["summary"]["distance"]>b["summary"]["distance"]
    assert p["summary"]["final_mu"][1]-p["summary"]["final_mu"][0]<.2
    assert n["summary"]["final_mu"]==[.45,1.05]
    assert b["summary"]["final_mu"][1]-b["summary"]["final_mu"][0]>p["summary"]["final_mu"][1]-p["summary"]["final_mu"][0]
    # No method flag is passed into certification.
    event=next(e for e in p['events'] if e['status']=='committed')
    selected=event['selected'];m=Model(Track('technical'))
    from dual_gtk_drs_racing.control_safety import ControlCertifier
    cert=ControlCertifier(m.track,traction=True,backup_duration=10.)
    x=np.array(p['frames'][0]['state'])
    c=cert.check_plan(x,FrictionSet(),selected['controls'],selected['references'])
    assert c.accepted


def test_weighted_optimizes_one_control_plan_and_certifies_its_prefixes():
    m=Model(Track("technical"));x=initial(m)
    g=Gatekeeper(m,"weighted",scenario="traction",information_weight=.4)
    g.replan(x,FrictionSet(),0.)
    event=g.events[-1];attempts=event['weighted_attempts']
    valid=[a for a in attempts if a['certificate']['accepted']]
    assert event['selected']['duration']==max(a['duration'] for a in valid)
    assert g.charged==0.
    assert event['planner_diagnostics']['weighted']['gamma']==.4
    for shorter,longer in zip(attempts,attempts[1:]):
        assert shorter['controls']==longer['controls'][:shorter['steps']]
    g.replan(x,FrictionSet(.849999,.850001),.5)
    assert g.events[-1]['planner_diagnostics']['weighted']['gamma']==.4
    assert g.charged==0.


def test_quantized_prediction_retains_its_planning_center_and_intersects_strips():
    from dual_gtk_drs_racing.belief import predicted_quantized_width,excitation_score
    m=Model(Track("technical"));b=FrictionSet()
    xs,_,phi=m.rollout(initial(m),b.center,8.,3.,.6,0.)
    width=predicted_quantized_width(b,xs,phi)
    assert 0<width<b.width
    assert b.lower==.45 and b.upper==1.05
    assert excitation_score(phi,"quantized_yaw")>0


def test_high_speed_tracker_is_locally_stable():
    from dual_gtk_drs_racing.model import GRAPH,OUTPUTS
    from dual_gtk_drs_racing.expressions import evaluate
    idx=[1,2,4,5,6];k=np.zeros(21)
    for vx in (24.,35.,45.,55.):
        q=np.array([0.,0,0,vx,0,0,0,.85,vx,0,0]);columns=[]
        for j in idx:
            d=np.zeros(11);d[j]=1e-6
            columns.append((evaluate(q+d,GRAPH,OUTPUTS[:7],k,1.)[idx]-evaluate(q-d,GRAPH,OUTPUTS[:7],k,1.)[idx])/2e-6)
        assert max(abs(np.linalg.eigvals(np.column_stack(columns))))<1.


def test_high_speed_recovery_certificate_covers_holdout_parameters():
    m=Model(Track("technical"));s=300.;v=52.;k=m.track.kappa(s)
    x=np.array([s,0.,0.,v,0.,v*k,np.arctan(3.05*k)])
    b=FrictionSet(.849,.851)
    cert=DRSCertifier(m.track,backup_duration=10.,traction=True).check(x,b,55.,0.,.6)
    assert cert.accepted and cert.horizon>8.
    for mu in np.linspace(b.lower,b.upper,41):
        xs,_,_=m.rollout(x,mu,55.,0.,.6,cert.horizon-.6)
        assert point_margin(xs,m.track.curvature,m.track.ds)>=0
        assert traction_point_margin(xs,mu,55.,0.,.6,m.track.curvature,m.track.ds)>=0


def test_racing_reaches_fifty_and_weighted_keeps_learning():
    proposed=simulate("proposed",seconds=20.,scenario="traction")
    weighted=simulate("weighted",seconds=20.,scenario="traction")
    for record in (proposed,weighted):assert audit_record(record)["passed"]
    assert max(f["state"][3] for f in proposed["frames"])>=50.
    lo,hi=weighted["summary"]["final_mu"]
    assert lo<=.85<=hi and hi-lo<.01
    assert weighted["summary"]["informative_seconds"]>10.
    assert proposed["summary"]["distance"]>weighted["summary"]["distance"]>8.*20
