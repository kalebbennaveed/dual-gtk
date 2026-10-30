"""Independent checks for explicit dynamics, MPPI and control certification."""
import numpy as np
import pytest
from dual_gtk_drs_racing.model import Model
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.dynamics import plant_step,fallback_control,tracking_control,rollout_controls
from dual_gtk_drs_racing.safety import DRSCertifier,scalar_margin,terminal_margin
from dual_gtk_drs_racing.control_safety import ControlCertifier,control_rollouts
from dual_gtk_drs_racing.belief import FrictionSet
from dual_gtk_drs_racing.mppi import MPPI,MPPIConfig,objective,physical_controls


def test_explicit_inputs_reproduce_original_physics_and_fallback():
    rng=np.random.default_rng(551);model=Model(Track('technical'));tr=model.track
    for _ in range(300):
        x=np.array([rng.uniform(0,tr.length),rng.uniform(-4,4),rng.uniform(-.4,.4),rng.uniform(3,60),rng.uniform(-2,2),rng.uniform(-.5,.5),rng.uniform(-.3,.3)])
        mu=rng.uniform(.45,1.05);refs=(rng.uniform(8,55),rng.uniform(-3,3),rng.uniform(-4,4))
        expected,u,phi,f0=model.step(x,mu,*refs)
        control=fallback_control(x,tr.curvature,tr.ds,*refs)
        np.testing.assert_allclose(control,u,atol=1e-11)
        actual,g,b,tire=plant_step(x,mu,control,tr.curvature,tr.ds)
        np.testing.assert_allclose(actual,expected,atol=2e-12)
        np.testing.assert_allclose(g,phi,atol=2e-12)
        np.testing.assert_allclose(b,f0,atol=2e-12)
        assert min(tire)==pytest.approx(model.traction_margin(x,mu,*refs),abs=2e-12)


def test_adapter_keeps_fallback_certificate_and_binomial_decision():
    tr=Track('technical');k=tr.kappa(0.);x=np.array([0.,0.,0.,8.,0.,8*k,np.arctan(3.05*k)])
    for eps in (.01,.05):
        legacy=DRSCertifier(tr,traction=True,epsilon=eps,seed=19).check(x,FrictionSet(),8.,0.,0.)
        adapter=ControlCertifier(tr,traction=True,epsilon=eps,seed=19).check_plan(x,FrictionSet(),[],[])
        assert legacy.accepted==adapter.accepted
        assert legacy.failures==adapter.failures and legacy.samples==adapter.samples
        assert legacy.failure_upper==adapter.failure_upper and legacy.sample_seed==adapter.sample_seed
        assert legacy.margin==pytest.approx(adapter.margin,abs=1e-12)


def test_control_certificate_matches_full_independent_policy_rollouts():
    tr=Track('technical');x=np.array([0.,0.,0.,8.,0.,0.,0.]);mu=.75
    controls=np.tile([700.,0.,.06],(6,1));states,_=rollout_controls(x,mu,controls,tr.curvature,tr.ds)
    mus=np.random.default_rng(1).uniform(.45,1.05,29);tail=70;kmax=max(abs(tr.curvature))
    got=control_rollouts(x,mus,controls,states[:-1],tail,tr.curvature,tr.ds,kmax,True,29)
    failures=0
    for truth in mus:
        state=x.copy();h=scalar_margin(state,kmax)
        for j in range(len(controls)+tail):
            u=tracking_control(state,states[j],controls[j]) if j<len(controls) else fallback_control(state,tr.curvature,tr.ds)
            state,_,_,tire=plant_step(state,truth,u,tr.curvature,tr.ds)
            h=min(h,min(tire),scalar_margin(state,kmax))
        failures+=min(h,terminal_margin(state,tr.curvature,tr.ds))<=0
    assert got[0]==failures and got[1]==29


def test_mppi_optimizes_controls_and_information_changes_objective():
    tr=Track('technical');x=np.array([0.,0.,0.,8.,0.,0.,0.]);previous=np.array([84.7584,0.,0.])
    cfg=MPPIConfig(horizon=24,samples=64,iterations=2,candidate_steps=(6,12,24))
    planner=MPPI(tr,55.,'quantized_yaw',True,seed=8,config=cfg)
    result=planner.optimize(x,.75,0.,previous,gamma=.4)
    assert result['controls'].shape==(24,3)
    assert result['diagnostics']['objective']<=result['diagnostics']['initial_objective']+1e-12
    assert len(np.unique(result['controls'][:,2]))>10
    args=(tr.curvature,tr.ds,max(abs(tr.curvature)),55.)
    a=objective(x,.75,result['coordinates'],*args,0.,1.,1,True,previous)
    b=objective(x,.75,result['coordinates'],*args,.4,1.,1,True,previous)
    gramian=.05*np.sum((result['phis'][:,5]/.18)**2)
    assert a-b==pytest.approx(.4*np.log(1.+gramian))
    assert not np.any((result['controls'][:,0]>0)&(result['controls'][:,1]<0))


def test_zero_information_weight_is_exactly_the_passive_controller():
    from dual_gtk_drs_racing.simulation import simulate
    a=simulate('passive',seconds=1.,scenario='traction')
    b=simulate('weighted',seconds=1.,scenario='traction',information_weight=0.)
    assert a['frames']==b['frames']
    assert a['summary']['cost']==b['summary']['cost']


def test_active_planner_has_no_neighboring_repository_imports():
    import ast
    from pathlib import Path
    root=Path(__file__).parents[1]/'dual_gtk_drs_racing'
    for name in ('mppi.py','dynamics.py','gatekeeper.py','control_safety.py','simulation.py'):
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('uncertain_racecar_gym','pi_mpc','racing_mppi'))
            if isinstance(node,ast.Import):assert not any(n.name.startswith(('uncertain_racecar_gym','pi_mpc','racing_mppi')) for n in node.names)


def test_invalid_control_policy_cannot_receive_a_certificate():
    tr=Track('technical');x=np.array([0.,0.,0.,8.,0.,0.,0.]);cert=ControlCertifier(tr)
    for control in ([3000.,0.,0.],[0.,1.,0.],[0.,0.,1.],[float('nan'),0.,0.]):
        with pytest.raises(ValueError):cert.check_plan(x,FrictionSet(),[control],[x])


def test_proposed_information_objective_is_fixed_and_not_a_tuning_input():
    from dual_gtk_drs_racing.simulation import simulate
    config=MPPIConfig(horizon=12,samples=32,iterations=1,candidate_steps=(6,12))
    low=simulate('proposed',seconds=.3,scenario='traction',information_weight=0.,planner_config=config)
    high=simulate('proposed',seconds=.3,scenario='traction',information_weight=20.,planner_config=config)
    assert low['frames']==high['frames']
    assert low['config']['information_weight']==high['config']['information_weight']==2.
    for record in (low,high):
        assert record['events'][0]['planner_diagnostics']['informative']['gamma']==2.
        assert record['events'][0]['planner_diagnostics']['nominal']['gamma']==0.
