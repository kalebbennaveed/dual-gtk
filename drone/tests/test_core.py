from dataclasses import replace
import inspect
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from drone_dual_gatekeeper.config import load_config
from drone_dual_gatekeeper.types import Interval,DisturbanceSet
from drone_dual_gatekeeper.dynamics.quadrotor_drag import derivative,drag_regressor
from drone_dual_gatekeeper.planning.curve import Curve
from drone_dual_gatekeeper.control.tube_propagation import propagate,build_tube,delta_bound
from drone_dual_gatekeeper.control.ancillary_controller import ancillary_control
from drone_dual_gatekeeper.estimation.smid import intersect_regression,update_interval
from drone_dual_gatekeeper.estimation.uncertainty_prediction import predict_uncertainty_reduction
from drone_dual_gatekeeper.planning.dual_gatekeeper import reconnection,budget_feasible,candidate_horizons
from drone_dual_gatekeeper.planning.optimizer import TrajectoryOptimizer
from drone_dual_gatekeeper.environment.collision import cell_clearance


def test_drag_and_gravity():
    x=np.array([0,0,1,1,2,0]);u=np.array([0,0,9.81]);g=np.array([0,0,-9.81])
    assert np.allclose(derivative(x,u,.2,g)[3:],.2*drag_regressor(x[3:]))
    assert np.allclose(drag_regressor(np.zeros(3)),0)

@pytest.mark.parametrize('alpha,lam',[(2.,2.),(1.5,2.),(3.,1.)])
def test_exact_tube_propagation(alpha,lam):
    p=np.array([.1,.2,.3]);r=np.array([.2,.1,0]);f=np.array([.5,.2,.1])
    q=solve_ivp(lambda t,y:np.r_[-alpha*y[:3]+f,-lam*y[3:]+y[:3]],[0,2],np.r_[p,r],rtol=1e-11,atol=1e-12).y[:,-1]
    p1,r1=propagate(p,r,f,2,alpha,lam)
    assert np.allclose(np.r_[p1,r1],q,atol=1e-10)


def test_bernstein_bounds_cover_continuous_spline():
    rng=np.random.default_rng(8);curve=Curve(rng.normal(size=(10,3)),8.,6)
    for i in range(len(curve.edges)-1):
        ts=np.linspace(curve.edges[i]*8,curve.edges[i+1]*8,41)
        for derivative in range(3):
            lo,hi=curve.cell_bounds(i,derivative);v=curve.value(ts,derivative)
            assert np.all(v>=lo-1e-10) and np.all(v<=hi+1e-10)


def test_tube_bandwidth_fixed_geometry_changes():
    c,_=load_test_config();cp=np.zeros((10,3));cp[:,0]=np.linspace(0,3,10);cp[:,2]=1
    curve=Curve(cp,5.,c.subdivisions);tube=build_tube(curve,Interval(0,.5),c)
    assert tube.alpha==c.alpha and tube.sliding_lambda==c.sliding_lambda
    assert np.ptp(tube.position_radius[:,0])>.01
    src=inspect.getsource(TrajectoryOptimizer._build)
    assert "MX.sym('alpha" not in src and "MX.sym('radius" not in src and "MX.sym('bandwidth" not in src
    assert 'tube-size penalty' in src


def test_sliding_boundary_attractive_for_parameter_extremes():
    c,_=load_test_config();interval=Interval(0,.5);nom=np.array([0,0,1,1.,.3,-.2])
    phi=np.array([.4,.2,.1]);rho=np.array([.1,.06,.03]);forcing=delta_bound(interval,.25,abs(nom[3:])+0.1)
    for truth in [interval.lower,interval.upper]:
        for sign in [-1.,1.]:
            er=sign*rho;ev=sign*phi-c.sliding_lambda*er;actual=nom+np.r_[er,ev]
            ubar=.25*np.linalg.norm(nom[3:])*nom[3:]-np.array(c.gravity)
            anc=ancillary_control(actual,nom,phi,forcing,interval,c)
            dot=derivative(actual,ubar+anc,truth,np.array(c.gravity),sign*np.ones(3)*c.disturbance_bound)
            sdot=dot[3:]+c.sliding_lambda*ev
            phidot=-c.alpha*phi+forcing+c.disturbance_bound+c.eta
            assert np.all(sign*sdot<=phidot-c.eta+1e-10)


def test_smid_nested_negative_zero_noisy_regressors():
    prior=Interval(0,.5);truth=.147
    f=np.array([-1.,2.,0.]);noise=np.array([.02,-.02,.01]);y=f*truth+noise
    result=intersect_regression(prior,f,y,.02)
    assert result.contains(truth) and result.width<prior.width
    with pytest.raises(ValueError):intersect_regression(prior,[0],[1],.1)
    with pytest.raises(ValueError):intersect_regression(prior,[1],[4],.01)


def test_prediction_width_factor_four_and_no_truth():
    class Plan:nominal_states=np.array([[0,0,0,2.,0,0]])
    p=predict_uncertainty_reduction(Plan(),Interval(0,.5),DisturbanceSet(.08))
    assert p.predicted_width==pytest.approx(.08)
    assert p.interval_is_surrogate and p.absolute_reduction==pytest.approx(.42)
    assert 'true_drag' not in inspect.signature(predict_uncertainty_reduction).parameters
    with pytest.raises(TypeError):predict_uncertainty_reduction(Plan(),Interval(0,.5),DisturbanceSet(.08),true_drag_coefficient=.1)


def test_full_tube_collision_not_centerline():
    obs={'kind':'cylinder','center':[0,0,1],'height':2,'radius':1}
    assert cell_clearance(np.array([1.3,0,1]),np.array([1.3,0,1]),np.zeros(3),obs,.1)>0
    assert cell_clearance(np.array([1.3,0,1]),np.array([1.3,0,1]),np.array([.3,.3,.1]),obs,.1)<0


def test_budget_and_horizon_cap():
    assert budget_feasible(2,3,5)
    assert not budget_feasible(2,3.1,5)
    assert candidate_horizons(7,2)==[2,4,6,7]


def test_reconnection_requires_tube_inclusion():
    class Tube:
        def __init__(self,size):self.size=size
        def at(self,t):return np.ones(3)*self.size,np.ones(3)*self.size
    class Plan:
        def __init__(self,size):self.tube=Tube(size)
        def state(self,t):return np.zeros(6)
    assert reconnection(Plan(.1),Plan(.2),2)
    assert not reconnection(Plan(.3),Plan(.2),2)


def test_no_ground_truth_planning_interface():
    assert 'true_drag_coefficient' not in inspect.signature(TrajectoryOptimizer.solve).parameters
    c,s=load_test_config()
    with pytest.raises(TypeError):TrajectoryOptimizer(c,s).solve(np.array(s.start),Interval(0,.5),true_drag_coefficient=.1)

@pytest.mark.parametrize('truth',[0.0,0.5])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_adversarial_bound_disturbances_and_initial_tube_edge(truth,sign):
    from drone_dual_gatekeeper.simulation.simulator import TrueDroneSimulator,SimulationTruth
    from drone_dual_gatekeeper.types import RobustPlan
    c,s=load_test_config();c=replace(c,input_limits=(1.,1.,12.),simulation_step=.01)
    cp=np.zeros((c.control_points,3));cp[:,2]=1.;curve=Curve(cp,2.,c.subdivisions)
    prior=Interval(0,.5);rho=np.full(3,.025);phi=c.initial_boundary_layer
    tube=build_tube(curve,prior,c,phi0=phi,rho0=rho)
    nominal=curve.state(0);er=sign*rho*.99;ev=sign*phi*.99-c.sliding_lambda*er
    actual=nominal+np.r_[er,ev]
    plan=RobustPlan(curve.state([0,2]),np.zeros((2,3)),np.array([0,2]),np.array([phi,phi]),np.zeros((2,6)),0.,2.,curve,tube,.25,'test',0.,0.,0.)
    simulator=TrueDroneSimulator(SimulationTruth(truth,19,measurement_noise=.001),c,s)
    simulator.disturbances[:]=sign*c.disturbance_bound
    final,samples,observations=simulator.execute(plan,prior,actual,0,2.)
    updated=update_interval(prior,observations,c.disturbance_bound)
    assert updated.contains(truth)
    assert max(x['tube_utilization'] for x in samples)<=1+2e-5
    assert max(max(abs(np.array(x['control']))) for x in samples)<=12.


def test_seeded_simulation_reproducibility():
    from drone_dual_gatekeeper.simulation.simulator import TrueDroneSimulator,SimulationTruth
    c,s=load_test_config();truth=SimulationTruth(.1,17)
    a=TrueDroneSimulator(truth,c,s);b=TrueDroneSimulator(truth,c,s)
    assert np.array_equal(a.disturbances,b.disturbances)
    assert np.max(abs(a.disturbances))<=c.disturbance_bound


def test_sensor_noise_shared_on_physical_clock():
    from drone_dual_gatekeeper.simulation.simulator import TrueDroneSimulator,SimulationTruth
    c,s=load_test_config();c=replace(c,measurement_noise=.002)
    truth=SimulationTruth(.4,17,disturbance_profile='constant_positive')
    a=TrueDroneSimulator(truth,c,s);b=TrueDroneSimulator(truth,replace(c,simulation_step=.005),s)
    for t in [0,.2,1.337,4.]:
        assert np.array_equal(a.sensor_noise(t),b.sensor_noise(t))
        assert np.max(abs(a.sensor_noise(t)))<=c.measurement_noise
    assert np.all(a.disturbances==c.disturbance_bound)


def test_optimizer_and_executed_tube_use_identical_forcing():
    c,s=load_test_config()
    cp=np.zeros((c.control_points,3));cp[:,0]=np.linspace(0,3,c.control_points);cp[:,2]=1
    curve=Curve(cp,4.,c.subdivisions);prior=Interval(0,.5)
    tube=build_tube(curve,prior,c);phi,rho=tube.at(4.)
    _,evaluate,*_=TrajectoryOptimizer(c,s)._build(4.,True)
    parameters=np.r_[curve.state(0),curve.state(4),0,.5,0,phi,rho]
    constraints,_,_=evaluate(cp.flatten(order='F'),parameters)
    # Exact terminal inclusion must hold even in a zero nominal-motion axis.
    assert np.max(abs(np.array(constraints).ravel()[-6:]))<5e-12


def test_diagonal_bernstein_support_preserves_coordinate_correlation():
    coefficients=np.array([[0,0,1],[1/3,1/3,1],[2/3,2/3,1],[1,1,1]])
    lo=coefficients.min(axis=0);hi=coefficients.max(axis=0);rho=np.full(3,.02)
    obstacle={'kind':'cylinder','center':[.75,.25,1],'radius':.15,'height':3}
    assert cell_clearance(lo,hi,rho,obstacle,.1)<0  # AABB cannot certify this path.
    lower=cell_clearance(lo,hi,rho,obstacle,.1,coefficients)
    assert lower>0
    from drone_dual_gatekeeper.environment.collision import obstacle_clearance
    from itertools import product
    for t in np.linspace(0,1,101):
        for signs in product([-1,1],repeat=3):
            point=np.array([t,t,1])+np.array(signs)*rho
            assert obstacle_clearance(point,obstacle,.1)>=lower-1e-12


def load_test_config():
    from pathlib import Path
    import yaml
    from drone_dual_gatekeeper.config import CONFIG_DIR, ControlConfig, Scenario
    data=yaml.safe_load((Path(__file__).parent/'fixtures/open_mission.yaml').read_text())
    common=yaml.safe_load((CONFIG_DIR/'common.yaml').read_text())
    common.update(data.pop('overrides',{}))
    return ControlConfig(**common),Scenario(**{**data,'obstacles':tuple(data['obstacles'])})
