"""Information discounting changes the reward without changing safety constraints."""
import numpy as np
from drone_dual_gatekeeper.planning.optimizer import TrajectoryOptimizer
from drone_dual_gatekeeper.planning.curve import basis
from test_core import load_test_config


def test_discounted_reward_preserves_constraints_and_mission_cost():
    config,scenario=load_test_config()
    optimizer=TrajectoryOptimizer(config,scenario)
    _,evaluate,*_=optimizer._build(None,False)
    n=config.control_points
    cp=np.column_stack([np.linspace(0,3,n),np.zeros(n),np.ones(n)])
    duration=6.
    z=np.r_[cp.flatten(order='F'),duration]
    parameters=np.r_[scenario.start,scenario.goal,[0.,0.,0.],0.,.5,2.,[1e6]*6]
    constraints0,cost0,information0=evaluate(z,parameters)
    _,evaluate,*_=optimizer._build(None,False,information_discount=.5)
    constraints1,cost1,information1=evaluate(z,parameters)
    np.testing.assert_allclose(constraints0,constraints1,atol=0,rtol=0)
    assert float(cost0)==float(cost1)
    assert 0<float(information1)<float(information0)
    # Independent dense quadrature checks the time units in exp(-discount*t).
    b,_,_=basis(n,config.subdivisions)
    t=np.linspace(0,duration,20001);v=b(t/duration,nu=1)@cp/duration
    expected=np.trapezoid(np.exp(-.5*t)*np.sum(v*v,axis=1)**2,t)
    np.testing.assert_allclose(float(information1),expected,rtol=.02)


def test_numerical_retry_preserves_problem_and_records_attempts(monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from drone_dual_gatekeeper.types import Interval
    from drone_dual_gatekeeper.planning import optimizer as module
    config,scenario=load_test_config()
    config=replace(config,solver_hessian_approximation='exact')
    optimizer=TrajectoryOptimizer(config,scenario)
    class FailedSolver:
        def __call__(self,**kwargs):return {'x':kwargs['x0']}
        def stats(self):return {'return_status':'Restoration_Failed','iter_count':1}
    n=3*config.control_points+1
    monkeypatch.setattr(optimizer,'_build',lambda *args:(FailedSolver(),lambda *args:(np.zeros(13),1.,1.),12,13,
        np.full(n,-np.inf),np.full(n,np.inf),True))
    calls=[]
    class Retry:
        def __init__(self,c,s):
            assert s is scenario
            assert replace(c,solver_iterations=config.solver_iterations,solver_hessian_approximation='exact')==config
            self.config=c
        def solve(self,state,interval,**kwargs):
            calls.append((state,interval,kwargs))
            return SimpleNamespace(certificate={'safe':True},planning_time=0.)
    monkeypatch.setattr(module,'TrajectoryOptimizer',Retry)
    state=np.array(scenario.start);interval=Interval(0,.5)
    result=optimizer.solve(state,interval)
    assert len(calls)==1 and calls[0][1] is interval
    np.testing.assert_array_equal(calls[0][0],state)
    assert result.certificate['solver_retry']['maximum_iterations']>=1000
    assert len(result.certificate['solver_retry']['initial_attempts'])==3
