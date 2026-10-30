"""Rejected exploration must preserve the passive controller and estimator."""
import json
from pathlib import Path
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from drone_dual_gatekeeper.config import ControlConfig, Scenario
from drone_dual_gatekeeper.types import Candidate, CandidateStatus
from drone_dual_gatekeeper.simulation import runner
from drone_dual_gatekeeper.simulation.simulator import SimulationTruth
from scripts.validate_results import reconstruct_plan

@pytest.mark.parametrize('threshold,active_speed,accepted',[(0.,1.,True),(.05,1.,False),(.05,2.,True)])
def test_information_gate_compares_with_passive_backup(monkeypatch,threshold,active_speed,accepted):
    from drone_dual_gatekeeper.planning import dual_gatekeeper as gate
    from drone_dual_gatekeeper.types import Interval
    config=replace(ControlConfig(),minimum_information_advantage=threshold)
    def plan(speed):
        states=np.zeros((3,6));states[:,3]=speed
        return SimpleNamespace(certificate={'safe':True},nominal_states=states,timestamps=np.linspace(0,2,3),
                               state=lambda t:np.tile(states[0],(len(np.atleast_1d(t)),1)),horizon=2.)
    optimizer=SimpleNamespace(config=config,scenario=SimpleNamespace(goal=[2,0,1]),
                              solve=lambda *args,**kwargs:plan(active_speed))
    backup=plan(1.);backup.tube=SimpleNamespace(at=lambda t:(np.zeros(3),np.zeros(3)))
    monkeypatch.setattr(gate,'mission_cost',lambda *args:1.)
    monkeypatch.setattr(gate,'reconnection',lambda *args:True)
    selected,rows=gate.generate_and_select(optimizer,backup,np.zeros(6),Interval(0,.5),0.,1.)
    assert (selected is not None)==accepted
    if not accepted:assert rows[0].reason=='insufficient_information_advantage'


def test_every_candidate_rejected_recovers_passive_execution(monkeypatch):
    source=Path(__file__).resolve().parents[1]/'results/focused_demo/informative_demo_passive.json'
    record=json.loads(source.read_text())
    config=ControlConfig(**record['config']);scenario=Scenario(**record['scenario'])
    truth=SimulationTruth(**record['truth'])
    class RecordedBackupOptimizer:
        def __init__(self,c,s):
            self.config=c;self.scenario=s;self.index=0
        def solve(self,state,interval,warm=None):
            epoch=record['epochs'][self.index];self.index+=1
            assert np.allclose([interval.lower,interval.upper],epoch['interval_before'],atol=1e-10)
            plan=reconstruct_plan(epoch['backup'],config)
            assert np.allclose(plan.state(0),state,atol=3e-6)
            return plan
    monkeypatch.setattr(runner,'TrajectoryOptimizer',RecordedBackupOptimizer)
    def reject(*args):
        rows=[Candidate(2.,status=CandidateStatus.UNSAFE,reason='tube_safety'),
              Candidate(4.,status=CandidateStatus.UNSAFE,reason='reconnection')]
        return None,rows
    monkeypatch.setattr(runner,'generate_and_select',reject)
    passive=runner.run_method(config,scenario,'passive',truth,quiet=True)
    proposed=runner.run_method(config,scenario,'proposed',truth,quiet=True)
    assert proposed['metrics']['committed']==0
    assert proposed['metrics']['completion_time']==passive['metrics']['completion_time']
    assert proposed['observations']==passive['observations']
    assert proposed['samples']==passive['samples']
    # Current recordings stop at first arrival; legacy ones rounded to epoch end.
    if all('planned_duration' in e for e in record['epochs']):
        assert passive['metrics']['completion_time']==record['metrics']['completion_time']
    else:
        assert passive['metrics']['completion_time']<record['metrics']['completion_time']
    final=np.array(passive['samples'][-1]['state']);previous=np.array(passive['samples'][-2]['state'])
    assert np.linalg.norm(final[:3]-scenario.goal)<=config.goal_tolerance
    assert np.linalg.norm(final[3:])<=config.goal_velocity_tolerance
    assert np.linalg.norm(previous[:3]-scenario.goal)>config.goal_tolerance or np.linalg.norm(previous[3:])>config.goal_velocity_tolerance
    for epoch,baseline in zip(proposed['epochs'],passive['epochs']):
        assert epoch['passive_fallback']
        assert epoch['interval_after']==baseline['interval_after']
        assert epoch['committed']['coefficients']==baseline['committed']['coefficients']
        assert epoch['duration']==baseline['duration']


def test_discount_bound_skips_only_dominated_horizons(monkeypatch):
    from drone_dual_gatekeeper.planning import dual_gatekeeper as gate
    from drone_dual_gatekeeper.types import Interval
    config=ControlConfig();states=np.zeros((3,6));states[:,3]=2.
    candidate=SimpleNamespace(certificate={'safe':True},nominal_states=states,
                              timestamps=np.linspace(0,2,3))
    backup=SimpleNamespace(horizon=10.,state=lambda t:np.zeros((len(np.atleast_1d(t)),6)),
                           tube=SimpleNamespace(at=lambda t:(np.zeros(3),np.zeros(3))))
    horizons=[]
    def solve(*args,**kwargs):horizons.append(kwargs['horizon']);return candidate
    optimizer=SimpleNamespace(config=config,scenario=SimpleNamespace(goal=[2,0,1]),solve=solve)
    monkeypatch.setattr(gate,'mission_cost',lambda *args:1.)
    monkeypatch.setattr(gate,'reconnection',lambda *args:True)
    selected,rows=gate.generate_and_select(optimizer,backup,np.zeros(6),Interval(0,.5),0.,1.)
    assert selected.horizon==2. and horizons==[2.]
    assert selected.score>=np.exp(-config.score_discount*4)*.5
