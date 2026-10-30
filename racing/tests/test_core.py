import numpy as np
import pytest
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.model import Model,GRAPH,OUTPUTS
from dual_gtk_drs_racing.expressions import evaluate
from dual_gtk_drs_racing.belief import FrictionSet,SENSOR_BOUND
from dual_gtk_drs_racing.gatekeeper import Gatekeeper,Policy
from dual_gtk_drs_racing.safety import DRSCertifier,Certificate,point_margin


@pytest.fixture(scope="module")
def model(): return Model(Track())


def test_appendix_friction_regressor(model):
    x=np.array([53.,.3,.04,17.,.4,.12,.08])
    a,_,phi,f0=model.step(x,.45,22.)
    b,_,_,_=model.step(x,1.05,22.)
    np.testing.assert_allclose((b-a)/.05,.6*phi,atol=2e-13)
    np.testing.assert_allclose((a-x)/.05,f0+.45*phi,atol=2e-13)
    p=model.vehicle
    front=.5*p.mass*9.81*p.lr/(p.lf+p.lr)*np.sin(p.C*np.arctan(p.B*(x[6]-np.arctan((p.lf*x[5]+x[4])/(x[3]+.001)))))
    rear=.5*p.mass*9.81*p.lf/(p.lf+p.lr)*np.sin(p.C*np.arctan(p.B*np.arctan((p.lr*x[5]-x[4])/(x[3]+.001))))
    np.testing.assert_allclose(phi[3:6],[-2*front*np.sin(x[6])/p.mass,2*(rear+front*np.cos(x[6]))/p.mass,2*(-p.lr*rear+p.lf*front*np.cos(x[6]))/p.inertia])


def test_set_membership_no_information_and_noisy_excitation():
    b=FrictionSet();b.update(np.zeros(3),np.zeros(3),SENSOR_BOUND);assert b.width==pytest.approx(.6)
    rng=np.random.default_rng(5)
    for _ in range(100):
        phi=rng.normal(size=3)*4
        b.update(.82*phi+rng.uniform(-SENSOR_BOUND,SENSOR_BOUND),phi,SENSOR_BOUND)
        assert b.lower<=.82<=b.upper
    assert b.width<.01
    with pytest.raises(ValueError): b.update(np.ones(3)*100,np.zeros(3),SENSOR_BOUND)


def test_rejection_preserves_exact_committed_policy(model):
    g=Gatekeeper(model)
    old=Policy("mission",-1.,3.,[[100.,0.,0.]]*60,[[0.,0.,0.,8.,0.,0.,0.]]*60)
    g.committed=old
    g.certifier.check_plan=lambda *a: Certificate(False,1,-1,1,"rejected",(.45,1.05))
    g.replan(np.array([0.,0,0,8,0,0,0]),FrictionSet(),0.)
    assert g.committed is old
    assert g.events[-1]["status"]=="retained"


def test_without_informative_options_proposed_recovers_passive(model):
    x=np.array([0.,0,0,8,0,0,0]);b=FrictionSet(.849,.851)
    proposed,passive=Gatekeeper(model,"proposed"),Gatekeeper(model,"passive")
    proposed.replan(x,b,0.);passive.replan(x,b,0.)
    assert proposed.committed==passive.committed


def test_rejecting_all_informative_candidates_recovers_passive(model):
    x=np.array([0.,0,0,8,0,0,0]);b=FrictionSet()
    proposed,passive=Gatekeeper(model,"proposed"),Gatekeeper(model,"passive")
    certify=proposed.certify
    count=[0]
    def reject_info(x,belief,candidate):
        accepted=certify(x,belief,candidate);count[0]+=1
        if count[0]>len(proposed.config.candidate_steps):
            candidate['certificate']['accepted']=False
            return False
        return accepted
    proposed.certify=reject_info
    proposed.replan(x,b,0.);passive.replan(x,b,0.)
    assert proposed.committed==passive.committed


def test_budget_blocks_informative_commit(model):
    g=Gatekeeper(model,budget=0.)
    g.replan(np.array([0.,0,0,8,0,0,0]),FrictionSet(),0.)
    assert g.charged==0
    if g.committed.kind=="informative":assert g.events[-1]["selected"]["charge"]==0


def test_graph_steering_and_force_saturation(model):
    x=np.array([0.,4.,.5,8.,1.,.2,.32])
    _,u,_,_=model.step(x,.8,50.)
    assert 0<=u[0]<=720*4 and -720*7<=u[1]<=0 and abs(u[2])<=.65


def test_short_closed_loop_replay_and_truth_containment():
    from dual_gtk_drs_racing.simulation import simulate
    from dual_gtk_drs_racing.validation import audit_record
    for truth in (.45,1.05):
        result=simulate(seconds=1.,true_mu=truth)
        assert audit_record(result)["passed"]


def test_positive_exploration_charge_is_blocked_by_zero_budget(model):
    g=Gatekeeper(model,budget=0.);original=g.candidate;counter=[0]
    def costly_information(plan,n,belief):
        a=original(plan,n,belief);counter[0]+=1
        if counter[0]>len(g.config.candidate_steps):a['cost']+=100.
        return a
    g.candidate=costly_information
    g.replan(np.array([0.,0.,0.,8.,0.,0.,0.]),FrictionSet(),0.)
    assert g.committed.kind!='informative' and g.charged==0
    assert all(not a['budget_feasible'] for a in g.events[-1]['informative_attempts'])
