"""Certificate statistics, faithful rollouts, and honest violation reporting."""
import numpy as np
import pytest
from scipy.stats import beta
from dual_gtk_drs_racing.fast_model import step_and_tire
from dual_gtk_drs_racing.model import Model,commands
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.belief import FrictionSet
from dual_gtk_drs_racing.safety import DRSCertifier,binomial_upper,sampled_rollouts,point_margin,traction_point_margin
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record


def initial(tr):
    k=tr.kappa(0.)
    return np.array([0.,0.,0.,8.,0.,8*k,np.arctan(3.05*k)])


def test_compiled_model_matches_canonical_graph_across_states_and_clipping():
    rng=np.random.default_rng(871)
    for name in ('technical','barcelona'):
        model=Model(Track(name));tr=model.track
        for _ in range(500):
            x=np.array([rng.uniform(0,tr.length),rng.uniform(-5.5,5.5),rng.uniform(-.4,.4),
                rng.uniform(3,64),rng.uniform(-3,3),rng.uniform(-1,1),rng.uniform(-.32,.32)])
            mu=rng.uniform(.45,1.05);refs=(rng.uniform(8,55),rng.uniform(-3,3),rng.uniform(-4,4))
            expected=model.step(x,mu,*refs)[0]
            actual=step_and_tire(tuple(x),mu,*refs,tr.curvature,tr.ds)
            np.testing.assert_allclose(actual[:7],expected,rtol=2e-13,atol=2e-12)
            assert min(actual[7:])==pytest.approx(model.traction_margin(x,mu,*refs),abs=2e-12)


def test_full_sample_classification_matches_independent_rollouts():
    tr=Track('technical');model=Model(tr);x=initial(tr)
    for speed,amp in ((8.,3.),(55.,0.)):
        duration=.6;tail=3.5;mus=np.random.default_rng(81).uniform(.45,1.05,113)
        refs=np.array([commands(j*.05,duration,speed,amp,8.) for j in range(82)])
        failures,n,margin,steps=sampled_rollouts(x,mus,refs,tr.curvature,tr.ds,max(abs(tr.curvature)),True,len(mus))
        expected=0
        for mu in mus:
            xs,_,_=model.rollout(x,mu,speed,amp,duration,tail)
            h=min(point_margin(xs,tr.curvature,tr.ds),traction_point_margin(xs,mu,speed,amp,duration,tr.curvature,tr.ds))
            expected+=int(h<=0)
        assert failures==expected and n==len(mus)


def test_binomial_formula_correction_and_tolerance():
    assert binomial_upper(0,1000,.05/64)==pytest.approx(1-(.05/64)**.001)
    for k in (0,1,15,1000):
        expected=1. if k==1000 else beta.ppf(1-.05/64,k+1,1000-k)
        assert binomial_upper(k,1000,.05/64)==pytest.approx(expected)
    tr=Track('technical');tight=DRSCertifier(tr);relaxed=DRSCertifier(tr,epsilon=.05)
    assert relaxed.allowed_failures>tight.allowed_failures>=0
    np.testing.assert_array_equal(tight.upper,relaxed.upper)


def test_early_rejection_never_produces_an_acceptance_certificate():
    tr=Track('technical');x=initial(tr);b=FrictionSet()
    cert=DRSCertifier(tr,traction=True).check(x,b,55.,0.,1.8)
    assert not cert.accepted and not cert.complete_batch and cert.evaluated<1000
    assert cert.failure_upper==1.
    c=DRSCertifier(tr,traction=True,samples=10).check(x,b,8.,0.,0.)
    assert not c.accepted and c.reason=='insufficient_sample_budget' and c.evaluated==0


def test_accepted_batches_and_fresh_samples_are_reproducible():
    tr=Track('technical');x=initial(tr);b=FrictionSet()
    a=DRSCertifier(tr,traction=True,seed=19);z=DRSCertifier(tr,traction=True,seed=19)
    c=a.check(x,b,8.,0.,0.)
    assert c==z.check(x,b,8.,0.,0.)
    assert c.accepted and c.complete_batch and c.evaluated==1000
    assert c==a.check(x,b,8.,0.,0.) and len(a.checks)==1
    d=a.check(x,b,8.,1.5,.6)
    assert d.sample_seed!=c.sample_seed
    a.begin_decision(.5)
    assert a.check(x,b,8.,0.,0.).sample_seed!=c.sample_seed


def test_candidate_confidence_budget_is_enforced():
    tr=Track('technical');v=DRSCertifier(tr,traction=True,candidate_limit=1)
    v.check(initial(tr),FrictionSet(),8.,0.,0.)
    with pytest.raises(RuntimeError,match='confidence budget'):
        v.check(initial(tr),FrictionSet(),8.,1.5,.6)


def test_real_constraint_failure_is_preserved_as_an_outcome(monkeypatch):
    from dual_gtk_drs_racing.gatekeeper import Gatekeeper,Policy
    def force_bad_policy(self,x,b,t):self.committed=Policy('mission',0.,3.,[[2880.,0.,0.]]*60,[x.tolist()]*60)
    monkeypatch.setattr(Gatekeeper,'replan',force_bad_policy)
    record=simulate('passive',seconds=2.,true_mu=.45,scenario='traction')
    assert record['summary']['constraint_violated']
    assert record['summary']['termination']=='constraint_violation'
    assert record['summary']['violation']['kind']=='tire_force'
    assert len(record['frames'])==1


def test_sampled_certificate_audit_detects_tampering():
    record=simulate('proposed',seconds=.1,scenario='traction')
    assert audit_record(record)['passed']
    c=next(c for e in record['events'] for c in e['certification_checks'] if c['accepted'])
    c['failure_upper']=0.
    with pytest.raises(AssertionError):audit_record(record)


def test_zero_step_failure_metrics_keep_the_episode():
    from dual_gtk_drs_racing.experiments import metrics
    r=simulate('passive',seconds=.05,scenario='traction')
    r['frames']=r['frames'][:1]
    r['summary'].update(time=0.,cost=0.,lap_complete=False,constraint_violated=True,termination='constraint_violation')
    m=metrics(r,dict(max_replay_error=0.,passed=True))
    assert m['constraint_violated'] and m['time']==0. and m['cost']==0.


def test_report_counts_violations_separately_from_audit_errors():
    from dual_gtk_drs_racing.experiments import design,metrics
    from dual_gtk_drs_racing.paper_report import aggregate
    r=simulate('passive',seconds=.05,scenario='traction');m=metrics(r,audit_record(r))
    cases=design(1,grid=False,weights=False)
    rows=[dict(case=c,status='ok',metrics=dict(m,method=c['method'],constraint_violated=c['method']=='weighted')) for c in cases]
    _,safety,_=aggregate(rows)
    weighted=next(s for s in safety if s['method']=='weighted')
    assert weighted['constraint_violations']==1 and weighted['errors']==0
    assert weighted['unsafe_or_unresolved_upper95']==1.
