import numpy as np
import pytest
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record
from dual_gtk_drs_racing.stress_study import design


def test_stress_design_is_paired_and_cohorts_use_independent_streams():
    a=design(5,3);assert a==design(5,3) and len(a)==34
    for group in ('launch','lap'):
        for i in range(3):
            rows=[c for c in a if c['group']==group and c['trial']==i]
            assert len({c['true_mu'] for c in rows})==len({c['seed'] for c in rows})==1
            assert all(.76<=c['true_mu']<=1.05 for c in rows)
    assert a[0]['true_mu']!=next(c['true_mu'] for c in a if c['group']=='lap')


def test_nonzero_accepted_risk_can_produce_a_real_tire_constraint_failure():
    from dual_gtk_drs_racing.track import Track
    from dual_gtk_drs_racing.control_safety import ControlCertifier
    from dual_gtk_drs_racing.belief import FrictionSet
    from dual_gtk_drs_racing.dynamics import plant_step
    tr=Track('technical');k=tr.kappa(0.)
    x=np.array([0.,0.,0.,8.,0.,8*k,np.arctan(3.05*k)])
    controls=np.array([[2*1678.9573770491806*(.766+.012),0.,0.]])
    b=FrictionSet(.76,1.05)
    tight=ControlCertifier(tr,traction=True,epsilon=.01).check_plan(x,b,controls,x[None,:])
    relaxed=ControlCertifier(tr,traction=True,epsilon=.05).check_plan(x,b,controls,x[None,:])
    assert not tight.accepted and relaxed.accepted and relaxed.failures>0
    assert relaxed.failure_upper<=.05
    assert min(plant_step(x,.762,controls[0],tr.curvature,tr.ds)[3])<0


def test_default_prior_and_explicit_original_prior_are_equivalent():
    a=simulate('proposed',seconds=.65,scenario='traction')
    b=simulate('proposed',seconds=.65,scenario='traction',initial_mu=(.45,1.05))
    for r in (a,b):
        for e in r['events']:
            for d in e.get('planner_diagnostics',{}).values():d.pop('solve_seconds')
    assert a['frames']==b['frames'] and a['events']==b['events']


@pytest.mark.parametrize('prior,mu', [((.9,.8),.85),((0.,1.),.85),((.76,1.05),.75),((.76,np.inf),.85)])
def test_incorrect_prior_is_rejected(prior,mu):
    with pytest.raises(ValueError):simulate(seconds=.05,true_mu=mu,initial_mu=prior)
