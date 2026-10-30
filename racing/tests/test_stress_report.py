import numpy as np
from dual_gtk_drs_racing.stress_report import aggregate,interval
from dual_gtk_drs_racing.stress_diagnostics import rollout_outcomes
from dual_gtk_drs_racing.safety import sampled_rollouts
from dual_gtk_drs_racing.track import Track
from dual_gtk_drs_racing.model import commands


def test_diagnostic_separates_terminal_failure_from_physical_failure():
    tr=Track('technical');x=np.array([0.,0.,0.,8.,0.,8*tr.kappa(0),np.arctan(3.05*tr.kappa(0))])
    refs=np.array([commands(j*.05,1.8,42.44,0.,8.) for j in range(116)])
    mus=np.linspace(.76,1.05,51)
    t,kind=rollout_outcomes(x,mus,refs,tr.curvature,tr.ds)
    failures,*_=sampled_rollouts(x,mus,refs,tr.curvature,tr.ds,np.max(np.abs(tr.curvature)),True,len(mus))
    assert (kind>0).sum()==failures and kind[0]==2 and t[0]==0
    x[1]=2.
    t,kind=rollout_outcomes(x,np.array([.8]),np.empty((0,3)),tr.curvature,tr.ds)
    assert kind[0]==3
    x[1]=6.
    t,kind=rollout_outcomes(x,np.array([.8]),np.empty((0,3)),tr.curvature,tr.ds)
    assert kind[0]==1


def test_all_trials_stay_in_stress_safety_denominator():
    base=dict(certificate_unavailable=False,lap_complete=False,termination='timeout',time=8.,containment_violations=0,minimum_tire_margin=.1)
    rows=[]
    for i in range(4):
        row=dict(case=dict(group='launch',scenario='traction_relaxed',method='proposed',seconds=8.),status='ok')
        row['metrics']=dict(base,violation=dict(kind='tire_force') if i==0 else None)
        if i==0:row['metrics'].update(termination='constraint_violation',time=0.,minimum_tire_margin=-.01)
        if i==3:row.pop('metrics');row['status']='error'
        rows.append(row)
    a=aggregate(rows)[0]
    assert a['trials']==4 and a['violations']==1 and a['audit_errors']==1
    assert a['violation_rate']==.25 and a['horizon_reached']==2
    assert a['mean_completed_lap'] is None
    assert a['unsafe_or_unresolved_upper95']>a['violation_rate']


def test_two_sided_interval_keeps_zero_and_all_failure_edges():
    assert interval(0,100)[0]==0 and interval(100,100)[1]==1
    lo,hi=interval(25,1000)
    assert lo<.025<hi
