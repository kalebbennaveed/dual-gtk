import json
import numpy as np
import pytest
from dual_gtk_drs_racing.belief import FrictionSet, observation
from dual_gtk_drs_racing.experiments import design, metrics, load_rows
from dual_gtk_drs_racing.paper_report import failure_upper, bootstrap_mean_ci, aggregate
from dual_gtk_drs_racing.simulation import simulate
from dual_gtk_drs_racing.validation import audit_record


def test_paired_design_uses_independent_friction_and_keeps_endpoints_separate():
    cases=design(100)
    assert len(cases)==470
    draws=[]
    for trial in range(100):
        paired=[c for c in cases if c['group']=='monte_carlo' and c['trial']==trial]
        assert len(paired)==4
        assert len({(c['true_mu'],c['seed'],c['start']) for c in paired})==1
        draws.append(paired[0]['true_mu'])
    assert len(set(draws))==100
    assert all(.45<mu<1.05 for mu in draws)
    assert {c['true_mu'] for c in cases if c['group']=='mu_grid'} >= {.45,1.05}
    assert cases==design(100)
    assert cases!=design(100,seed=8)


def test_exact_failure_bound_is_episode_level():
    assert failure_upper(0,100)==pytest.approx(1-.05**.01)
    assert failure_upper(0,1000)==pytest.approx(1-.05**.001)
    assert failure_upper(1,100)>failure_upper(0,100)
    assert failure_upper(100,100)==1
    with pytest.raises(ValueError):failure_upper(0,0)


def test_paired_bootstrap_preserves_constant_difference():
    assert bootstrap_mean_ci(np.full(100,3.5))==(3.5,3.5)


def test_smid_contains_truth_for_signed_quantized_regressions():
    # Random fixed parameters, both regressor signs, zero excitation, quantizer
    # thresholds and support endpoints. Sensor receives truth only via rates.
    rng=np.random.default_rng(14)
    for truth in np.r_[.45,1.05,rng.uniform(.45,1.05,1000)]:
        belief=FrictionSet()
        for g in np.r_[0.,rng.uniform(-20,20,12)]:
            phi=np.zeros(7);phi[5]=g
            f0=np.zeros(7);f0[5]=rng.uniform(-.5,.5)
            rate=f0+truth*phi
            residual,regressor,bound=observation(rate,phi,f0,rng,'quantized_yaw')
            old=(belief.lower,belief.upper)
            belief.update(residual,regressor,bound)
            assert belief.lower-1e-12<=truth<=belief.upper+1e-12
            assert belief.lower>=old[0] and belief.upper<=old[1]


def test_passive_learning_is_enabled_when_observations_are_informative():
    from dual_gtk_drs_racing.track import Track
    track=Track()
    # Begin in a turn: three seconds on the initial straight has no excitation.
    start=float(np.argmax(np.abs(track.curvature))*track.ds)
    record=simulate('passive',seconds=3.,start=start,scenario='barcelona')
    assert audit_record(record)['passed']
    lo,hi=record['summary']['final_mu']
    assert lo<=.85<=hi and hi-lo<.6-1e-3
    assert record['summary']['informative_commits']==0


def test_metrics_preserve_cost_and_truth_containment():
    record=simulate('proposed',seconds=2.,scenario='traction')
    measured=metrics(record,audit_record(record))
    assert measured['containment_violations']==0
    assert measured['nested_set_violations']==0
    assert sum(measured[k] for k in ('time_cost','speed_cost','lateral_cost','heading_cost','force_cost','steering_cost'))==pytest.approx(measured['cost'])
    record['frames'][-1]['mu']=[.9,1.0]
    with pytest.raises(AssertionError,match='excluded truth'):metrics(record,{'passed':True,'max_replay_error':0.})


def test_missing_episode_prevents_completed_report(tmp_path):
    case=design(1,grid=False,weights=False)[0]
    (tmp_path/'manifest.json').write_text(json.dumps({'cases':[case],'implementation':'abc'}))
    with pytest.raises(ValueError,match='incomplete'):load_rows(tmp_path)


def test_unresolved_runs_remain_in_failure_denominator():
    rows=[dict(case=c,status='error',error='certificate unavailable') for c in design(1,grid=False,weights=False)]
    summary,safety,paired=aggregate(rows)
    assert all(r['trials']==1 and r['errors']==1 and r['unsafe_or_unresolved_upper95']==1. for r in safety)
    assert all(r['n']==0 and r['mean'] is None for r in summary)
    assert all(r['complete_pairs']==0 for r in paired)
