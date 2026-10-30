"""Weight variants may differ in λ, but must retain a paired experiment."""
import json
import pytest
from dual_gtk_drs_racing.web import build


def record(weight=4., mu=.85):
    return dict(implementation='paired-source',seed=7,config=dict(scenario='traction',true_mu=mu,information_weight=weight),
                summary=dict(method='weighted'),scenario=dict(track='technical'),frames=[],events=[])


def test_export_includes_distinct_weights(tmp_path):
    source=tmp_path/'input';source.mkdir()
    for name,weight in [('weighted',4.),('weighted_tuned',.125),('weighted_fast',.25)]:
        (source/f'{name}.json').write_text(json.dumps(record(weight)))
    page=build(source,tmp_path/'web')
    data=json.loads((page.parent/'data.js').read_text().removeprefix('window.RACING_DATA=').strip().removesuffix(';'))
    assert {k:r['config']['information_weight'] for k,r in data['runs'].items()}==dict(weighted=4.,weighted_tuned=.125,weighted_fast=.25)


def test_different_friction_is_not_a_paired_weight_comparison(tmp_path):
    source=tmp_path/'input';source.mkdir()
    (source/'weighted.json').write_text(json.dumps(record()))
    (source/'weighted_tuned.json').write_text(json.dumps(record(.125,.77)))
    with pytest.raises(ValueError,match='different configurations'):
        build(source,tmp_path/'web')
