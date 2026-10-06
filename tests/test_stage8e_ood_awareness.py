"""Frozen OOD roles, independent trajectory bootstrap, and unmodified ID bands."""
import json
from pathlib import Path
import numpy as np
import pytest
from stage8_uq.ood_awareness import (bootstrap_indices,regime_effect,auroc,category_ranks,
                                    scalar_and_bands,analyze_regime)
from stage8_uq.ood_awareness_io import QUADRANTS,REGIMES,aggregate_quadrants


def metrics(n,role='fresh_id'):
    e=np.tile(np.linspace(.1,1,n)[:,None],(1,10))
    return {'error':e,'U_rel':2*e,'U_rms':3*e,'W_rel':np.ones((n,10,5)),
            'covered':np.ones((n,10,5),bool),'qualified_ids':np.asarray([f'{role}:{i}' for i in range(n)])}


def test_regime_labels_and_no_combined_archive_or_calibration_role():
    assert REGIMES==('fresh_id','historical_id','high_nu','velocity_pp','velocity_pn','velocity_np','velocity_nn')
    from stage8_uq.ood_awareness_io import load_archive,load_outputs
    for role in ('calibration','audit','train','validation','high_velocity','resolution_128'):
        with pytest.raises(ValueError):load_archive(role,{})
        with pytest.raises(ValueError):load_outputs(role,{},{})


def test_deterministic_independent_draws_and_fixed_quadrant_mixture():
    sizes={'fresh_id':64,'high_nu':64,**{q:16 for q in QUADRANTS}}
    a=bootstrap_indices(sizes,replicates=20);b=bootstrap_indices(sizes,replicates=20)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert not np.array_equal(a['fresh_id'],a['high_nu'])
    for block,q in enumerate(QUADRANTS):
        np.testing.assert_array_equal(a['high_velocity'][:,16*block:16*(block+1)],a[q]+16*block)


def test_regime_effect_independent_and_baseline_identity():
    control=np.arange(1,5.);values=control+4
    di=np.array([[0,1,2,3],[0,0,1,1],[2,2,3,3]])
    dj=np.array([[3,2,1,0],[3,3,2,2],[0,0,1,1]])
    result=regime_effect(values,control,di,dj)
    assert result['mean_difference']==4 and result['median_difference']==4
    assert result['mean_ratio']==pytest.approx(values.mean()/control.mean())
    expected=values[di].mean(1)-control[dj].mean(1)
    np.testing.assert_allclose(result['mean_difference_ci95'],np.quantile(expected,[.025,.975]))
    baseline=regime_effect(control,control,di,di)
    assert baseline['mean_difference']==0 and baseline['mean_difference_ci95']==[0.,0.]
    assert baseline['mean_ratio_ci95']==[1.,1.]
    bad=regime_effect(values,np.zeros(4),di,dj)
    assert bad['mean_ratio'] is None and bad['mean_ratio_ci95'] is None and bad['invalid_ratio_bootstraps']==3


def test_known_ood_auroc_and_direction_ties():
    assert auroc([0,1],[2,3])==1
    assert auroc([2,3],[0,1])==0
    assert auroc([1,1],[1,1])==.5
    assert auroc([0,2],[1,3])==.75
    assert auroc([],[]) is None


def test_fixed_quartile_categories_rank_entire_pool():
    er,ur,labels=category_ranks([0,1,2,3],[0,1,2,3],[4,5,6,7],[4,7,0,6])
    assert len(er)==len(ur)==len(labels)==8
    assert np.all((er>0)&(er<1))
    assert er[-1]==.9375
    assert labels[-2]=='high_error_modest_uncertainty'
    assert labels[-1]=='high_error_high_uncertainty'
    _,_,labels=category_ranks([0,1,2,3],[0,1,2,3],[-2,-1,4,5],[10,11,0,0])
    assert labels[4]==labels[5]=='low_error_high_uncertainty'


def test_quadrant_aggregation_never_duplicates_local_ids():
    data={q:metrics(16,q) for q in QUADRANTS}
    result=aggregate_quadrants(data)
    assert len(result['error'])==64 and len(set(result['qualified_ids']))==64
    np.testing.assert_array_equal(result['qualified_ids'][:16],data['velocity_pp']['qualified_ids'])
    with pytest.raises(ValueError):aggregate_quadrants({'velocity_pp':data['velocity_pp']})
    data['velocity_pn']['qualified_ids']=data['velocity_pp']['qualified_ids'].copy()
    with pytest.raises(ValueError):aggregate_quadrants(data)


def test_fixed_band_application_coverage_and_time_zero_exclusion(monkeypatch):
    import stage8_uq.calibration as calibration
    monkeypatch.setattr(calibration,'conformal_factors',lambda *a:pytest.fail('Calibration refit'))
    mean=np.full((2,11,2,2),4.);spread=np.ones_like(mean);truth=mean.copy()
    truth[0,0]=1e9;truth[0,1,0,0]=5.5
    arrays={'mean':mean,'spread':spread,'truth':truth,'error':np.zeros((2,11)),
            'U_rel':np.full((2,11),.25),'U_rms':np.ones((2,11)),
            'times':np.arange(11)/10,'qualified_ids':np.array(['high_nu:0','high_nu:1'])}
    q=np.tile(np.arange(1,6.),(10,1));before=q.copy()
    result=scalar_and_bands(arrays,q,1e-12)
    assert result['covered'].shape==(2,10,5)
    assert not result['covered'][0,0,0] and result['covered'][0,0,1]
    np.testing.assert_allclose(result['W_rel'][0,0],np.arange(1,6)/2)
    np.testing.assert_array_equal(q,before)


def test_horizon_specific_correlations_and_category_conservation():
    data=metrics(16,'high_nu');control=metrics(16)
    data['U_rel'][:,1]=-data['U_rel'][:,1]
    di=bootstrap_indices({'fresh_id':16,'high_nu':16},replicates=5)
    result,cases=analyze_regime('high_nu',data,control,di['high_nu'],di['fresh_id'])
    assert len(result['by_horizon'])==10 and len(result['coverage'])==50
    assert result['by_horizon'][0]['within_regime']['pearson']==pytest.approx(1)
    assert result['by_horizon'][1]['within_regime']['spearman']==pytest.approx(-1)
    assert cases['pooled_error_rank'].shape==(32,10)
    for row in result['failure_categories']:assert sum(row['counts'].values())==16


def test_fixed_spatial_selection_retains_independent_extrema():
    from stage8_uq.ood_awareness_plots import select_examples
    velocity=aggregate_quadrants({q:metrics(16,q) for q in QUADRANTS})
    velocity['error'][20,-1]=100;velocity['U_rel'][50,-1]=100
    selected=select_examples({'high_velocity':velocity})
    assert [v for _,v in selected]==['fresh_id:0','high_nu:0','velocity_pp:0','velocity_pn:4','velocity_nn:2']


def test_actual_frozen_provenance_and_no_fit(monkeypatch):
    if not Path('runs/stage8/calibration/factors.json').exists():pytest.skip('Ignored frozen artifacts unavailable')
    from stage8_uq import calibration,calibration_data
    from neural_operator.normalization import Normalization
    from stage8_uq.ood_awareness_io import verify_inputs
    def forbidden(*a,**k):pytest.fail('Fitting or generation is forbidden')
    monkeypatch.setattr(calibration,'conformal_factors',forbidden)
    monkeypatch.setattr(calibration_data,'generate_archives',forbidden)
    monkeypatch.setattr(Normalization,'fit',forbidden)
    frozen,c,b,models,norm=verify_inputs()
    assert len(models)==5
    assert norm.to_dict()==json.loads(Path('docs/results/stage8c_freeze.json').read_text())['normalization']
    assert not c['factors']['audit_used_for_fit']


def test_exact_ood_archive_tamper_refused(tmp_path,monkeypatch):
    from stage8_uq.ood_awareness_io import load_archive
    frozen=json.loads(Path('docs/results/stage8e_freeze.json').read_text())
    fake=tmp_path/'high_nu.npz';fake.write_bytes(b'changed')
    frozen['archives']['high_nu']['path']=str(fake)
    with pytest.raises(ValueError):load_archive('high_nu',frozen)


def test_stage8d_definitions_tamper_refused(tmp_path,monkeypatch):
    from stage8_uq import ood_awareness_io as io
    frozen=json.loads(io.FREEZE.read_text());frozen['protected_tracked_sha256']['stage8_uq/rollout_warning.py']='0'*64
    path=tmp_path/'freeze.json';path.write_text(json.dumps(frozen));monkeypatch.setattr(io,'FREEZE',path)
    with pytest.raises(ValueError):io.verify_inputs()


def test_no_ood_fit_or_generation_calls_in_stage8e_source():
    import ast
    forbidden={'fit','conformal_factors','generate_dataset','generate_archives','train_member','full_training'}
    for path in ['examples/evaluate_stage8e.py','stage8_uq/ood_awareness_io.py','stage8_uq/ood_awareness.py']:
        calls=[node.func for node in ast.walk(ast.parse(Path(path).read_text())) if isinstance(node,ast.Call)]
        names={node.id if isinstance(node,ast.Name) else node.attr if isinstance(node,ast.Attribute) else '' for node in calls}
        assert not forbidden.intersection(names)
