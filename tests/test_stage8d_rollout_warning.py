"""Frozen roles, conditional associations, trajectory units, and rare new entrants."""
import json
from pathlib import Path
import numpy as np
import pytest
from stage8_uq.awareness import ranks,correlation
from stage8_uq.rollout_warning import (residualize,partial_correlation,partial_association,
                                       aligned,increments,new_failure,distribution,analyze)


def test_horizon_lag_and_increment_alignment():
    error=np.arange(30.).reshape(3,10);u=error+100
    for lag in (1,2):
        for h in range(10-lag):
            score,current,future,delta=aligned(error,u,h,lag)
            np.testing.assert_equal(score,u[:,h]);np.testing.assert_equal(current,error[:,h])
            np.testing.assert_equal(future,error[:,h+lag]);np.testing.assert_equal(delta,lag)
    with pytest.raises(ValueError):aligned(error,u,9,1)
    with pytest.raises(ValueError):aligned(error,u,0,3)
    with pytest.raises(ValueError):aligned(np.zeros((3,11)),np.zeros((3,11)),0,1)


def test_delta_uncertainty_and_error_can_be_negative():
    error=np.tile(np.arange(10.)**2,(5,1));u=-error
    np.testing.assert_equal(increments(error),np.tile(np.arange(1,18,2),(5,1)))
    np.testing.assert_equal(increments(u,2),u[:,2:]-u[:,:-2])
    assert np.all(increments(u)<0)


def test_residualized_pearson_matches_independent_least_squares():
    c=np.array([1,4,2,6,3,8,9,5.]);x=np.array([2,3,6,1,8,9,4,7.]);y=np.array([5,6,2,8,4,1,9,7.])
    design=np.column_stack([np.ones(len(c)),c])
    rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    np.testing.assert_allclose(residualize(x,c),rx,atol=1e-12)
    assert partial_correlation(x,y,c)==pytest.approx(np.corrcoef(rx,ry)[0,1])


def test_partial_rank_ranks_before_projection_with_ties():
    c=np.array([1,1,2,3,4,5,7,8.]);x=np.array([8,2,3,4,4,9,7,6.]);y=np.array([1,2,6,4,5,4,3,9.])
    rc,rx,ry=ranks(c),ranks(x),ranks(y)
    design=np.column_stack([np.ones(8),rc])
    a=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
    b=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
    assert partial_correlation(x,y,c,True)==pytest.approx(np.corrcoef(a,b)[0,1])
    assert not np.isclose(correlation(a,b,True),partial_correlation(x,y,c,True))


def test_degenerate_partial_not_forced():
    x=np.arange(8.)
    assert partial_correlation(x,x,np.ones(8)) is None
    assert partial_correlation(2*x+3,x,x) is None
    assert partial_correlation(x,x,np.full(8,np.nan)) is None


def test_projection_removes_shared_current_error_confounding():
    c=np.array([-3,-3,-1,-1,1,1,3,3.])
    residual=np.array([-1,1,-1,1,-1,1,-1,1.])
    x=c+residual;y=10*c-residual
    assert correlation(x,y)>.8
    assert partial_correlation(x,y,c)==pytest.approx(-1.)


def test_bootstrap_determinism_and_reprojection():
    rng=np.random.default_rng(3);x,y,c=rng.normal(size=(3,12))
    indices=np.random.default_rng(8401).integers(12,size=(30,12))
    a=partial_association(x,y,c,indices);b=partial_association(x,y,c,indices)
    assert a==b
    for name,rank in [('pearson',False),('spearman',True)]:
        values=[partial_correlation(x[i],y[i],c[i],rank) for i in indices]
        np.testing.assert_allclose(a[name+'_ci95'],np.quantile([v for v in values if v is not None],[.025,.975]))


def test_new_failure_excludes_current_high_errors_and_uses_eligible_top_fifth():
    current=np.arange(50.);future=current.copy();future[:5]=np.arange(100,105)
    score=-current.copy();score[40:]=1000  # Already-high cases must not enter the detector.
    result=new_failure(score,current,future)
    assert result['eligible_count']==40 and result['excluded_already_high_count']==10
    assert result['event_indices']==list(range(5))
    assert result['event_count']==5 and result['non_event_count']==35
    assert result['selected_count']==8 and result['hits']==5
    assert result['auroc']==1 and result['precision']==5/8 and result['recall']==1


def test_rare_events_and_no_events_are_reported_not_scored():
    current=np.arange(25.);future=current.copy();future[0]=100
    result=new_failure(current,current,future)
    assert result['event_count']==1 and result['missed_event_indices']==[0]
    assert result['auroc'] is result['precision'] is result['recall'] is None
    none=new_failure(current,current,current)
    assert none['event_count']==0 and none['auroc'] is None


def test_saturation_quantiles_and_gap():
    result=distribution(np.arange(11.))
    assert result=={'q10':1.,'median':5.,'q90':9.,'q90_minus_q10':8.,'gap_over_median':1.6}
    assert distribution(np.zeros(8))['gap_over_median'] is None


def test_analysis_time_zero_absent_and_width_adds_no_fixed_horizon_information():
    rng=np.random.default_rng(4);error=rng.random((8,10));u=rng.random((8,10))
    q=np.arange(1,11);covered=error<.7
    result,indices=analyze(error,u,u,u*2*q,covered,replicates=5)
    assert indices.shape==(5,8)
    assert len(result['early_warning'])==17 and len(result['growth_by_horizon'])==9
    assert result['by_horizon'][0]['time']==.1
    for row in result['early_warning']:
        assert row['target_time']<=1 and row['target_time']>row['time']
        assert row['width_future']['pearson']==pytest.approx(row['uncertainty_future']['pearson'])
        assert row['width_future']['spearman']==pytest.approx(row['uncertainty_future']['spearman'])


@pytest.mark.parametrize('role',['calibration','train','validation','historical_id','high_nu','velocity_pp'])
def test_forbidden_roles_rejected_before_archive_access(role,monkeypatch):
    from stage8_uq.rollout_warning_io import load_role
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Forbidden archive opened'))
    with pytest.raises(ValueError):load_role(role,{}, {})


def test_no_refit_no_generation_and_real_frozen_provenance(monkeypatch):
    if not Path('runs/stage8/calibration/factors.json').exists():pytest.skip('Ignored frozen artifacts unavailable')
    from stage8_uq import calibration,calibration_data
    from neural_operator.normalization import Normalization
    import dataset_generation
    from stage8_uq.rollout_warning_io import verify_inputs,load_role
    def forbidden(*args,**kwargs):pytest.fail('Fitting or generation invoked')
    monkeypatch.setattr(calibration,'conformal_factors',forbidden)
    monkeypatch.setattr(calibration_data,'generate_archives',forbidden)
    monkeypatch.setattr(dataset_generation,'generate_dataset',forbidden)
    monkeypatch.setattr(Normalization,'fit',forbidden)
    frozen,c,b,norm=verify_inputs()
    allowed={'runs/stage8/calibration/audit_predictions/error.npy','runs/stage8/calibration/audit_predictions/U_rel.npy',
             'runs/stage8/calibration/audit_predictions/U_rms.npy','runs/stage8/calibration/audit_diagnostics.npz',
             'runs/stage8/error_awareness/fresh_id/raw_predictions.npz'}
    original=np.load
    def only_outputs(path,*args,**kwargs):
        assert str(path) in allowed
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',only_outputs)
    for role,n in [('audit',256),('fresh_id',64)]:
        arrays,origin=load_role(role,c,b)
        assert arrays['error'].shape==(n,10)
        assert arrays['times'][0]==.1 and arrays['times'][-1]==1


def test_stage8c_factor_hash_tamper_refused(tmp_path,monkeypatch):
    from stage8_uq import rollout_warning_io as io
    if not Path('runs/stage8/calibration/factors.json').exists():pytest.skip('Ignored frozen artifacts unavailable')
    frozen=json.loads(io.FREEZE.read_text());frozen['factors_sha256']='0'*64
    path=tmp_path/'freeze.json';path.write_text(json.dumps(frozen));monkeypatch.setattr(io,'FREEZE',path)
    with pytest.raises(ValueError):io.verify_inputs()


def test_example_selection_is_fixed_including_missed_case():
    from stage8_uq.rollout_warning_plots import select_examples
    error=np.tile(np.arange(10.),(5,1));error[2,-1]=100
    u=np.zeros_like(error);u[3,-1]=10
    result={'early_warning':[{'time':.1,'target_time':.2,'lag':1,'new_failure':{'missed_event_indices':[4,1],'event_count':2,'status':'rare'}}]}
    examples=select_examples({'error':error,'U_rel':u},result)
    assert [e['trajectory_id'] for e in examples]==[0,2,3,2,1]
    assert examples[-1]['source_time']==.1
