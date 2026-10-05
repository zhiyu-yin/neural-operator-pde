"""Stage 8B statistical units, fixed definitions, and provenance guards."""
import json
from pathlib import Path
import numpy as np
import pytest
from stage8_uq.awareness import (ranks, correlation, association, prediction_horizons,
    quintiles, top_fifth, detection, early_pairs, spatial_association, analyze)


def test_pearson_known_value_and_degeneracy():
    assert correlation([1,2,3],[1,4,2]) == pytest.approx(np.sqrt(3/28))
    assert correlation([1,1,1],[1,2,3]) is None
    assert correlation([1,2,np.nan],[1,2,3]) is None


def test_spearman_average_ties():
    np.testing.assert_equal(ranks([3,1,1,4]),[3,1.5,1.5,4])
    assert correlation([1,2,3,4],[1,8,27,64],True) == pytest.approx(1)


def test_bootstrap_determinism_and_invalid_count():
    indices=np.random.default_rng(8301).integers(4,size=(100,4))
    x=np.arange(4.)
    assert association(x,x*x,indices)==association(x,x*x,indices)
    result=association(np.ones(4),x,indices)
    assert result['pearson_ci95'] is None and result['pearson_invalid_bootstraps']==100


def test_quintiles_are_exhaustive_disjoint_balanced_with_stable_ties():
    labels=quintiles(np.ones(64))
    assert np.bincount(labels).tolist()==[13,13,13,13,12]
    joined=np.concatenate([np.flatnonzero(labels==q) for q in range(5)])
    assert sorted(joined.tolist())==list(range(64))
    assert len(set(joined))==64
    assert labels[0]==0 and labels[-1]==4


def test_top_twenty_percent_and_discrimination():
    x=np.arange(64.)
    assert top_fifth(x).sum()==13
    assert np.flatnonzero(top_fifth(x)).tolist()==list(range(51,64))
    result=detection(x,x)
    assert result['auroc']==result['precision']==result['recall']==1
    assert detection(-x,x)['auroc']==0
    assert detection(np.ones(64),x)['auroc']==.5
    assert top_fifth(np.arange(8)).sum()==2


def test_time_zero_excluded_and_horizon_specific_correlations():
    x=np.arange(8.)
    score=np.column_stack([np.full(8,999.)]+[x+h*100 for h in range(10)])
    error=np.column_stack([np.full(8,-999.)]+[x if h%2 else -x for h in range(10)])
    result=analyze(score,score,error,np.arange(11)/10,replicates=10)
    assert len(result['by_horizon'])==10
    for h,row in enumerate(result['by_horizon']):
        assert row['time']==pytest.approx((h+1)/10)
        assert row['pearson']==pytest.approx(1 if h%2 else -1)
        assert row['spearman']==pytest.approx(1 if h%2 else -1)
        assert row['n_trajectories']==8
    with pytest.raises(ValueError): prediction_horizons(np.arange(12)/10)


def test_early_warning_alignment():
    assert early_pairs(delta=1)==[(i,i+1) for i in range(9)]
    assert early_pairs(delta=2)==[(i,i+2) for i in range(8)]


def test_spatial_fields_not_pooled():
    x=np.arange(4.).reshape(1,1,2,2)
    error=np.concatenate([x,x+100],axis=0)
    spread=np.concatenate([x,-x+100],axis=0)
    result=spatial_association(error,spread)
    np.testing.assert_allclose(result,[[1],[-1]])
    assert correlation(error,spread,True)>0


def test_physical_error_uses_inverse_normalized_values():
    from stage4_evaluation.diagnostics import physical_diagnostics
    target=np.full((2,11,2,2),5.)
    prediction=np.full_like(target,6.)
    np.testing.assert_allclose(physical_diagnostics(prediction,target,target[:,0])['relative_l2'],.2)


def test_evaluation_roles_and_no_fit(monkeypatch):
    import examples.analyze_stage8b as runner
    from neural_operator.normalization import Normalization
    monkeypatch.setattr(Normalization,'fit',lambda *a:pytest.fail('Refitting prohibited'))
    monkeypatch.setattr(runner,'load_dataset',lambda p:{'fields':np.zeros((64,11,2,2)),'coefficients':np.zeros((64,3)),'times':np.arange(11)/10})
    frozen={'split_ids':{'test':[51,16,49,17,40,47,22,28]}}
    assert len(runner.load_evaluation('fresh_id',frozen)[3])==64
    assert runner.load_evaluation('historical_id',frozen)[3].tolist()==frozen['split_ids']['test']
    for role in ['train','validation','high_nu','high_velocity','calibration']:
        with pytest.raises(ValueError): runner.load_evaluation(role,frozen)
    source=Path(runner.__file__).read_text()
    assert 'calibration_fitted\':False' in source
    assert '.fit(' not in source and 'train_member(' not in source


def test_frozen_real_inputs_and_archive_identity():
    if not Path('runs/stage8/ensemble_manifest.json').exists(): pytest.skip('Private frozen artifacts unavailable')
    from examples.analyze_stage8b import verified_inputs
    freeze,frozen,models,norm=verified_inputs()
    assert len(models)==5
    assert freeze['fresh_id_sha256']=='52292b4e91eb9aeb92277f429f0d126ba2a43e76f8f9cda6cd32c2347d12d9e2'
    assert norm.to_dict()==frozen['normalization']


def test_changed_manifest_refused(tmp_path,monkeypatch):
    import examples.analyze_stage8b as runner
    freeze=json.loads(runner.FREEZE.read_text()); freeze['ensemble_manifest_sha256']='0'*64
    path=tmp_path/'freeze.json'; path.write_text(json.dumps(freeze)); monkeypatch.setattr(runner,'FREEZE',path)
    if not Path('runs/stage8/ensemble_manifest.json').exists(): pytest.skip('Private artifacts unavailable')
    with pytest.raises(ValueError): runner.verified_inputs()


def test_demeaned_bootstrap_recenters_each_trajectory_draw():
    x=np.array([[1,10],[2,30],[4,20],[6,50.]])
    y=np.array([[2,20],[4,10],[1,40],[8,30.]])
    indices=np.array([[0,0,1,3],[1,2,3,3]])
    result=association(x,y,indices,demean=True)
    expected=[correlation(x[i]-x[i].mean(0),y[i]-y[i].mean(0)) for i in indices]
    np.testing.assert_allclose(result['pearson_ci95'],np.quantile(expected,[.025,.975]))


def test_analysis_seed_reproducible():
    rng=np.random.default_rng(1)
    x=rng.random((8,11)); y=rng.random((8,11))
    a=analyze(x,x,y,np.arange(11)/10,replicates=20)
    b=analyze(x,x,y,np.arange(11)/10,replicates=20)
    assert a==b


def test_fresh_archive_tampering_is_refused(tmp_path,monkeypatch):
    import examples.analyze_stage8b as runner
    if not Path('runs/stage8/ensemble_manifest.json').exists(): pytest.skip('Private artifacts unavailable')
    fake=tmp_path/'fresh_id.npz'; fake.write_bytes(b'changed archive')
    monkeypatch.setitem(runner.ARCHIVES,'fresh_id',fake)
    # Keep the actual identity guard, avoiding repeated expensive checkpoint loads.
    monkeypatch.setattr(runner,'frozen_protocol',lambda:{})
    frozen=json.loads(runner.FREEZE.read_text())
    monkeypatch.setattr(runner,'load_member',lambda folder,protocol:(None,None,frozen['members'][int(folder.name[-1])]))
    with pytest.raises(ValueError): runner.verified_inputs()


def test_changed_member_manifest_refused(tmp_path,monkeypatch):
    import examples.analyze_stage8b as runner
    if not Path('runs/stage8/ensemble_manifest.json').exists(): pytest.skip('Private artifacts unavailable')
    frozen=json.loads(runner.FREEZE.read_text()); frozen['member_manifest_sha256']['0']='0'*64
    path=tmp_path/'freeze.json';path.write_text(json.dumps(frozen));monkeypatch.setattr(runner,'FREEZE',path)
    monkeypatch.setattr(runner,'frozen_protocol',lambda:{})
    with pytest.raises(ValueError): runner.verified_inputs()
