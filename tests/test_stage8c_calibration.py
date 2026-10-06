"""Calibration role separation, conformal order statistics, and field sampling units."""
import json
from pathlib import Path
import numpy as np
import pytest
from stage8_uq.calibration import (residual_scores, conformal_factors, wilson_interval,
    band_diagnostics, reliability_rows, grouped_summary)
from stage8_uq.calibration_data import (SEEDS, config, identities, require_disjoint,
    require_unused_seeds, verified_archive)

TIMES=np.arange(11)/10


def test_seed_uniqueness_and_original_ranges():
    assert SEEDS=={'calibration':8302,'audit':8303}
    require_unused_seeds({2026,4100,4101,4110,4111,4112,4113,4120})
    for seed in SEEDS.values():
        with pytest.raises(ValueError):require_unused_seeds({seed})
    for role in SEEDS:
        c=config(role)
        assert (c.num_trajectories,c.grid_size,c.num_snapshots,c.t_final)==(256,64,11,1.)
        assert c.cx_range==c.cy_range==(-1.,1.) and c.nu_range==(.005,.05)


def test_archive_disjointness_uses_physical_parameters_not_local_ids():
    data={k:np.arange(6).reshape(2,3) for k in ('coefficients','blob_counts','blob_mask','blob_centers','blob_widths','blob_amplitudes')}
    first=identities(data)
    require_disjoint(first,set())
    with pytest.raises(ValueError):require_disjoint(first,set(first))
    with pytest.raises(ValueError):require_disjoint([first[0],first[0]],set())
    changed={k:v.copy() for k,v in data.items()};changed['coefficients'][0,0]=99
    assert identities(changed)[0]!=first[0]


@pytest.mark.parametrize('role',['train','validation','test','fresh_id','high_nu','high_velocity','historical_id'])
def test_no_historical_or_ood_archive_roles(role):
    with pytest.raises(ValueError):config(role)
    with pytest.raises(ValueError):verified_archive(role)


def test_standardized_residual_max_and_t0_exclusion():
    mean=np.zeros((3,11,2,2));truth=np.ones_like(mean)*2;spread=np.ones_like(mean)*.5
    truth[:,0]=99999
    truth[1,4,0,1]=5
    result=residual_scores(truth,mean,spread,TIMES,1e-12)
    assert result['scores'].shape==(3,10)
    assert result['scores'][1,3]==10
    assert result['scores'][0,0]==4
    assert result['scores'].max()==10
    assert result['floor_cells_by_horizon'].sum()==0


def test_floor_accounting_and_raw_band_distinction():
    mean=np.zeros((2,11,2,2));truth=np.ones_like(mean);spread=np.ones_like(mean)
    spread[:,0]=0;spread[0,1,0,0]=0;spread[0,1,0,1]=.01
    result=residual_scores(truth,mean,spread,TIMES,.1)
    assert result['scores'][0,0]==10
    assert result['floor_cells_by_horizon'].tolist()==[2]+[0]*9
    assert result['floor_fields_by_horizon'].tolist()==[1]+[0]*9
    # Floor-based scores alone do not guarantee raw-spread bands at zero-spread cells.
    result=band_diagnostics(truth[:,1:],mean[:,1:],spread[:,1:],np.full(10,10.),.1)
    assert not result['field_covered'][0,0]


def test_higher_order_statistic_and_monotonicity():
    scores=np.tile(np.arange(1,257.)[:,None],(1,10))
    result=conformal_factors(scores,'calibration')
    assert result['k'].tolist()==[129,180,206,232,245]
    assert result['q'][0].tolist()==[129,180,206,232,245]
    assert np.all(np.diff(result['q'],axis=1)>=0)
    assert not result['clipped'].any()
    small=conformal_factors(np.ones((2,10)),'calibration')
    assert small['clipped'][-1] and small['k'][-1]==2


def test_audit_cannot_fit_factors():
    with pytest.raises(ValueError,match='Only calibration'):
        conformal_factors(np.ones((256,10)),'audit')


def test_coverage_is_one_binary_event_per_field_and_width():
    mean=np.full((2,10,2,2),4.);spread=np.ones_like(mean)*.5
    truth=mean.copy();truth[0,0,0,0]=6
    result=band_diagnostics(truth,mean,spread,np.ones(10),1e-12)
    assert result['field_covered'].shape==(2,10)
    assert not result['field_covered'][0,0] and result['field_covered'][1,0]
    assert result['pointwise_fraction_secondary'][0,0]==.75
    np.testing.assert_allclose(result['mean_physical_width'],1.)
    np.testing.assert_allclose(result['W_rel'],.25)
    truth[0,0,0,0]=4.5
    assert band_diagnostics(truth,mean,spread,np.ones(10),1e-12)['field_covered'].all()


def test_reliability_n_is_trajectories_not_cells_or_horizons():
    diagnostic={key:np.ones((8,10,5)) for key in ('field_covered','mean_physical_width','rms_physical_width','W_rel','pointwise_fraction_secondary')}
    diagnostic['field_covered'][0,0,:]=0
    rows=reliability_rows(diagnostic,TIMES[1:])
    assert len(rows)==50
    assert rows[0]['n_trajectories']==8 and rows[0]['covered_fields']==7
    assert rows[0]['simultaneous_coverage']==.875
    assert rows[0]['coverage_error']==.375
    np.testing.assert_allclose([rows[0]['coverage_ci95_low'],rows[0]['coverage_ci95_high']],wilson_interval(7,8))
    assert rows[0]['mean_pointwise_fraction_secondary']==1


def test_grouped_bootstrap_preserves_trajectory_horizons():
    diagnostic={key:np.ones((8,10,5)) for key in ('field_covered','W_rel','pointwise_fraction_secondary')}
    diagnostic['field_covered'][:4]=0
    a,indices=grouped_summary(diagnostic,replicates=100)
    b,repeat=grouped_summary(diagnostic,replicates=100)
    assert a==b and np.array_equal(indices,repeat)
    expected=np.quantile(diagnostic['field_covered'][:,:,0].mean(1)[indices].mean(1),[.025,.975])
    np.testing.assert_allclose(a[0]['trajectory_bootstrap_ci95'],expected)
    assert a[0]['mean_horizon_coverage']==.5


def test_fit_does_not_load_audit(tmp_path,monkeypatch):
    import examples.calibrate_stage8c as runner
    monkeypatch.chdir(tmp_path)
    Path('docs/results').mkdir(parents=True)
    Path('docs/results/stage8c_freeze.json').write_text('{}')
    root=tmp_path/'run';(root/'calibration_predictions').mkdir(parents=True)
    (root/'calibration_predictions/manifest.json').write_text('{}')
    monkeypatch.setattr(runner,'ROOT',root)
    calls=[]
    def cal_only(role):
        assert role=='calibration';calls.append(role)
        return {'fields':np.ones((8,11,2,2)),'times':TIMES},{'sha256':'calibration-only'}
    monkeypatch.setattr(runner,'verified_archive',cal_only)
    def predictions(role):
        assert role=='calibration'
        return {'mean':np.zeros((8,11,2,2)),'spread':np.ones((8,11,2,2))},{}
    monkeypatch.setattr(runner,'read_predictions',predictions)
    runner.fit({'epsilon_spread':1e-12})
    result=json.loads((root/'factors.json').read_text())
    assert calls==['calibration'] and not result['audit_used_for_fit']
    assert np.array_equal(result['q'],np.ones((10,5)))
    assert not json.loads((root/'factor_freeze_receipt.json').read_text())['audit_predictions_existed_at_freeze']


def test_no_normalization_refit_and_frozen_provenance(monkeypatch):
    if not Path('runs/stage8/ensemble_manifest.json').exists():pytest.skip('Ignored frozen checkpoints unavailable')
    from neural_operator.normalization import Normalization
    import examples.calibrate_stage8c as runner
    monkeypatch.setattr(Normalization,'fit',lambda *a:pytest.fail('Normalization refit'))
    frozen,models,norm=runner.verify_frozen()
    assert len(models)==5 and norm.to_dict()==frozen['normalization']


def test_frozen_stage8b_tampering_refused(tmp_path,monkeypatch):
    import examples.calibrate_stage8c as runner
    frozen=json.loads(runner.FREEZE.read_text());frozen['stage8b_summary_sha256']='0'*64
    p=tmp_path/'freeze.json';p.write_text(json.dumps(frozen));monkeypatch.setattr(runner,'FREEZE',p)
    with pytest.raises(ValueError):runner.verify_frozen()


def test_member_specific_feedback_with_fixed_coefficients():
    torch=pytest.importorskip('torch')
    from neural_operator.normalization import Normalization
    from stage8_uq.ensemble import rollout_members
    class Scale(torch.nn.Module):
        def __init__(self,factor):super().__init__();self.factor=factor;self.conditions=[]
        def forward(self,x):self.conditions.append(x[:,1:].clone());return self.factor*x[:,:1]
    norm=Normalization(2.,3.,[0,0,0],[1,1,1])
    models=[Scale(v) for v in [1.,2.,1.5,1.2,.9]]
    fields=np.full((1,11,4,4),5.);coeff=np.array([[.2,-.3,.01]])
    ar=rollout_members(models,norm,fields,coeff,TIMES)
    np.testing.assert_array_equal(ar['members'][1,0,:,0,0,0],2+3*2.**np.arange(11))
    for model in models:
        for c in model.conditions:np.testing.assert_allclose(c[:,:,0,0],coeff)


def test_real_new_archive_identities_and_disjointness():
    if not Path('data/stage8/manifest.json').exists():pytest.skip('Ignored new archives unavailable')
    manifest=json.loads(Path('data/stage8/manifest.json').read_text())
    previous=set(v for entry in manifest['historical_identity_inventory'].values() for v in entry['input_fingerprints'])
    for role in ('calibration','audit'):
        data,entry=verified_archive(role)
        require_disjoint(identities(data),previous)
        previous.update(entry['input_fingerprints'])
        assert data['fields'].shape==(256,11,64,64)


def test_frozen_factor_tamper_refused(tmp_path,monkeypatch):
    import examples.calibrate_stage8c as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'factors.json').write_text('{}')
    (tmp_path/'factor_freeze_receipt.json').write_text(json.dumps({'factors_sha256':'0'*64}))
    with pytest.raises(ValueError):runner.verified_factors()


def test_audit_inference_requires_frozen_factors(tmp_path,monkeypatch):
    import examples.calibrate_stage8c as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verified_archive',lambda *a:pytest.fail('Audit loaded before fit'))
    with pytest.raises(FileNotFoundError):runner.predict('audit',[],None)


def test_real_factors_equal_calibration_only_order_statistics():
    root=Path('runs/stage8/calibration')
    if not (root/'factors.json').exists():pytest.skip('Ignored frozen factors unavailable')
    factors=json.loads((root/'factors.json').read_text())
    with np.load(root/'calibration_scores.npz') as archive:
        expected=conformal_factors(archive['scores'],'calibration')
    np.testing.assert_array_equal(factors['q'],expected['q'])
    assert factors['n_calibration']==256 and not factors['audit_used_for_fit']
