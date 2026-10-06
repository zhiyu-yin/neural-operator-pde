"""Read only frozen audit/fresh-ID predictions. No generation, prediction, or fitting."""
import json
from pathlib import Path
import subprocess
import numpy as np
from examples.calibrate_stage8c import verify_frozen as verify_stage8c, verified_factors, prediction_manifest
from stage4_evaluation.checkpoint import sha256, verify_hash
from stage8_uq.awareness import prediction_horizons
from stage8_uq.calibration import band_diagnostics

FREEZE=Path('docs/results/stage8d_freeze.json')
ROOT=Path('runs/stage8/rollout_uncertainty')
ROLES=('audit','fresh_id')


def verify_inputs():
    frozen=json.loads(FREEZE.read_text())
    subprocess.run(['git','merge-base','--is-ancestor',frozen['stage8c_revision'],'HEAD'],check=True)
    for path,digest in frozen['protected_tracked_sha256'].items():verify_hash(path,digest)
    _,models,norm=verify_stage8c()
    factors,receipt=verified_factors()
    c=json.loads(Path('docs/results/stage8c_summary.json').read_text())
    b=json.loads(Path('docs/results/stage8b_summary.json').read_text())
    if len(models)!=5 or factors!=c['factors'] or receipt!=c['factor_freeze_receipt']:
        raise ValueError('Frozen ensemble or calibration differs')
    verify_hash('runs/stage8/calibration/factors.json',frozen['factors_sha256'])
    verify_hash('docs/results/stage8c_calibration_factors.csv',frozen['public_factors_sha256'])
    for role in ('calibration','audit'):
        # Identity-only checksum for the calibration archive; no trajectory loading.
        verify_hash(c['archives'][role]['path'],c['archives'][role]['sha256'])
    if prediction_manifest('audit')!=c['predictions']['audit']:
        raise ValueError('Audit predictions differ')
    verify_hash('runs/stage8/calibration/audit_diagnostics.npz',c['diagnostics_sha256'])
    fresh=b['datasets']['fresh_id']
    verify_hash(fresh['archive'],fresh['archive_sha256'])
    verify_hash('runs/stage8/error_awareness/fresh_id/raw_predictions.npz',fresh['raw_predictions_sha256'])
    if factors['times']!=c['predictions']['audit']['times'][1:]:
        raise ValueError('Frozen horizon mismatch')
    return frozen,c,b,norm


def load_role(role,c,b):
    if role not in ROLES:
        raise ValueError('Only existing audit and fresh-ID prediction outputs allowed')
    q90=np.asarray(c['factors']['q'])[:,c['factors']['levels'].index(.9)]
    if role=='audit':
        folder=Path('runs/stage8/calibration/audit_predictions')
        raw={key:np.load(folder/f'{key}.npy',mmap_mode='r') for key in ('error','U_rel','U_rms')}
        times=np.asarray(c['predictions']['audit']['times'])
        hs=prediction_horizons(times)
        with np.load('runs/stage8/calibration/audit_diagnostics.npz',allow_pickle=False) as d:
            width=d['W_rel'][:,:,3].copy();covered=d['field_covered'][:,:,3].copy()
        output={key:np.asarray(value[:,hs]) for key,value in raw.items()}
        origin={'archive':c['archives']['audit']['path'],'archive_sha256':c['archives']['audit']['sha256'],
                'prediction_manifest':'runs/stage8/calibration/audit_predictions/manifest.json',
                'prediction_manifest_sha256':sha256(folder/'manifest.json'),
                'members_and_spread':'retained in original verified audit_predictions arrays',
                'role':'primary retrospective; previously used for Stage 8C coverage'}
    else:
        path=Path('runs/stage8/error_awareness/fresh_id/raw_predictions.npz')
        with np.load(path,allow_pickle=False) as raw:
            times=raw['times'];hs=prediction_horizons(times)
            output={key:raw[key][:,hs] for key in ('error','U_rel','U_rms')}
            # Apply the existing q, without fitting, using the exact raw-spread band.
            bands=band_diagnostics(raw['truth'][:,hs],raw['mean'][:,hs,0],raw['spread'][:,hs,0],q90,
                                   json.loads(Path('docs/results/stage8c_freeze.json').read_text())['epsilon_spread'])
            width=bands['W_rel'];covered=bands['field_covered']
        fresh=b['datasets']['fresh_id']
        origin={'archive':fresh['archive'],'archive_sha256':fresh['archive_sha256'],
                'predictions':str(path),'predictions_sha256':fresh['raw_predictions_sha256'],
                'members_and_spread':'retained in original verified raw_predictions.npz',
                'role':'secondary retrospective replication; Stage 4/8B historical exposure'}
    np.testing.assert_allclose(width,2*q90*output['U_rel'],rtol=1e-12,atol=1e-14)
    output.update(width=width,covered=covered,times=times[1:],trajectory_ids=np.arange(len(width)))
    return output,origin
