"""Frozen Stage 8E regimes; inference-only reuse of Stage 8A machinery."""
import json
from pathlib import Path
import subprocess
import numpy as np
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import sha256,verify_hash
from stage4_evaluation.diagnostics import physical_diagnostics
from stage8_uq.rollout_warning_io import verify_inputs as verify_stage8d
from examples.calibrate_stage8c import verify_frozen as verify_stage8c
from stage8_uq.ensemble import rollout_members

FREEZE=Path('docs/results/stage8e_freeze.json')
ROOT=Path('runs/stage8/ood_awareness')
QUADRANTS=('velocity_pp','velocity_pn','velocity_np','velocity_nn')
OOD=('high_nu',)+QUADRANTS
REGIMES=('fresh_id','historical_id')+OOD


def verify_inputs():
    frozen=json.loads(FREEZE.read_text())
    subprocess.run(['git','merge-base','--is-ancestor',frozen['stage8d_revision'],'HEAD'],check=True)
    for path,digest in frozen['protected_tracked_sha256'].items():verify_hash(path,digest)
    _,c,b,norm=verify_stage8d()
    _,models,norm=verify_stage8c()
    for role,entry in frozen['archives'].items():verify_hash(entry['path'],entry['sha256'])
    d=json.loads(Path('docs/results/stage8d_summary.json').read_text())
    verify_hash('docs/results/stage8d_freeze.json',d['freeze_sha256'])
    for role,item in d['datasets'].items():verify_hash(f'runs/stage8/rollout_uncertainty/{role}/analysis_arrays.npz',item['analysis_arrays_sha256'])
    for role in ('fresh_id','historical_id'):
        verify_hash(f'runs/stage8/error_awareness/{role}/raw_predictions.npz',b['datasets'][role]['raw_predictions_sha256'])
    return frozen,c,b,models,norm


def load_archive(role,frozen):
    if role not in REGIMES:raise ValueError('Unknown Stage 8E archive role')
    entry=frozen['archives'][role]
    verify_hash(entry['path'],entry['sha256'])
    data=load_dataset(entry['path'])
    if role=='historical_id':
        ids=np.asarray(entry['selected_ids'])
    else:
        if json.loads(data['metadata_json'].item())['config']!=entry['config']:
            raise ValueError('Regime generation configuration differs')
        ids=np.arange(len(data['fields']))
    return data['fields'][ids],data['coefficients'][ids],data['times'],ids


def predict_ood(frozen,models,norm):
    if len(models)!=5:raise ValueError('All five frozen members are required')
    for role in OOD:
        folder=ROOT/role
        if (folder/'prediction_manifest.json').exists():
            load_outputs(role,{},frozen)
            print('Reused verified predictions',role,flush=True)
            continue
        folder.mkdir(exist_ok=False)
        truth,coefficients,times,ids=load_archive(role,frozen)
        print('Running all five members:',role,len(ids),'trajectories',flush=True)
        ar=rollout_members(models,norm,truth,coefficients,times)
        np.testing.assert_array_equal(ar['mean'][:,0,0],truth[:,0])
        np.testing.assert_array_equal(ar['spread'][:,0],0)
        error=physical_diagnostics(ar['mean'][:,:,0],truth,truth[:,0])['relative_l2']
        np.savez_compressed(folder/'raw_predictions.npz',**ar,error=error,truth=truth,
                            coefficients=coefficients,times=times,trajectory_ids=ids)
        manifest={'regime':role,'archive_sha256':frozen['archives'][role]['sha256'],
                  'freeze_sha256':sha256(FREEZE),'raw_predictions_sha256':sha256(folder/'raw_predictions.npz'),
                  'members':5,'member_shape':list(ar['members'].shape),'feedback':'own-member','generated_data':False}
        with (folder/'prediction_manifest.json').open('x') as file:json.dump(manifest,file,indent=2);file.write('\n')
        print('Completed',role,flush=True)


def load_outputs(role,stage8b,frozen):
    if role not in REGIMES:raise ValueError('Unknown prediction regime')
    if role in ('fresh_id','historical_id'):
        path=Path(f'runs/stage8/error_awareness/{role}/raw_predictions.npz')
        verify_hash(path,stage8b['datasets'][role]['raw_predictions_sha256'])
    else:
        folder=ROOT/role;path=folder/'raw_predictions.npz'
        manifest=json.loads((folder/'prediction_manifest.json').read_text())
        verify_hash(FREEZE,manifest['freeze_sha256'])
        if manifest['regime']!=role or manifest['members']!=5 or manifest['archive_sha256']!=frozen['archives'][role]['sha256']:
            raise ValueError('Prediction role/provenance mismatch')
        verify_hash(path,manifest['raw_predictions_sha256'])
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in ('mean','spread','truth','error','U_rel','U_rms','floor_active','times','trajectory_ids')}
    arrays['mean']=arrays['mean'][:,:,0];arrays['spread']=arrays['spread'][:,:,0]
    arrays['qualified_ids']=np.asarray([f'{role}:{i}' for i in arrays['trajectory_ids']])
    return arrays,{'prediction_path':str(path),'prediction_sha256':sha256(path),**frozen['archives'][role]}


def aggregate_quadrants(arrays):
    if set(arrays)!=set(QUADRANTS):raise ValueError('Exactly the four frozen quadrants are required')
    if any(len(arrays[q]['error'])!=16 for q in QUADRANTS):raise ValueError('Expected 16 trajectories per quadrant')
    result={key:np.concatenate([arrays[q][key] for q in QUADRANTS],axis=0)
            for key in ('error','U_rel','U_rms','W_rel','covered','qualified_ids')}
    if len(set(result['qualified_ids']))!=64:raise ValueError('Duplicate trajectory identity in aggregate')
    return result
