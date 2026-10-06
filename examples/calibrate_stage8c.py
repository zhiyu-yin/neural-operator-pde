"""Stage 8C: new ID data, calibration-only factors, then untouched audit coverage.

Run phases in order: generate, predict-calibration, fit, predict-audit, evaluate.
No phase trains models, refits normalization, or uses old data for calibration.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from dataset_generation import load_dataset
from examples.analyze_stage8b import verified_inputs
from stage4_evaluation.checkpoint import sha256, verify_hash
from stage4_evaluation.diagnostics import physical_diagnostics
from stage8_uq.calibration_data import generate_archives, verified_archive, write_new_json
from stage8_uq.calibration import (LEVELS, residual_scores, conformal_factors,
                                  band_diagnostics, reliability_rows, grouped_summary)
from stage8_uq.ensemble import rollout_members

ROOT = Path('runs/stage8/calibration')
FREEZE = Path('docs/results/stage8c_freeze.json')


def verify_frozen():
    frozen=json.loads(FREEZE.read_text())
    subprocess.run(['git','cat-file','-e',frozen['stage8b_revision']+'^{commit}'],check=True)
    subprocess.run(['git','merge-base','--is-ancestor',frozen['stage8b_revision'],'HEAD'],check=True)
    verify_hash('docs/results/stage8b_freeze.json',frozen['stage8b_freeze_sha256'])
    verify_hash('docs/results/stage8b_summary.json',frozen['stage8b_summary_sha256'])
    for path,digest in frozen['source_sha256'].items():
        verify_hash(path,digest)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    _,_,models,norm=verified_inputs()
    if norm.to_dict()!=frozen['normalization']:
        raise ValueError('Frozen normalization differs')
    return frozen,models,norm


def prediction_manifest(role):
    if role not in ('calibration','audit'):
        raise ValueError('Only the two new ID archives are prediction roles')
    path=ROOT/f'{role}_predictions'
    manifest=json.loads((path/'manifest.json').read_text())
    verify_hash(FREEZE,manifest['freeze_sha256'])
    verify_hash(f'data/stage8/{role}.npz',manifest['archive_sha256'])
    for name,digest in manifest['array_sha256'].items():
        verify_hash(path/name,digest)
    return manifest


def read_predictions(role):
    manifest=prediction_manifest(role)
    return {Path(name).stem:np.load(ROOT/f'{role}_predictions'/name,mmap_mode='r')
            for name in manifest['array_sha256']},manifest


def verified_factors():
    receipt=json.loads((ROOT/'factor_freeze_receipt.json').read_text())
    verify_hash(ROOT/'factors.json',receipt['factors_sha256'])
    verify_hash('docs/results/stage8c_calibration_factors.csv',receipt['public_factors_sha256'])
    factors=json.loads((ROOT/'factors.json').read_text())
    verify_hash(FREEZE,factors['freeze_sha256'])
    verify_hash('data/stage8/calibration.npz',factors['calibration_archive_sha256'])
    if factors['fit_role']!='calibration' or factors['audit_used_for_fit']:
        raise ValueError('Factors must be fitted only on calibration archive')
    return factors,receipt


def predict(role, models, norm):
    if role=='audit':
        verified_factors()  # Audit inference cannot precede a frozen fit.
    data,entry=verified_archive(role)
    folder=ROOT/f'{role}_predictions'
    folder.mkdir(exist_ok=False)
    count,steps,n,_=data['fields'].shape
    shapes={'members':(5,count,steps,1,n,n),'mean':(count,steps,n,n),
            'spread':(count,steps,n,n),'U_rms':(count,steps),'U_rel':(count,steps),
            'floor_active':(count,steps),'error':(count,steps)}
    arrays={key:np.lib.format.open_memmap(folder/f'{key}.npy',mode='w+',shape=shape,
                                        dtype=bool if key=='floor_active' else np.float64)
            for key,shape in shapes.items()}
    for start in range(0,count,16):
        stop=min(start+16,count)
        truth=data['fields'][start:stop]
        ar=rollout_members(models,norm,truth,data['coefficients'][start:stop],data['times'])
        np.testing.assert_array_equal(ar['spread'][:,0],0)
        np.testing.assert_array_equal(ar['mean'][:,0,0],truth[:,0])
        for key in arrays:
            if key=='members': arrays[key][:,start:stop]=ar[key]
            elif key=='error': arrays[key][start:stop]=physical_diagnostics(ar['mean'][:,:,0],truth,truth[:,0])['relative_l2']
            elif key in ('mean','spread'): arrays[key][start:stop]=ar[key][:,:,0]
            else: arrays[key][start:stop]=ar[key]
        for value in arrays.values(): value.flush()
        print(f'{role}: {stop}/{count} member-wise ten-step rollouts',flush=True)
    manifest={'role':role,'archive_sha256':entry['sha256'],'freeze_sha256':sha256(FREEZE),
              'array_sha256':{p.name:sha256(p) for p in sorted(folder.glob('*.npy'))},
              'members':5,'feedback':'own member','times':data['times'].tolist()}
    write_new_json(folder/'manifest.json',manifest)


def csv_output(path, rows):
    with Path(path).open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def fit(frozen):
    # Deliberately no audit argument, loading, metrics, or references in this function.
    data,entry=verified_archive('calibration')
    predictions,manifest=read_predictions('calibration')
    scores=residual_scores(data['fields'],predictions['mean'],predictions['spread'],data['times'],frozen['epsilon_spread'])
    result=conformal_factors(scores['scores'],role='calibration')
    np.savez_compressed(ROOT/'calibration_scores.npz',**scores)
    factors={'fit_role':'calibration','audit_used_for_fit':False,'n_calibration':len(data['fields']),
             'calibration_archive_sha256':entry['sha256'],'prediction_manifest_sha256':sha256(ROOT/'calibration_predictions/manifest.json'),
             'freeze_sha256':sha256(FREEZE),'score_arrays_sha256':sha256(ROOT/'calibration_scores.npz'),
             'times':data['times'][1:].tolist(),'levels':list(LEVELS),
             **{key:value.tolist() for key,value in result.items()},
             'spread_floor_cells_by_horizon':scores['floor_cells_by_horizon'].tolist(),
             'spread_floor_fields_by_horizon':scores['floor_fields_by_horizon'].tolist()}
    write_new_json(ROOT/'factors.json',factors)
    rows=[{'time':float(t),'nominal':float(level),'n_calibration':len(data['fields']),
           'k':int(result['k'][l]),'clipped':bool(result['clipped'][l]),'q':float(result['q'][h,l])}
          for h,t in enumerate(data['times'][1:]) for l,level in enumerate(LEVELS)]
    csv_output('docs/results/stage8c_calibration_factors.csv',rows)
    write_new_json(ROOT/'factor_freeze_receipt.json',{
        'factors_sha256':sha256(ROOT/'factors.json'),
        'public_factors_sha256':sha256('docs/results/stage8c_calibration_factors.csv'),
        'audit_predictions_existed_at_freeze':(ROOT/'audit_predictions').exists()})
    print('Frozen calibration factors; no audit predictions have been evaluated',flush=True)


def evaluate(frozen):
    factors,receipt=verified_factors()
    if receipt['audit_predictions_existed_at_freeze']:
        raise ValueError('Audit predictions preceded calibration freeze')
    data,entry=verified_archive('audit')
    predictions,manifest=read_predictions('audit')
    scores=residual_scores(data['fields'],predictions['mean'],predictions['spread'],data['times'],frozen['epsilon_spread'])
    q=np.asarray(factors['q'])
    diagnostics={key:[] for key in ['field_covered','pointwise_fraction_secondary','mean_physical_width','rms_physical_width','W_rel']}
    bands=ROOT/'audit_bands'; bands.mkdir(exist_ok=False)
    for l,level in enumerate(LEVELS):
        result=band_diagnostics(data['fields'][:,1:],predictions['mean'][:,1:],predictions['spread'][:,1:],q[:,l],frozen['epsilon_spread'])
        np.save(bands/f'lower_{int(level*100)}.npy',result.pop('lower'))
        np.save(bands/f'upper_{int(level*100)}.npy',result.pop('upper'))
        for key in diagnostics: diagnostics[key].append(result[key])
        np.testing.assert_allclose(result['W_rel'],2*q[:,l]*predictions['U_rel'][:,1:],rtol=1e-12,atol=1e-14)
    diagnostics={key:np.stack(value,axis=-1) for key,value in diagnostics.items()}
    rows=reliability_rows(diagnostics,data['times'][1:])
    aggregate,indices=grouped_summary(diagnostics)
    np.savez_compressed(ROOT/'audit_diagnostics.npz',**diagnostics,**scores,bootstrap_trajectory_indices=indices)
    csv_output('docs/results/stage8c_coverage.csv',rows)
    from stage8_uq.calibration_plots import make_plots
    examples=make_plots(ROOT,rows,aggregate,data,predictions,q,diagnostics)
    archive_manifest=json.loads(Path('data/stage8/manifest.json').read_text())
    summary={'stage':'8C','study':'independent ID field-coverage audit; calibration only',
             'freeze_sha256':sha256(FREEZE),'stage8b_revision':frozen['stage8b_revision'],
             'ensemble_checkpoint_sha256':frozen['checkpoint_sha256'],
             'archive_manifest_sha256':sha256('data/stage8/manifest.json'),
             'archives':{role:{k:v for k,v in item.items() if k!='input_fingerprints'} for role,item in archive_manifest['archives'].items()},
             'previous_seeds':archive_manifest['previous_seeds'],'all_new_inputs_disjoint':True,
             'historical_OOD_use':'identity/seed exclusion only; no predictions, fitting, or coverage conclusions',
             'factors':factors,'factor_freeze_receipt':receipt,'coverage_by_horizon':rows,
             'horizon_aggregate':aggregate,'aggregate_interpretation':'mean of ten per-horizon coverage probabilities, not simultaneous-time coverage',
             'audit_spread_floor_cells_by_horizon':scores['floor_cells_by_horizon'].tolist(),
             'audit_spread_floor_fields_by_horizon':scores['floor_fields_by_horizon'].tolist(),
             'examples':examples,'global_comparison_performed':False,
             'source_sha256':{str(p):sha256(p) for p in [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),Path('stage8_uq/calibration.py'),Path('stage8_uq/calibration_data.py'),Path('stage8_uq/calibration_plots.py')]},
             'diagnostics_sha256':sha256(ROOT/'audit_diagnostics.npz'),
             'band_sha256':{str(p):sha256(p) for p in sorted(bands.glob('*.npy'))},
             'predictions':{role:prediction_manifest(role) for role in ('calibration','audit')},
             'scalar_diagnostics':{}}
    del data,predictions
    for role in ('calibration','audit'):
        predictions,_=read_predictions(role)
        summary['scalar_diagnostics'][role]=[
            {'time':float(t),'mean_U_rel':float(predictions['U_rel'][:,h].mean()),
             'median_U_rel':float(np.median(predictions['U_rel'][:,h])),
             'mean_U_rms':float(predictions['U_rms'][:,h].mean()),
             'mean_error':float(predictions['error'][:,h].mean()),
             'U_rel_floor_count':int(predictions['floor_active'][:,h].sum())}
            for h,t in enumerate(factors['times'],start=1)]
    write_new_json('docs/results/stage8c_summary.json',summary)
    print('Audit complete; all factors remain unchanged',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['generate','predict-calibration','fit','predict-audit','evaluate'])
    args=parser.parse_args()
    frozen,models,norm=verify_frozen()
    if args.phase=='generate': generate_archives()
    elif args.phase.startswith('predict-'): predict(args.phase.removeprefix('predict-'),models,norm)
    elif args.phase=='fit': fit(frozen)
    else: evaluate(frozen)

if __name__=='__main__': main()
