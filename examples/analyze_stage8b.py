"""Verified, inference-only Stage 8B. Run from repository root with python -m."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import verify_hash, sha256
from stage4_evaluation.diagnostics import physical_diagnostics
from stage8_uq.checkpoint import frozen_protocol, load_member
from stage8_uq.ensemble import rollout_members
from stage8_uq.awareness import analyze, spatial_association

FREEZE = Path('docs/results/stage8b_freeze.json')
ARCHIVES = {'fresh_id':Path('data/stage4/fresh_id.npz'), 'historical_id':Path('data/dev/trajectories.npz')}


def verified_inputs():
    freeze=json.loads(FREEZE.read_text()); root=Path('runs/stage8')
    verify_hash(root/'ensemble_manifest.json',freeze['ensemble_manifest_sha256'])
    frozen=frozen_protocol(); models=[]
    for i in range(5):
        verify_hash(root/f'member_{i}/manifest.json',freeze['member_manifest_sha256'][str(i)])
        model,norm,manifest=load_member(root/f'member_{i}',frozen)
        if manifest != freeze['members'][i]: raise ValueError('Frozen member changed')
        models.append(model)
    verify_hash(ARCHIVES['fresh_id'],freeze['fresh_id_sha256'])
    verify_hash(ARCHIVES['historical_id'],frozen['training_archive_sha256'])
    return freeze,frozen,models,norm


def load_evaluation(role, frozen):
    if role not in ARCHIVES: raise ValueError('Only ID evaluation allowed')
    data=load_dataset(ARCHIVES[role])
    ids=np.arange(64) if role=='fresh_id' else np.asarray(frozen['split_ids']['test'])
    return data['fields'][ids],data['coefficients'][ids],data['times'],ids


def figures(folder, result, ar, truth, ids):
    rows=result['by_horizon']; times=[r['time'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['mean_error','mean_U_rel'],['Mean actual relative L2 error','Mean raw U_rel']):
        ax.plot(times,[r[key] for r in rows],marker='o'); ax.set(xlabel='Prediction horizon',ylabel=title)
    fig.tight_layout(); fig.savefig(folder/'rollout_trends.png',dpi=150); plt.close(fig)
    bins=result['quintiles_horizon_balanced']; means=np.array([b['mean_error'] for b in bins]); ci=np.array([b['mean_ci95'] for b in bins])
    fig,ax=plt.subplots(figsize=(6,4)); ax.errorbar(range(1,6),means,yerr=np.abs(ci.T-means),marker='o',capsize=4)
    ax.set(xlabel='Within-horizon uncertainty quintile',ylabel='Horizon-balanced mean error',xticks=range(1,6))
    fig.tight_layout(); fig.savefig(folder/'quintiles.png',dpi=150); plt.close(fig)
    err=np.abs(ar['mean'][:,:,0]-truth)
    spatial=spatial_association(err[:,1:],ar['spread'][:,1:,0])
    result['spatial_spearman_by_horizon']=[]
    for h in range(10):
        values=[v[h] for v in spatial if v[h] is not None]
        result['spatial_spearman_by_horizon'].append({'time':float(times[h]), 'median':float(np.median(values)) if values else None, 'n_defined':len(values)})
    result['spatial_examples']=[]
    for label,i in [('predetermined',0),('highest_final_error',int(np.argmax(result.pop('_final_error')))),('highest_final_uncertainty',int(np.argmax(ar['U_rel'][:,-1])))]:
        result['spatial_examples'].append({'rule':label,'trajectory_id':int(ids[i]),'final_spatial_spearman':spatial[i][-1]})
        fields=[truth[i,-1],ar['mean'][i,-1,0],err[i,-1],ar['spread'][i,-1,0]]
        fig,axes=plt.subplots(1,4,figsize=(14,3.5))
        for k,(ax,field,title) in enumerate(zip(axes,fields,['Ground Truth','Ensemble Mean','Absolute Error','Ensemble Spread'])):
            lo=min(fields[0].min(),fields[1].min()) if k<2 else 0
            hi=max(fields[0].max(),fields[1].max()) if k<2 else max(fields[2].max(),fields[3].max())
            im=ax.imshow(field,origin='lower',vmin=lo,vmax=hi,cmap='viridis' if k<2 else 'magma'); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{label}: trajectory {ids[i]}, t=1.0'); fig.tight_layout(); fig.savefig(folder/f'spatial_{label}.png',dpi=150); plt.close(fig)


def main():
    freeze,frozen,models,norm=verified_inputs()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    root=Path('runs/stage8/error_awareness'); root.mkdir(exist_ok=True)
    summary={'calibration_fitted':False,'primary_score':'U_rel','study':'retrospective association study','freeze_sha256':sha256(FREEZE),'bootstrap':{'seed':freeze['bootstrap_seed'],'replicates':freeze['bootstrap_replicates'],'interval':'pointwise percentile 95%; paired trajectory resampling across all horizons'},'quintile_bootstrap':'fixed observed bin assignments; whole trajectories resampled; equal horizon weight; empty-bin draws reported and omitted','source_sha256':{p:sha256(p) for p in ['stage8_uq/awareness.py','examples/analyze_stage8b.py']},'datasets':{}}
    for role in ARCHIVES:
        print('Running',role,flush=True)
        truth,coefficients,times,ids=load_evaluation(role,frozen)
        ar=rollout_members(models,norm,truth,coefficients,times)
        error=physical_diagnostics(ar['mean'][:,:,0],truth,truth[:,0])['relative_l2']
        folder=root/role; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'raw_predictions.npz',**ar,error=error,truth=truth,times=times,trajectory_ids=ids,coefficients=coefficients)
        result=analyze(ar['U_rel'],ar['U_rms'],error,times,freeze['bootstrap_seed'],freeze['bootstrap_replicates'])
        result.update(archive=str(ARCHIVES[role]),archive_sha256=sha256(ARCHIVES[role]),trajectory_ids=ids.tolist(),role='primary retrospective' if role=='fresh_id' else 'exploratory historical',floor_activations=int(ar['floor_active'].sum()),raw_predictions_sha256=sha256(folder/'raw_predictions.npz'))
        result['_final_error']=error[:,-1]
        figures(folder,result,ar,truth,ids)
        summary['datasets'][role]=result
        print('Completed',role,flush=True)
    Path('docs/results/stage8b_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for name,key in [('by_horizon','by_horizon'),('quintiles','quintiles_by_horizon'),('early_warning','early_warning')]:
        rows=[{'dataset':role,**row} for role,result in summary['datasets'].items() for row in result[key]]
        with open(f'docs/results/stage8b_{name}.csv','w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

if __name__=='__main__': main()
