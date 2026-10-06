"""Frozen Stage 8D retrospective rollout warning analysis; no refit or generation."""
import csv
import json
from pathlib import Path
import numpy as np
from stage4_evaluation.checkpoint import sha256
from stage8_uq.rollout_warning_io import verify_inputs,load_role,ROOT,FREEZE,ROLES
from stage8_uq.rollout_warning import analyze
from stage8_uq.rollout_warning_plots import make_plots


def flatten_association(prefix,value):
    out={}
    for name in ('pearson','spearman'):
        out[prefix+'_'+name]=value[name]
        interval=value[name+'_ci95']
        out[prefix+'_'+name+'_ci95_low']=interval[0] if interval else None
        out[prefix+'_'+name+'_ci95_high']=interval[1] if interval else None
        out[prefix+'_'+name+'_invalid_bootstraps']=value[name+'_invalid_bootstraps']
    out[prefix+'_status']=value['status']
    return out


def write_csv(path,rows):
    with Path(path).open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    frozen,c,b,norm=verify_inputs()
    summary={'stage':'8D','freeze_sha256':sha256(FREEZE),'stage8c_revision':frozen['stage8c_revision'],
             'no_warning_model_or_threshold_fit':True,'calibration_refitted':False,'new_predictions_generated':False,
             'study':'retrospective audit primary; historically exposed fresh ID replication',
             'source_sha256':{p:sha256(p) for p in ['examples/analyze_stage8d.py','stage8_uq/rollout_warning.py','stage8_uq/rollout_warning_io.py','stage8_uq/rollout_warning_plots.py']},
             'datasets':{}}
    by_horizon=[];early_rows=[];failure_rows=[]
    for role in ROLES:
        arrays,origin=load_role(role,c,b)
        print('Analyzing',role,len(arrays['error']),'trajectories',flush=True)
        result,indices=analyze(arrays['error'],arrays['U_rel'],arrays['U_rms'],arrays['width'],arrays['covered'],
                               frozen['bootstrap_replicates'],frozen['bootstrap_seed'])
        folder=ROOT/role;folder.mkdir(exist_ok=False)
        np.savez_compressed(folder/'analysis_arrays.npz',**arrays,bootstrap_trajectory_indices=indices)
        result['examples']=make_plots(folder/'plots',arrays,result,primary=role=='audit')
        result['origin']=origin;result['analysis_arrays_sha256']=sha256(folder/'analysis_arrays.npz')
        summary['datasets'][role]=result
        for row in result['by_horizon']:
            by_horizon.append({'dataset':role,**{k:v for k,v in row.items() if k!='current_error_association'},
                               **flatten_association('current_U_E',row['current_error_association'])})
        for row in result['early_warning']:
            flat={'dataset':role,'time':row['time'],'target_time':row['target_time'],'lag':row['lag'],'n_trajectories':row['n_trajectories']}
            for key in ('uncertainty_future','current_error_future','partial_controlling_current','uncertainty_delta_error','width_future','noncoverage_future'):
                flat.update(flatten_association(key,row[key]))
            flat.update({k:row[k] for k in ['mean_delta_error','source_noncoverage_count','future_error_mean_given_noncoverage','future_error_mean_given_coverage']})
            early_rows.append(flat)
            failure_rows.append({'dataset':role,'time':row['time'],'target_time':row['target_time'],'lag':row['lag'],**row['new_failure']})
        print('Completed',role,flush=True)
    with Path('docs/results/stage8d_summary.json').open('x') as file:
        json.dump(summary,file,indent=2,allow_nan=False);file.write('\n')
    write_csv('docs/results/stage8d_by_horizon.csv',by_horizon)
    write_csv('docs/results/stage8d_early_warning.csv',early_rows)
    write_csv('docs/results/stage8d_new_failures.csv',failure_rows)

if __name__=='__main__':main()
