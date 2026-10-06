"""Frozen ID-versus-OOD study; predict then analyze, with no data generation or fitting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from stage4_evaluation.checkpoint import sha256
from stage8_uq.ood_awareness_io import verify_inputs,predict_ood,load_outputs,aggregate_quadrants,REGIMES,QUADRANTS,ROOT,FREEZE
from stage8_uq.ood_awareness import bootstrap_indices,scalar_and_bands,analyze_regime


def flatten(prefix,value):
    out={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,dict):out.update(flatten(key+'_',v))
        elif isinstance(v,list) and len(v)==2:out.update({key+'_low':v[0],key+'_high':v[1]})
        else:out[key]=v
    return out


def csv_file(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def analyze(frozen,c,b):
    data={};origins={};q=np.asarray(c['factors']['q'])
    epsilon=json.loads(Path('docs/results/stage8c_freeze.json').read_text())['epsilon_spread']
    for role in REGIMES:
        raw,origin=load_outputs(role,b,frozen)
        data[role]=scalar_and_bands(raw,q,epsilon);origins[role]=origin
        del raw
    data['high_velocity']=aggregate_quadrants({role:data[role] for role in QUADRANTS})
    origins['high_velocity']={'source_quadrants':list(QUADRANTS),'counts':[16]*4,'new_archive_generated':False}
    draws=bootstrap_indices({role:len(data[role]['error']) for role in REGIMES},frozen['bootstrap_replicates'],frozen['bootstrap_seed'])
    summary={'stage':'8E','study':'retrospective OOD failure-awareness; ID bands are transfer stress tests only',
             'freeze_sha256':sha256(FREEZE),'stage8d_revision':frozen['stage8d_revision'],
             'calibration_factors_sha256':c['factor_freeze_receipt']['factors_sha256'],
             'fitted_on_ood':False,'generated_new_data':False,'regimes':{},
             'source_sha256':{p:sha256(p) for p in ['examples/evaluate_stage8e.py','stage8_uq/ood_awareness_io.py','stage8_uq/ood_awareness.py','stage8_uq/ood_awareness_plots.py']}}
    horizon_rows=[];coverage_rows=[]
    for role,values in data.items():
        print('Analyzing',role,flush=True)
        result,cases=analyze_regime(role,values,data['fresh_id'],draws[role],draws['fresh_id'])
        folder=ROOT/role;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'analysis_arrays.npz',**values,bootstrap_trajectory_indices=draws[role],**cases)
        result['origin']=origins[role];result['analysis_arrays_sha256']=sha256(folder/'analysis_arrays.npz')
        summary['regimes'][role]=result
        horizon_rows.extend(flatten('',r) for r in result['by_horizon'])
        coverage_rows.extend(flatten('',r) for r in result['coverage'])
    from stage8_uq.ood_awareness_plots import make_plots
    summary['spatial_examples']=make_plots(ROOT,data,summary,lambda role:load_outputs(role,b,frozen)[0],q)
    with Path('docs/results/stage8e_summary.json').open('x') as file:json.dump(summary,file,indent=2,allow_nan=False);file.write('\n')
    csv_file('docs/results/stage8e_by_horizon.csv',horizon_rows)
    csv_file('docs/results/stage8e_coverage.csv',coverage_rows)
    print('Analysis complete; all ID factors unchanged',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['predict','analyze'])
    args=parser.parse_args();frozen,c,b,models,norm=verify_inputs()
    if args.phase=='predict':predict_ood(frozen,models,norm)
    else:analyze(frozen,c,b)

if __name__=='__main__':main()
