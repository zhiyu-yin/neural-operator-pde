"""Descriptive ID/OOD effects and quartile categories; trajectory resampling only."""
import numpy as np
from .awareness import association,ranks,prediction_horizons
from .calibration import LEVELS,band_diagnostics,wilson_interval
from .ood_awareness_io import QUADRANTS


def bootstrap_indices(sizes,replicates=2000,seed=8501):
    rng=np.random.default_rng(seed)
    draws={role:rng.integers(n,size=(replicates,n)) for role,n in sizes.items()}
    if all(q in draws for q in QUADRANTS):
        draws['high_velocity']=np.concatenate([draws[q]+16*i for i,q in enumerate(QUADRANTS)],axis=1)
    return draws


def interval(values):
    values=np.asarray(values)
    finite=values[np.isfinite(values)]
    return np.quantile(finite,[.025,.975]).tolist() if len(finite) else None


def regime_effect(values,control,draws,control_draws):
    """Independent ID/OOD resamples, never pair archive-local trajectory IDs."""
    values,control=np.asarray(values),np.asarray(control)
    vm=values[draws].mean(1);cm=control[control_draws].mean(1)
    vmed=np.median(values[draws],axis=1);cmed=np.median(control[control_draws],axis=1)
    valid=cm>1e-12
    ratios=np.divide(vm,cm,out=np.full_like(vm,np.nan),where=valid)
    return {'mean_difference':float(values.mean()-control.mean()),'mean_difference_ci95':interval(vm-cm),
            'median_difference':float(np.median(values)-np.median(control)),'median_difference_ci95':interval(vmed-cmed),
            'mean_ratio':float(values.mean()/control.mean()) if control.mean()>1e-12 else None,
            'mean_ratio_ci95':interval(ratios),'invalid_ratio_bootstraps':int((~valid).sum())}


def auroc(id_score,ood_score):
    """Larger U_rel is the OOD direction; average ties receive half credit."""
    id_score,ood_score=np.asarray(id_score),np.asarray(ood_score)
    if not len(id_score) or not len(ood_score) or not np.isfinite(id_score).all() or not np.isfinite(ood_score).all():return None
    return float(((ood_score[:,None]>id_score).sum()+.5*(ood_score[:,None]==id_score).sum())/(len(id_score)*len(ood_score)))


def category_ranks(id_error,id_u,ood_error,ood_u):
    """Percentile-style average ranks in one fixed-horizon ID+OOD pool."""
    n=len(id_error);total=n+len(ood_error)
    er=(ranks(np.concatenate([id_error,ood_error]))-.5)/total
    ur=(ranks(np.concatenate([id_u,ood_u]))-.5)/total
    high_e=er>=.75;high_u=ur>=.75;low_e=er<=.25
    labels=np.full(total,'other',dtype='<U40')
    labels[high_e & high_u]='high_error_high_uncertainty'
    labels[high_e & ~high_u]='high_error_modest_uncertainty'
    labels[low_e & high_u]='low_error_high_uncertainty'
    return er,ur,labels


def scalar_and_bands(arrays,q,epsilon):
    hs=prediction_horizons(arrays['times'])
    metrics={key:arrays[key][:,hs] for key in ('error','U_rel','U_rms')}
    widths=[];covered=[]
    for l in range(len(LEVELS)):
        result=band_diagnostics(arrays['truth'][:,hs],arrays['mean'][:,hs],arrays['spread'][:,hs],q[:,l],epsilon)
        widths.append(result['W_rel']);covered.append(result['field_covered'])
        np.testing.assert_allclose(result['W_rel'],2*q[:,l]*metrics['U_rel'],rtol=1e-12,atol=1e-14)
    metrics.update(W_rel=np.stack(widths,axis=-1),covered=np.stack(covered,axis=-1),qualified_ids=arrays['qualified_ids'])
    return metrics


def analyze_regime(role,data,control,draws,id_draws):
    rows=[];coverage=[];categories=[];eranks=[];uranks=[];labels=[]
    n=len(data['error'])
    for h in range(10):
        row={'regime':role,'time':(h+1)/10,'n_trajectories':n}
        for key in ('error','U_rel','U_rms'):
            v=data[key][:,h]
            row['mean_'+key]=float(v.mean());row['median_'+key]=float(np.median(v))
            row['mean_'+key+'_ci95']=interval(v[draws].mean(1))
        row.update({f'U_rel_q{q}':float(np.quantile(data['U_rel'][:,h],q/100)) for q in (10,50,90)})
        row['within_regime']=association(data['U_rel'][:,h],data['error'][:,h],draws)
        row['versus_fresh_id']={key:regime_effect(data[key][:,h],control[key][:,h],draws,id_draws) for key in ('error','U_rel')}
        rows.append(row)
        for l,level in enumerate(LEVELS):
            events=data['covered'][:,h,l];count=int(events.sum())
            ci=interval(events[draws].mean(1))
            coverage.append({'regime':role,'time':(h+1)/10,'nominal':level,'n_trajectories':n,'covered_fields':count,
                             'coverage':count/n,'coverage_error':count/n-level,'coverage_ci95':ci,
                             'coverage_interval_method':'whole-trajectory percentile; quadrant-stratified for aggregate',
                             'wilson_ci95_secondary':wilson_interval(count,n) if role!='high_velocity' else None,
                             'mean_W_rel':float(data['W_rel'][:,h,l].mean()),'median_W_rel':float(np.median(data['W_rel'][:,h,l])),
                             'mean_W_rel_ci95':interval(data['W_rel'][:,h,l][draws].mean(1))})
        if role not in ('fresh_id','historical_id'):
            er,ur,lab=category_ranks(control['error'][:,h],control['U_rel'][:,h],data['error'][:,h],data['U_rel'][:,h])
            eranks.append(er);uranks.append(ur);labels.append(lab)
            lab=lab[len(control['error']):]
            categories.append({'time':(h+1)/10,'n_ood':n,
                'counts':{name:int(np.sum(lab==name)) for name in ('high_error_high_uncertainty','high_error_modest_uncertainty','low_error_high_uncertainty','other')},
                'case_ids':{name:data['qualified_ids'][lab==name].tolist() for name in ('high_error_high_uncertainty','high_error_modest_uncertainty','low_error_high_uncertainty')}})
    ood_detection=[]
    if role not in ('fresh_id','historical_id'):
        for h in range(10):
            id_u,u=control['U_rel'][:,h],data['U_rel'][:,h]
            values=[auroc(id_u[ii],u[jj]) for ii,jj in zip(id_draws,draws)]
            ood_detection.append({'time':(h+1)/10,'auroc':auroc(id_u,u),'ci95':interval(values),
                                  'n_id':len(id_u),'n_ood':len(u)})
    case_arrays={'pooled_error_rank':np.stack(eranks,axis=1),'pooled_uncertainty_rank':np.stack(uranks,axis=1),'pooled_category':np.stack(labels,axis=1),'pooled_qualified_ids':np.concatenate([control['qualified_ids'],data['qualified_ids']])} if labels else {}
    return {'by_horizon':rows,'coverage':coverage,'ood_detection':ood_detection,'failure_categories':categories},case_arrays
