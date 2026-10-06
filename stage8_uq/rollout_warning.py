"""Retrospective incremental warning diagnostics; no trained warning model or calibration."""
import numpy as np
from .awareness import ranks, correlation, association, top_fifth, early_pairs


def residualize(values, control, tolerance=1e-12):
    """Projection off one centered control, including an intercept; not prediction fitting."""
    values,control=np.asarray(values,float),np.asarray(control,float)
    if values.ndim!=1 or values.shape!=control.shape or len(values)<4:
        return None
    if not np.isfinite(values).all() or not np.isfinite(control).all():
        return None
    x,c=values-values.mean(),control-control.mean()
    if np.linalg.norm(c)==0 or np.linalg.norm(x)==0:
        return None
    residual=x-c*(c@x)/(c@c)
    if np.linalg.norm(residual)<=tolerance*np.linalg.norm(x):
        return None
    return residual


def partial_correlation(x,y,control,rank=False):
    """Rank original variables first for partial Spearman; never rerank residuals."""
    x,y,control=(np.asarray(v,float) for v in (x,y,control))
    if not all(np.isfinite(v).all() for v in (x,y,control)):
        return None
    if rank:
        x,y,control=(ranks(v) for v in (x,y,control))
    rx,ry=residualize(x,control),residualize(y,control)
    return None if rx is None or ry is None else correlation(rx,ry)


def partial_association(x,y,control,indices):
    result={'n_trajectories':len(x)}
    for name,rank in [('pearson',False),('spearman',True)]:
        result[name]=partial_correlation(x,y,control,rank)
        values=[partial_correlation(x[i],y[i],control[i],rank) for i in indices]
        valid=[v for v in values if v is not None]
        result[name+'_ci95']=np.quantile(valid,[.025,.975]).tolist() if valid else None
        result[name+'_invalid_bootstraps']=len(values)-len(valid)
    result['status']='ok' if all(result[k] is not None for k in ['pearson','spearman']) else 'undefined: constant/nonfinite control or degenerate residuals'
    return result


def aligned(error,uncertainty,source,lag):
    """Input arrays already exclude t=0: source index 0 is t=0.1."""
    error,uncertainty=np.asarray(error),np.asarray(uncertainty)
    if error.shape!=uncertainty.shape or error.ndim!=2 or error.shape[1]!=10:
        raise ValueError('Expected matching N by ten prediction-horizon arrays')
    if lag not in (1,2) or source<0 or source+lag>=10:
        raise ValueError('Invalid existing-horizon alignment')
    return uncertainty[:,source],error[:,source],error[:,source+lag],error[:,source+lag]-error[:,source]


def increments(values,lag=1):
    values=np.asarray(values)
    if values.ndim!=2 or values.shape[1]!=10 or lag not in (1,2):
        raise ValueError('Ten prediction horizons and lag one/two required')
    return values[:,lag:]-values[:,:-lag]


def new_failure(score,current,future,minimum=5):
    """Frozen top-fifth rule; only not-currently-high cases are eligible."""
    score,current,future=(np.asarray(v) for v in (score,current,future))
    if score.ndim!=1 or score.shape!=current.shape or score.shape!=future.shape or len(score)<5:
        raise ValueError('At least five matched trajectories required')
    if not all(np.isfinite(v).all() for v in (score,current,future)):
        raise ValueError('Nonfinite failure-analysis input')
    eligible=~top_fifth(current)
    events=eligible & top_fifth(future)
    ids=np.flatnonzero(eligible)
    selected=np.zeros(len(score),bool)
    selected[ids[top_fifth(score[ids])]]=True
    positives=int(events.sum());negatives=int(eligible.sum()-positives)
    hits=int((selected & events).sum())
    enough=positives>=minimum and negatives>=minimum
    pos,neg=score[events],score[eligible & ~events]
    auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(positives*negatives)) if enough else None
    return {'eligible_count':len(ids),'excluded_already_high_count':int((~eligible).sum()),
            'event_count':positives,'non_event_count':negatives,'selected_count':int(selected.sum()),'hits':hits,
            'auroc':auc,'precision':hits/int(selected.sum()) if enough else None,
            'recall':hits/positives if enough else None,
            'status':'defined; descriptive' if enough else f'unstable/undefined: require >= {minimum} events and non-events',
            'event_indices':np.flatnonzero(events).tolist(),
            'missed_event_indices':np.flatnonzero(events & ~selected).tolist()}


def distribution(values):
    q10,median,q90=np.quantile(values,[.1,.5,.9])
    gap=q90-q10
    return {'q10':float(q10),'median':float(median),'q90':float(q90),'q90_minus_q10':float(gap),
            'gap_over_median':float(gap/median) if median>0 else None}


def analyze(error,uncertainty,rms,width,covered,replicates=2000,seed=8401):
    arrays=[np.asarray(v) for v in (error,uncertainty,rms,width,covered)]
    if any(v.shape!=arrays[0].shape for v in arrays) or arrays[0].ndim!=2 or arrays[0].shape[1]!=10:
        raise ValueError('All inputs must have shape N,10 excluding time zero')
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('Nonfinite analysis arrays')
    error,uncertainty,rms,width,covered=arrays
    n=len(error)
    indices=np.random.default_rng(seed).integers(n,size=(replicates,n))
    trends=[]
    for h in range(10):
        row={'time':(h+1)/10,'n_trajectories':n,'coverage90':float(covered[:,h].mean())}
        for key,value in [('E',error),('U_rel',uncertainty),('U_rms',rms),('W_rel90',width)]:
            row['mean_'+key]=float(value[:,h].mean());row['median_'+key]=float(np.median(value[:,h]))
        for key,value in [('U_rel',uncertainty),('W_rel90',width)]:
            row.update({key+'_'+k:v for k,v in distribution(value[:,h]).items()})
        row['current_error_association']=association(uncertainty[:,h],error[:,h],indices)
        trends.append(row)
    early=[]
    for lag in (1,2):
        for h,target in early_pairs(delta=lag):
            u,current,future,delta=aligned(error,uncertainty,h,lag)
            noncoverage=~covered[:,h].astype(bool)
            ua=association(u,future,indices)
            row={'time':(h+1)/10,'target_time':(target+1)/10,'lag':lag,'n_trajectories':n,
                 'uncertainty_future':ua,'current_error_future':association(current,future,indices),
                 'partial_controlling_current':partial_association(u,future,current,indices),
                 'uncertainty_delta_error':association(u,delta,indices),
                 'mean_delta_error':float(delta.mean()),'new_failure':new_failure(u,current,future),
                 # At a fixed source horizon W=2*q90*U_rel, a frozen positive scale.
                 'width_future':association(width[:,h],future,indices),
                 'noncoverage_future':association(noncoverage.astype(float),future,indices),
                 'source_noncoverage_count':int(noncoverage.sum()),
                 'future_error_mean_given_noncoverage':float(future[noncoverage].mean()) if noncoverage.any() else None,
                 'future_error_mean_given_coverage':float(future[~noncoverage].mean()) if (~noncoverage).any() else None}
            early.append(row)
    du,de=increments(uncertainty),increments(error)
    growth=[{'time':(h+1)/10,'target_time':(h+2)/10,**association(du[:,h],de[:,h],indices)} for h in range(9)]
    result={'by_horizon':trends,'early_warning':early,'growth_by_horizon':growth,
            'pooled_growth_secondary_source_horizon_confounded':association(du,de,indices)}
    return result,indices
