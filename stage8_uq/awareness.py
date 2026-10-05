"""Retrospective raw-spread association; trajectories are the sampling units."""
import numpy as np


def ranks(x):
    """Average ranks for ties, including bootstrap duplicates."""
    x = np.asarray(x)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(x, y, rank=False):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if rank:
        x, y = ranks(x), ranks(y)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.clip(x@y/denominator, -1, 1)) if denominator > 0 else None


def association(x, y, indices, demean=False):
    def coefficient(a, b, rank):
        if demean:
            a, b = a-a.mean(axis=0), b-b.mean(axis=0)
        return correlation(a, b, rank)
    result = {'n_trajectories': len(x)}
    for name, rank in [('pearson', False), ('spearman', True)]:
        result[name] = coefficient(x, y, rank)
        values = [coefficient(x[i], y[i], rank) for i in indices]
        valid = [v for v in values if v is not None]
        result[name+'_ci95'] = np.quantile(valid, [.025,.975]).tolist() if valid else None
        result[name+'_invalid_bootstraps'] = len(values)-len(valid)
    result['status'] = 'ok' if result['pearson'] is not None and result['spearman'] is not None else 'undefined: nonfinite or constant values, or fewer than three trajectories'
    return result


def prediction_horizons(times):
    times = np.asarray(times)
    if times.shape != (11,) or not np.allclose(times, np.arange(11)/10):
        raise ValueError('Requires existing ten prediction horizons')
    return np.arange(1,11)


def quintiles(score):
    score = np.asarray(score)
    if len(score) < 5 or not np.isfinite(score).all():
        raise ValueError('Quintiles require five finite cases')
    labels = np.empty(len(score), int)
    for q, ids in enumerate(np.array_split(np.argsort(score, kind='stable'), 5)):
        labels[ids] = q
    return labels


def top_fifth(values):
    values = np.asarray(values)
    mask = np.zeros(len(values), bool)
    mask[np.argsort(values, kind='stable')[-int(np.ceil(len(values)/5)):]] = True
    return mask


def detection(score, error):
    actual, selected = top_fifth(error), top_fifth(score)
    pos, neg = score[actual], score[~actual]
    auc = float(((pos[:,None] > neg).sum()+.5*(pos[:,None] == neg).sum())/(len(pos)*len(neg))) if len(pos)*len(neg) else None
    hits = int((actual & selected).sum())
    return {'auroc':auc, 'precision':hits/int(selected.sum()), 'recall':hits/int(actual.sum()),
            'high_error_count':int(actual.sum()), 'selected_count':int(selected.sum()), 'hits':hits}


def early_pairs(steps=10, delta=1):
    return [(h,h+delta) for h in range(steps-delta)]


def spatial_association(error, spread):
    """One coefficient per field; never pool trajectories or horizons."""
    if error.shape != spread.shape or error.ndim != 4:
        raise ValueError('Expected trajectory, horizon, x, y fields')
    return [[correlation(e,s,True) for e,s in zip(er,sr)] for er,sr in zip(error,spread)]


def analyze(score, rms, error, times, seed=8301, replicates=2000):
    hs = prediction_horizons(times)
    score, rms, error = score[:,hs], rms[:,hs], error[:,hs]
    if not all(np.isfinite(v).all() for v in (score,rms,error)):
        raise ValueError('Nonfinite analysis arrays')
    n = len(score)
    indices = np.random.default_rng(seed).integers(n,size=(replicates,n))
    rows, bin_rows = [], []
    labels = np.stack([quintiles(score[:,h]) for h in range(10)],axis=1)
    for h in range(10):
        rows.append({'time':float(times[h+1]), **association(score[:,h],error[:,h],indices),
                     'mean_error':float(error[:,h].mean()), 'median_error':float(np.median(error[:,h])),
                     'mean_U_rel':float(score[:,h].mean()), 'median_U_rel':float(np.median(score[:,h])),
                     'mean_U_rms':float(rms[:,h].mean()), **detection(score[:,h],error[:,h])})
        for q in range(5):
            values=error[labels[:,h]==q,h]
            bin_rows.append({'time':float(times[h+1]),'quintile':q+1,'n':len(values),'mean_error':float(values.mean()),'median_error':float(np.median(values))})
    aggregated=[]
    for q in range(5):
        # Equal trajectory weight: average each trajectory's errors during its bin visits.
        mask=labels==q; counts=mask.sum(axis=1); valid=counts>0
        per=np.divide((error*mask).sum(axis=1),counts,out=np.zeros(n),where=valid)
        boot=[float(per[i][valid[i]].mean()) for i in indices if valid[i].any()]
        aggregated.append({'quintile':q+1,'n_unique_trajectories':int(valid.sum()),
                           'mean_error':float(per[valid].mean()),'median_trajectory_mean_error':float(np.median(per[valid])),
                           'mean_ci95':np.quantile(boot,[.025,.975]).tolist()})
    horizon_balanced=[]
    for q in range(5):
        mask=labels==q
        boot=[]
        for i in indices:
            counts=mask[i].sum(axis=0)
            if np.all(counts):
                boot.append(float(((error[i]*mask[i]).sum(axis=0)/counts).mean()))
        horizon_balanced.append({'quintile':q+1,
            'mean_error':float(np.mean([r['mean_error'] for r in bin_rows if r['quintile']==q+1])),
            'mean_horizon_median_error':float(np.mean([r['median_error'] for r in bin_rows if r['quintile']==q+1])),
            'mean_ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None,
            'invalid_bootstraps':len(indices)-len(boot)})
    early=[]
    for delta in (1,2):
        for h,k in early_pairs(delta=delta):
            early.append({'time':float(times[h+1]),'target_time':float(times[k+1]),'delta_steps':delta,**association(score[:,h],error[:,k],indices)})
    return {'by_horizon':rows,'quintiles_by_horizon':bin_rows,'quintiles_horizon_balanced':horizon_balanced,'quintiles_trajectory_weighted':aggregated,
            'quintile_mean_monotonic_by_horizon':[bool(np.all(np.diff([r['mean_error'] for r in bin_rows[h*5:h*5+5]])>=0)) for h in range(10)],
            'pooled_secondary_horizon_confounded':association(score,error,indices),
            'horizon_demeaned_secondary':association(score,error,indices,demean=True),
            'early_warning':early,
            'confidently_wrong_definition':'top ceil(n/5) error and bottom uncertainty quintile within horizon',
            'confidently_wrong_indices':[[int(i) for i in np.flatnonzero(top_fifth(error[:,h]) & (labels[:,h]==0))] for h in range(10)]}
