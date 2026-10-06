"""Field-level split-conformal scaling, with no individual-cell calibration trials."""
import math
from statistics import NormalDist
import numpy as np
from .awareness import prediction_horizons

LEVELS = (.5, .7, .8, .9, .95)


def residual_scores(truth, mean, spread, times, epsilon):
    """Return one max score per trajectory/horizon and explicit floor counts."""
    hs = prediction_horizons(times)
    truth, mean, spread = (np.asarray(v, dtype=float) for v in (truth, mean, spread))
    if truth.shape != mean.shape or mean.shape != spread.shape or truth.ndim != 4:
        raise ValueError('Expected matching trajectory/time/x/y arrays')
    if not all(np.isfinite(v).all() for v in (truth,mean,spread)) or np.any(spread < 0) or epsilon <= 0:
        raise ValueError('Invalid fields, spread, or floor')
    z = np.abs(truth[:,hs]-mean[:,hs])/np.maximum(spread[:,hs],epsilon)
    floor = spread[:,hs] < epsilon
    return {'scores':z.max(axis=(-2,-1)),
            'floor_cells_by_horizon':floor.sum(axis=(0,2,3)),
            'floor_fields_by_horizon':floor.any(axis=(-2,-1)).sum(axis=0)}


def conformal_factors(scores, role, levels=LEVELS):
    """Fit only calibration scores; explicitly use the clipped higher order statistic."""
    if role != 'calibration':
        raise ValueError('Only calibration archive may fit factors')
    scores = np.asarray(scores, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 10 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Expected finite nonnegative calibration trajectory/horizon scores')
    if np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError('Coverage levels must increase within (0,1)')
    requested = np.ceil((len(scores)+1)*levels).astype(int)
    indices = np.minimum(len(scores),requested)
    q = np.sort(scores,axis=0)[indices-1].T
    return {'q':q, 'k':indices, 'unclipped_k':requested, 'clipped':requested>len(scores)}


def wilson_interval(successes, n):
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError('Invalid binomial counts')
    z = NormalDist().inv_cdf(.975)
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def band_diagnostics(truth, mean, spread, q, epsilon):
    """One binary field event per trajectory/horizon; widths in physical and RMS units.

    Band uses raw spread exactly as requested, even where the score floor activates.
    The caller must report any resulting lack of score/band equivalence.
    """
    half = np.asarray(q)[None,:,None,None]*spread
    lower, upper = mean-half, mean+half
    covered = (truth >= lower) & (truth <= upper)
    width = upper-lower
    mean_rms = np.sqrt(np.mean(mean**2,axis=(-2,-1)))
    return {'field_covered':covered.all(axis=(-2,-1)),
            'pointwise_fraction_secondary':covered.mean(axis=(-2,-1)),
            'mean_physical_width':width.mean(axis=(-2,-1)),
            'rms_physical_width':np.sqrt(np.mean(width**2,axis=(-2,-1))),
            'W_rel':np.sqrt(np.mean(width**2,axis=(-2,-1)))/np.maximum(mean_rms,epsilon),
            'lower':lower,'upper':upper}


def reliability_rows(diagnostics, times, levels=LEVELS):
    """No cell counts enter binomial intervals; diagnostics shape is N,H,L."""
    rows=[]
    coverage=diagnostics['field_covered']
    n,horizons,nlevels=coverage.shape
    if horizons != 10 or nlevels != len(levels) or len(times) != 10:
        raise ValueError('Reliability shape mismatch')
    for h,t in enumerate(times):
        for l,level in enumerate(levels):
            count=int(coverage[:,h,l].sum())
            row={'time':float(t),'nominal':float(level),'n_trajectories':n,'covered_fields':count,
                 'simultaneous_coverage':count/n,'coverage_error':count/n-level,
                 'coverage_ci95_low':wilson_interval(count,n)[0],
                 'coverage_ci95_high':wilson_interval(count,n)[1]}
            for key in ('mean_physical_width','rms_physical_width','W_rel','pointwise_fraction_secondary'):
                row['mean_'+key]=float(diagnostics[key][:,h,l].mean())
                row['median_'+key]=float(np.median(diagnostics[key][:,h,l]))
            rows.append(row)
    return rows


def grouped_summary(diagnostics, levels=LEVELS, replicates=2000, seed=8304):
    """Average ten horizons per trajectory, then bootstrap those independent units."""
    events=diagnostics['field_covered'].astype(float)
    n=len(events)
    indices=np.random.default_rng(seed).integers(n,size=(replicates,n))
    rows=[]
    for l,level in enumerate(levels):
        per_trajectory=events[:,:,l].mean(axis=1)
        boot=per_trajectory[indices].mean(axis=1)
        rows.append({'nominal':float(level),'n_trajectories':n,
                     'mean_horizon_coverage':float(per_trajectory.mean()),
                     'coverage_error':float(per_trajectory.mean()-level),
                     'trajectory_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),
                     'mean_W_rel':float(diagnostics['W_rel'][:,:,l].mean()),
                     'median_trajectory_mean_W_rel':float(np.median(diagnostics['W_rel'][:,:,l].mean(axis=1))),
                     'pointwise_fraction_secondary':float(diagnostics['pointwise_fraction_secondary'][:,:,l].mean())})
    return rows,indices
