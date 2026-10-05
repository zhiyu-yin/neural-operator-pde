"""Raw physical-unit ensemble spread: no calibration or correlation analysis."""
import numpy as np
from .config import EPSILON_FACTOR


def ensemble_statistics(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation (ddof=1) over the leading member axis.

    Accept one-step (M,B,1,N,N) or rollout (M,B,S,1,N,N) arrays.
    """
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim not in (5, 6) or values.shape[0] < 2 or values.shape[-3] != 1:
        raise ValueError('Expected at least two members and a singleton field channel')
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite member predictions; never silently drop members')
    # Shift by a member to avoid roundoff spread for exactly identical predictions.
    delta = values - values[0:1]
    average = delta.mean(axis=0)
    mean = values[0] + average
    spread = np.sqrt(np.sum((delta - average)**2, axis=0) / (len(values) - 1))
    return mean, spread


def spread_scores(mean: np.ndarray, spread: np.ndarray, field_std: float) -> dict:
    """Per-trajectory(/horizon) RMS scores with a resolution-independent floor."""
    mean, spread = np.asarray(mean, dtype=np.float64), np.asarray(spread, dtype=np.float64)
    if mean.shape != spread.shape or mean.ndim not in (4, 5) or mean.shape[-3] != 1:
        raise ValueError('Mean/spread shape mismatch')
    if not np.isfinite(field_std) or field_std <= 0:
        raise ValueError('Frozen field std must be finite and positive')
    if not np.isfinite(mean).all() or not np.isfinite(spread).all() or np.any(spread < 0):
        raise ValueError('Invalid physical mean/spread')
    rms = np.sqrt(np.mean(spread**2, axis=(-3, -2, -1)))
    mean_rms = np.sqrt(np.mean(mean**2, axis=(-3, -2, -1)))
    epsilon = EPSILON_FACTOR * field_std
    return {'U_rms': rms, 'U_rel': rms / np.maximum(mean_rms, epsilon),
            'floor_active': mean_rms <= epsilon, 'epsilon_rms': epsilon}
