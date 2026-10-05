"""Physical-unit predictions and independent member-wise autoregressive feedback."""
import numpy as np
import torch
from stage4_evaluation.rollout import prepare_input, predict_trajectories
from .metrics import ensemble_statistics, spread_scores


@torch.inference_mode()
def predict_members(models, normalization, fields, coefficients):
    """Stack physical one-step predictions as (M,B,1,N,N)."""
    inputs = prepare_input(fields, coefficients, normalization)
    predictions = []
    for model in models:
        model.eval()
        predictions.append(normalization.inverse_field(model(inputs).double()).cpu().numpy())
    members = np.stack(predictions)
    mean, spread = ensemble_statistics(members)
    return {'members': members, 'mean': mean, 'spread': spread,
            **spread_scores(mean, spread, normalization.field_std)}


def rollout_members(models, normalization, fields, coefficients, times):
    """Reuse Stage 4 feedback separately per member; never feed back the mean."""
    predictions = []
    for member_id, model in enumerate(models):
        outputs, failures = predict_trajectories(model, normalization, fields, coefficients, times)
        if any(failures.values()):
            raise FloatingPointError(f'Member {member_id} failed: {failures}')
        predictions.append(outputs['autoregressive'][:, :, None])
    members = np.stack(predictions)  # M,B,S,1,N,N
    mean, spread = ensemble_statistics(members)
    return {'members': members, 'mean': mean, 'spread': spread,
            **spread_scores(mean, spread, normalization.field_std)}
