"""Basic historical-ID technical check only; no OOD, calibration, or correlations."""
import json
from pathlib import Path
import numpy as np
import torch
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import write_json, sha256, verify_hash
from stage4_evaluation.diagnostics import physical_diagnostics
from stage8_uq.checkpoint import frozen_protocol, load_member, ARCHIVE
from stage8_uq.ensemble import predict_members, rollout_members


def main():
    root = Path('runs/stage8')
    manifest = json.loads((root / 'ensemble_manifest.json').read_text())
    if not manifest['all_members_verified']:
        raise ValueError('Ensemble not frozen')
    frozen = frozen_protocol()
    models = []
    for i in range(5):
        verify_hash(root / f'member_{i}' / 'manifest.json', manifest['member_manifest_sha256'][str(i)])
        model, norm, _ = load_member(root / f'member_{i}', frozen)
        models.append(model)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data = load_dataset(ARCHIVE)
    ids = frozen['split_ids']['test']
    fields, coefficients = data['fields'][ids], data['coefficients'][ids]
    one = predict_members(models, norm, fields[:, 0], coefficients)
    ar = rollout_members(models, norm, fields, coefficients, data['times'])
    np.testing.assert_array_equal(ar['spread'][:, 0], 0)
    np.testing.assert_array_equal(ar['mean'][:, 0, 0], fields[:, 0])
    np.testing.assert_array_equal(ar['members'][:, :, 1], one['members'])
    metrics = physical_diagnostics(ar['mean'][:, :, 0], fields, fields[:, 0])
    output = root / 'id_check'
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'raw_predictions.npz',
                        **{f'one_step_{k}': v for k, v in one.items()},
                        **{f'rollout_{k}': v for k, v in ar.items()},
                        times=data['times'], trajectory_ids=np.asarray(ids),
                        coefficients=coefficients, mean_relative_l2=metrics['relative_l2'])
    summary = {'role': 'historical_ID_technical_check_only', 'trajectory_ids': ids,
               'times': data['times'].tolist(), 'one_step_member_shape': list(one['members'].shape),
               'rollout_member_shape': list(ar['members'].shape),
               'rollout_mean_shape': list(ar['mean'].shape),
               'one_step_U_rms_mean': float(one['U_rms'].mean()),
               'one_step_U_rel_mean': float(one['U_rel'].mean()),
               'U_rms_mean_by_horizon': ar['U_rms'].mean(axis=0).tolist(),
               'U_rel_mean_by_horizon': ar['U_rel'].mean(axis=0).tolist(),
               'ensemble_mean_relative_l2_by_horizon': metrics['relative_l2'].mean(axis=0).tolist(),
               'floor_activations': int(ar['floor_active'].sum()),
               'epsilon_rms': ar['epsilon_rms'], 'nonfinite_members': [],
               'raw_predictions_sha256': sha256(output / 'raw_predictions.npz'),
               'calibrated': False, 'formal_error_awareness_conclusion': False}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
