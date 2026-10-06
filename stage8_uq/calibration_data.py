"""New ID archives via the unchanged Stage 2 generator; identity checks only on old data."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset_generation import GenerationConfig, generate_dataset, save_dataset, load_dataset
from stage4_evaluation.checkpoint import sha256, verify_hash

SEEDS = {'calibration': 8302, 'audit': 8303}
IDENTITY_KEYS = ('coefficients', 'blob_counts', 'blob_mask', 'blob_centers',
                 'blob_widths', 'blob_amplitudes')


def config(role):
    if role not in SEEDS:
        raise ValueError('Only new calibration and audit ID archives allowed')
    return GenerationConfig(num_trajectories=256, grid_size=64, num_snapshots=11,
                            t_final=1.0, seed=SEEDS[role])


def identities(data):
    """Hash physical input parameters, independent of archive-local IDs and grid size."""
    return [hashlib.sha256(b''.join(np.ascontiguousarray(data[key][i]).tobytes()
                                  for key in IDENTITY_KEYS)).hexdigest()
            for i in range(len(data['coefficients']))]


def require_disjoint(candidate, previous):
    if len(set(candidate)) != len(candidate) or set(candidate).intersection(previous):
        raise ValueError('Duplicate physical input across archive roles')


def require_unused_seeds(used):
    if len(set(SEEDS.values())) != 2 or set(SEEDS.values()).intersection(used):
        raise ValueError('Calibration/audit seeds must be unused and different')


def historical_inventory():
    """Read metadata/input parameters only, including OOD solely to rule out duplicates."""
    inventory, previous = {}, set()
    for path in sorted(Path('data').rglob('*.npz')):
        if 'stage8' in path.parts:
            continue
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(archive['metadata_json'].item())
            ids = identities({key: archive[key] for key in IDENTITY_KEYS})
        inventory[str(path)] = {'sha256': sha256(path), 'seed': meta['config']['seed'],
                               'input_fingerprints': ids}
        previous.update(ids)
    return inventory, previous


def write_new_json(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def generate_archives():
    root = Path('data/stage8')
    if root.exists():
        raise FileExistsError('Refuse to overwrite or regenerate existing Stage 8 data')
    inventory, previous = historical_inventory()
    used = {v['seed'] for v in inventory.values()}
    freeze = json.loads(Path('docs/results/stage8b_freeze.json').read_text())
    used.update(v for m in freeze['members'] for v in (m['config']['seed'], m['config']['loader_seed']))
    used.add(freeze['bootstrap_seed'])
    require_unused_seeds(used)
    root.mkdir(parents=True)
    manifest = {'historical_identity_inventory': inventory, 'previous_seeds': sorted(used),
                'seeds_verified_unused_before_generation': True, 'archives': {}}
    for role in SEEDS:
        data = generate_dataset(config(role))
        ids = identities(data)
        require_disjoint(ids, previous)
        previous.update(ids)
        path = root / f'{role}.npz'
        save_dataset(path, data)
        fields = data['fields']
        mass = fields.sum(axis=(-2,-1))*(2*np.pi/64)**2
        manifest['archives'][role] = {
            'path': str(path), 'sha256': sha256(path), 'config': asdict(config(role)),
            'input_fingerprints': ids, 'disjoint_from_previous': True,
            'finite': bool(np.isfinite(fields).all()), 'minimum': float(fields.min()),
            'max_mass_drift': float(np.max(np.abs(mass-mass[:,:1])))}
        print('Generated', role, path, 'with unused seed', SEEDS[role], flush=True)
    write_new_json(root/'manifest.json', manifest)
    return manifest


def verified_archive(role):
    config(role)  # Fail before touching a historical or unknown role.
    manifest = json.loads(Path('data/stage8/manifest.json').read_text())
    entry = manifest['archives'][role]
    if entry['config'] != json.loads(json.dumps(asdict(config(role)))):
        raise ValueError('New archive configuration differs')
    if entry['path'] != f'data/stage8/{role}.npz':
        raise ValueError('Unexpected archive path')
    verify_hash(entry['path'], entry['sha256'])
    data = load_dataset(entry['path'])
    if identities(data) != entry['input_fingerprints']:
        raise ValueError('Archive input identities changed')
    return data, entry
