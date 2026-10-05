"""Strict provenance validation; never refit normalization or overwrite historical D."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import torch
from neural_operator.fno import FNO2d
from neural_operator.normalization import Normalization
from stage4_evaluation.checkpoint import sha256, verify_hash, write_json
from stage5_training.training import state_hash
from stage6_control.provenance import frozen_context, load_d, verify_snapshot
from stage6_control.config import Config
from .config import ARCHITECTURE, BASELINE, D_HASH, TAG, member_config

D_RUN = Path('runs/stage6')
ARCHIVE = Path('data/dev/trajectories.npz')


def object_hash(value: dict) -> str:
    """Canonical JSON hash for exact normalization/protocol comparison."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def frozen_protocol() -> dict:
    """Verify D, historical sources, data and split/normalization provenance."""
    protocol = json.loads((D_RUN / 'protocol.json').read_text())
    model, norm, selection = load_d(D_RUN)
    if selection['sha256'] != D_HASH or model.config != ARCHITECTURE:
        raise ValueError('Unexpected frozen D')
    previous_norm, baseline, previous, initial = frozen_context(Config())
    for key in ('architecture', 'normalization', 'split_ids', 'initial_state_sha256'):
        if protocol[key] != previous[key]:
            raise ValueError(f'D/Stage 5 mismatch: {key}')
    if norm.to_dict() != previous_norm.to_dict() or protocol['config'] != Config().to_dict():
        raise ValueError('Frozen configuration/normalization mismatch')
    verify_hash(ARCHIVE, protocol['training_archive_sha256'])
    verify_hash('runs/stage5/protocol.json', protocol['stage5_protocol_sha256'])
    if protocol['split_ids'] != baseline['split_ids']:
        raise ValueError('Frozen split mismatch')
    return protocol


def source_hashes() -> dict:
    """Hash uncommitted Stage 8 source too; revision alone is insufficient."""
    paths = sorted(Path('stage8_uq').glob('*.py')) + sorted(Path('examples').glob('*stage8*.py'))
    return {str(path): sha256(path) for path in paths}


def verify_protected(snapshot: dict) -> int:
    """Require byte-identical historical files, except exact package registration."""
    verify_snapshot({p: h for p, h in snapshot.items() if p != 'pyproject.toml'})
    original = subprocess.check_output(['git', 'show', f'{BASELINE}:pyproject.toml'], text=True)
    if hashlib.sha256(original.encode()).hexdigest() != snapshot['pyproject.toml']:
        raise ValueError('Original packaging hash mismatch')
    expected = original.replace('"stage6_control*"]', '"stage6_control*", "stage8_uq*"]')
    if Path('pyproject.toml').read_text() != expected:
        raise ValueError('Unexpected historical packaging change')
    return len(snapshot) - 1


def make_protocol(member_id: int, initial_hash: str, frozen: dict) -> dict:
    """Copy D protocol, changing only member seeds and provenance annotations."""
    return {
        'member_id': member_id, 'config': member_config(member_id).to_dict(),
        'architecture': frozen['architecture'], 'normalization': frozen['normalization'],
        'normalization_sha256': object_hash(frozen['normalization']),
        'split_ids': frozen['split_ids'], 'initial_state_sha256': initial_hash,
        'optimizer': frozen['optimizer'], 'objective': frozen['objective'],
        'selection_rule': frozen['selection_rule'], 'prediction_interval': 0.1,
        'training_archive_sha256': frozen['training_archive_sha256'],
        'trainable_parameter_elements': frozen['trainable_parameter_elements'],
        'trainable_real_scalar_parameters': frozen['trainable_real_scalar_parameters'],
        'pre_uq_revision': BASELINE, 'pre_uq_tag': TAG,
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'frozen_d_protocol_sha256': sha256(D_RUN / 'protocol.json'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
        'source_sha256': source_hashes(), 'device': 'cpu',
    }


def freeze_manifest(folder: Path, protocol: dict, selection: dict, gate: dict,
                    checkpoint_path: Path, reused: bool = False) -> dict:
    """Record a completed member, including the full budget and validation-only selection."""
    manifest = {**protocol, 'checkpoint_path': str(checkpoint_path),
                'checkpoint_sha256': selection['sha256'], 'selected_epoch': selection['epoch'],
                'full_training_epochs': selection['completed_budget']['epoch'],
                'completed_budget': selection['completed_budget'],
                'validation': selection['validation'], 'runtime_seconds': selection['runtime_seconds'],
                'training_only_seconds': selection['training_only_seconds'],
                'gate_runtime_seconds': gate['runtime_seconds'], 'gate_updates': gate['updates'],
                'gate_passed': gate['passed'], 'reused': reused, 'test_or_ood_used': False,
                'protocol_sha256': sha256(folder / 'protocol.json'),
                'selection_sha256': sha256(folder / 'selection.json'),
                'gate_sha256': sha256(folder / 'gate.json')}
    write_json(folder / 'manifest.json', manifest)
    return manifest


def register_member_zero(root: Path, frozen: dict) -> dict:
    """Reference (do not copy or retrain) the exact historical selected checkpoint."""
    folder = root / 'member_0'
    folder.mkdir(exist_ok=False)
    selection = json.loads((D_RUN / 'selection.json').read_text())
    gate = json.loads((D_RUN / 'gate.json').read_text())
    protocol = make_protocol(0, frozen['initial_state_sha256'], frozen)
    # Runtime/versions describe the historical training, not this registration.
    protocol['versions'] = frozen['versions']
    protocol['historical_training_source_revision'] = frozen['git_revision']
    for name, value in [('protocol', protocol), ('selection', selection), ('gate', gate)]:
        write_json(folder / f'{name}.json', value)
    return freeze_manifest(folder, protocol, selection, gate, D_RUN / 'best.pt', reused=True)


def load_member(folder: str | Path, frozen: dict | None = None):
    """Load a completed, hash-checked member as an inference-only model."""
    folder = Path(folder)
    frozen = frozen_protocol() if frozen is None else frozen
    manifest = json.loads((folder / 'manifest.json').read_text())
    member_id = manifest['member_id']
    if manifest['config'] != member_config(member_id).to_dict():
        raise ValueError('Member seeds or training protocol mismatch')
    for key in ('architecture', 'normalization', 'split_ids', 'optimizer', 'objective',
                'selection_rule', 'training_archive_sha256', 'trainable_parameter_elements',
                'trainable_real_scalar_parameters'):
        if manifest[key] != frozen[key]:
            raise ValueError(f'Member frozen metadata mismatch: {key}')
    if manifest['normalization_sha256'] != object_hash(frozen['normalization']):
        raise ValueError('Normalization hash mismatch')
    if (not manifest['gate_passed'] or manifest['test_or_ood_used'] or
            manifest['full_training_epochs'] != 60 or
            manifest['completed_budget'] != frozen['completed_budget'] or
            not 1 <= manifest['selected_epoch'] <= 60):
        raise ValueError('Incomplete or invalid member selection')
    for name in ('protocol', 'selection', 'gate'):
        verify_hash(folder / f'{name}.json', manifest[f'{name}_sha256'])
    protocol = json.loads((folder / 'protocol.json').read_text())
    if any(manifest[key] != value for key, value in protocol.items()):
        raise ValueError('Manifest/protocol mismatch')
    verify_snapshot(manifest['source_sha256'])
    verify_hash(ARCHIVE, manifest['training_archive_sha256'])
    verify_hash(D_RUN / 'protocol.json', manifest['frozen_d_protocol_sha256'])
    path = Path(manifest['checkpoint_path'])
    verify_hash(path, manifest['checkpoint_sha256'])
    if member_id == 0:
        if path != D_RUN / 'best.pt' or manifest['checkpoint_sha256'] != D_HASH or not manifest['reused']:
            raise ValueError('Member 0 must be the historical checkpoint')
        model, norm, selected = load_d(D_RUN)
    else:
        if manifest['reused'] or path != folder / 'best.pt':
            raise ValueError('New member checkpoint path mismatch')
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('architecture', 'normalization', 'split_ids', 'config', 'initial_state_sha256'):
            if state[key] != manifest[key]:
                raise ValueError(f'Checkpoint mismatch: {key}')
        model = FNO2d(**manifest['architecture'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        model.eval().requires_grad_(False)
        norm = Normalization(**manifest['normalization'])
        selected = {'epoch': state['epoch'], 'validation': state['validation']}
    if selected['epoch'] != manifest['selected_epoch'] or selected['validation'] != manifest['validation']:
        raise ValueError('Selected epoch/validation mismatch')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite checkpoint parameters')
    return model, norm, manifest
