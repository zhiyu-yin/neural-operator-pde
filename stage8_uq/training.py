"""Member-specific orchestration around the unchanged Model D training functions."""
import json
from pathlib import Path
import time
import torch
from neural_operator.data import TrajectoryPairDataset, validate_baseline
from neural_operator.normalization import Normalization
from stage4_evaluation.checkpoint import sha256, write_json
from stage5_training.training import fresh_model, state_hash, loader, selection_key
from stage6_control.config import budget
from stage6_control.training import tiny_gate, train_epoch, validation_metrics
from .config import OPTIMIZER, member_config
from .checkpoint import make_protocol, freeze_manifest


def training_inputs(data, frozen):
    """Expose only the ordered training pairs and validation IDs to training."""
    validate_baseline(data)
    norm = Normalization(**frozen['normalization'])
    splits = frozen['split_ids']
    joined = sum((splits[k] for k in ('train', 'validation', 'test')), [])
    if sorted(joined) != list(range(64)) or [len(splits[k]) for k in ('train', 'validation', 'test')] != [48, 8, 8]:
        raise ValueError('Invalid frozen trajectory assignments')
    dataset = TrajectoryPairDataset(data, splits['train'], norm)
    return dataset, list(splits['validation']), norm


def full_training(folder, protocol, data, dataset, validation_ids, norm, config):
    """Complete all 60 epochs; selection uses validation only, with D's tie breaks."""
    model = fresh_model(config, protocol['architecture'])
    if state_hash(model) != protocol['initial_state_sha256']:
        raise ValueError('Full training did not restart from the declared initialization')
    batches = loader(dataset, config, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), **OPTIMIZER)
    history, best, selected = [], None, None
    start, training_seconds = time.perf_counter(), 0.0
    for epoch in range(1, config.epochs + 1):
        tick = time.perf_counter()
        loss = train_epoch(model, batches, optimizer)
        training_seconds += time.perf_counter() - tick
        validation = validation_metrics(model, data, validation_ids, norm)
        key = selection_key(validation, epoch)
        if best is None or key < best:
            best = key
            state = {k: protocol[k] for k in ('architecture', 'normalization', 'split_ids', 'config', 'initial_state_sha256')}
            state.update(model_state_dict=model.state_dict(), epoch=epoch, validation=validation)
            torch.save(state, folder / 'best.pt')
            selected = {'epoch': epoch, 'validation': validation, 'sha256': sha256(folder / 'best.pt')}
        history.append({**budget(epoch), 'training_loss': loss, 'validation': validation,
                        'elapsed_wall_seconds': time.perf_counter() - start,
                        'cumulative_training_seconds': training_seconds})
        temporary = folder / 'history.tmp'
        temporary.write_text(json.dumps(history, indent=2, allow_nan=False) + '\n')
        temporary.replace(folder / 'history.json')
        print(f"member={protocol['member_id']} epoch={epoch}/60 loss={loss:.6g} val_AR={validation['final_relative_l2']:.6g}", flush=True)
    selection = {**selected, 'gate_passed': True, 'test_or_ood_used': False,
                 'runtime_seconds': time.perf_counter() - start, 'training_only_seconds': training_seconds,
                 'completed_budget': budget(config.epochs), 'selected_budget': budget(selected['epoch'])}
    write_json(folder / 'selection.json', selection)
    return selection


def train_member(member_id: int, root: Path, frozen: dict, data: dict) -> dict:
    """Run the unchanged training-only gate, discard it, and train a fresh member."""
    if member_id not in range(1, 5):
        raise ValueError('Only members 1..4 may be trained')
    folder = root / f'member_{member_id}'
    folder.mkdir(exist_ok=False)
    config = member_config(member_id)
    dataset, validation_ids, norm = training_inputs(data, frozen)
    initial = fresh_model(config, frozen['architecture'])
    protocol = make_protocol(member_id, state_hash(initial), frozen)
    del initial
    write_json(folder / 'protocol.json', protocol)
    try:
        gate = tiny_gate(dataset, norm, protocol['architecture'], config)
        write_json(folder / 'gate.json', gate)
        if not gate['passed']:
            raise RuntimeError('Training-only sanity gate failed; no full training')
        selected = full_training(folder, protocol, data, dataset, validation_ids, norm, config)
        return freeze_manifest(folder, protocol, selected, gate, folder / 'best.pt')
    except Exception as exc:
        write_json(folder / 'failure.json', {'member_id': member_id, 'reason': repr(exc),
                                          'replacement_seed_used': False})
        raise
