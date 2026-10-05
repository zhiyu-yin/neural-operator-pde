"""Train only four declared ensemble members after a recorded passing full suite."""
import argparse
import json
from pathlib import Path
import time
from dataset_generation import load_dataset
from stage4_evaluation.checkpoint import sha256, write_json
from stage6_control.provenance import verify_snapshot
from stage8_uq.checkpoint import (ARCHIVE, frozen_protocol, load_member,
                                  register_member_zero, source_hashes, verify_protected)
from stage8_uq.training import train_member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/stage8'))
    args = parser.parse_args()
    root = args.output
    sanity = json.loads((root / 'sanity.json').read_text())
    if not sanity['passed'] or sanity['source_sha256'] != source_hashes():
        raise RuntimeError('A passing full-suite sanity result for the current source is required')
    verify_snapshot(sanity['test_sha256'])
    verify_protected(json.loads((root / 'protected_before.json').read_text()))
    frozen = frozen_protocol()
    data = load_dataset(ARCHIVE)
    started = time.perf_counter()
    manifests = [register_member_zero(root, frozen)]
    for member_id in range(1, 5):
        manifests.append(train_member(member_id, root, frozen, data))
    for member_id in range(5):
        load_member(root / f'member_{member_id}', frozen)
    summary = {'member_manifest_sha256': {str(i): sha256(root / f'member_{i}' / 'manifest.json') for i in range(5)},
               'wall_seconds': time.perf_counter() - started,
               'new_member_full_run_seconds': sum(m['runtime_seconds'] for m in manifests[1:]),
               'new_member_gate_seconds': sum(m['gate_runtime_seconds'] for m in manifests[1:]),
               'all_members_verified': True, 'evaluation_used_for_selection': False}
    write_json(root / 'ensemble_manifest.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
