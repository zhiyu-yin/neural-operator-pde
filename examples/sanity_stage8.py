"""Run the complete regression/sanity suite before authorizing ensemble training."""
import subprocess
import sys
from pathlib import Path
from stage4_evaluation.checkpoint import sha256, write_json
from stage8_uq.checkpoint import frozen_protocol, source_hashes


def main():
    root = Path('runs/stage8')
    frozen_protocol()  # Requires real historical artifacts even on a public-only checkout.
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True)
    log = result.stdout + result.stderr
    (root / 'sanity_pytest.txt').write_text(log)
    print(log, flush=True)
    if result.returncode:
        raise SystemExit('Sanity failed; full training must not start')
    write_json(root / 'sanity.json', {'passed': True, 'pytest_log_sha256': sha256(root / 'sanity_pytest.txt'),
               'source_sha256': source_hashes(),
               'test_sha256': {str(p): sha256(p) for p in sorted(Path('tests').glob('*.py'))}})


if __name__ == '__main__':
    main()
