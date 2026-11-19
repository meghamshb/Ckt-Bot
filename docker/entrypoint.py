"""Explicit CPU packaging adapter around an unchanged private runtime copy."""
from pathlib import Path
import json
import os
import shutil
import signal
import subprocess
import sys
import time

SOURCE = Path('/opt/kbot-source')
STATE = Path('/state')
RUNTIME = STATE / 'runtime'


def main():
    if sys.argv[1:]:
        raise SystemExit('NO POSITIONAL ARGUMENTS. SEE docker/README.md FOR LOGIN AND MOUNTS.')
    cli = Path('/opt/codex/codex')
    for path in [cli, Path('/vendor/circuitvision/.git'), Path('/vendor/sam2/.git'),
                 Path('/models/yolo26n-final50.pt'), Path('/models/sam2.1_hiera_large.pt'),
                 Path('/models/circuitvision_finetuned.pth')]:
        if not path.exists():
            raise SystemExit('REQUIRED PRIVATE MOUNT MISSING: ' + str(path))
    STATE.mkdir(exist_ok=True)
    # Keep source checkouts in private state. Worktree .git pointer files cannot
    # be transplanted from a host into a Linux container.
    local_vendor = STATE / 'vendor'
    for name in ['circuitvision', 'sam2']:
        approved = Path('/vendor') / name
        if not (approved / '.git').is_dir():
            raise SystemExit('USE AN INDEPENDENT GIT CLONE, NOT A LINKED WORKTREE: ' + str(approved))
        target = local_vendor / name
        if not target.exists():
            shutil.copytree(approved, target, symlinks=True)
    # The state volume survives --rm; the container user site does not. Restore
    # this local editable installation on every new container, without downloads.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--no-deps',
                    '--no-build-isolation', '-e', str(local_vendor / 'sam2')], check=True)
    if not RUNTIME.exists():
        subprocess.run([sys.executable, str(SOURCE / 'scripts/configure-local.py'),
                        '--output', str(RUNTIME), '--yolo', '/models/yolo26n-final50.pt',
                        '--sam-artifacts', '/models', '--vendor-root', str(local_vendor / 'circuitvision'),
                        '--sam2-root', str(local_vendor / 'sam2'), '--codex', str(cli),
                        '--device', 'cpu'], check=True)
    receipt = json.loads((RUNTIME / 'provenance/DERIVED_ENVIRONMENT.json').read_text())
    import hashlib
    current = hashlib.sha256((SOURCE / 'CONFIG_FREEZE.json').read_bytes()).hexdigest()
    if receipt['source_CONFIG_FREEZE_sha256'] != current:
        raise SystemExit('STATE VOLUME BELONGS TO A DIFFERENT SOURCE FREEZE. USE A NEW PRIVATE STATE VOLUME.')
    subprocess.run([str(RUNTIME / 'check-ready')], check=True)
    # The frozen web server binds only loopback. This TCP forwarder exposes it
    # inside the container without modifying any Web UI or pipeline source.
    forwarder = subprocess.Popen(['socat', 'TCP-LISTEN:8766,reuseaddr,fork', 'TCP:127.0.0.1:8767'])
    server = subprocess.Popen([str(RUNTIME / 'run-web'), '--port', '8767'])

    def stop(signum, frame):
        server.terminate()
        forwarder.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while server.poll() is None and forwarder.poll() is None:
            time.sleep(.25)
        code = server.poll() if server.poll() is not None else 1
    finally:
        for process in [server, forwarder]:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
