#!/usr/bin/env python3
"""Prepare a separate machine-specific runtime; never edit the frozen source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--yolo', type=Path, required=True)
    parser.add_argument('--sam-artifacts', type=Path, required=True)
    parser.add_argument('--vendor-root', type=Path, required=True)
    parser.add_argument('--sam2-root', type=Path, required=True)
    parser.add_argument('--codex', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    if output.exists() or source == output or source in output.parents:
        parser.error('--output MUST BE NEW AND OUTSIDE THE FROZEN SOURCE REPOSITORY')
    if not Path('/usr/bin/python3').is_file():
        parser.error('FROZEN LAUNCHERS REQUIRE /usr/bin/python3')
    config = json.loads((source / 'runtime/config/pipeline.json').read_text())
    sam = json.loads((source / 'runtime/config/sam.json').read_text())
    original = json.loads((source / 'CONFIG_FREEZE.json').read_text())
    paths = {name: getattr(args, name).expanduser().resolve()
             for name in ['yolo', 'sam_artifacts', 'vendor_root', 'sam2_root', 'codex']}
    if not paths['codex'].is_file() or not os.access(paths['codex'], os.X_OK):
        parser.error('A COMPATIBLE EXECUTABLE CODEX CLI IS REQUIRED')
    if digest(paths['yolo']) != config['detector']['sha256']:
        parser.error('YOLO CHECKPOINT DOES NOT MATCH THE FROZEN MODEL HASH')
    sam['vendor_root'] = str(paths['vendor_root'])
    sam['sam2_root'] = str(paths['sam2_root'])
    for name in ['base_checkpoint', 'finetuned_checkpoint']:
        record = sam['artifacts'][name]
        record['path'] = str(paths['sam_artifacts'] / Path(record['path']).name)
    sam['artifacts']['config']['path'] = str(paths['sam2_root'] / 'sam2/configs/sam2.1/sam2.1_hiera_l.yaml')
    sam['runtime']['device'] = args.device
    sam['runtime']['mps_fraction'] = .35 if args.device == 'mps' else None
    # Dependency identities are captured for this new environment, not asserted
    # equal to the historically measured Mac runtime.
    versions = {name: importlib.metadata.version(name) for name in original['package_versions']}
    sam['runtime']['package_versions'] = {name: versions[name] for name in sam['runtime']['package_versions']}
    sys.path.insert(0, str(source / 'runtime/src'))
    from kbot_pipeline._sam_backend import validate_lock
    from kbot_pipeline.codex_backend import cli_metadata, invocation_command
    validate_lock(sam)  # Source commits, pinned source hashes and all SAM artifacts.
    spec = importlib.util.find_spec('sam2')
    if spec is None or Path(spec.origin).resolve() != paths['sam2_root'] / 'sam2/__init__.py':
        parser.error('INSTALLED SAM2 MUST RESOLVE TO THE APPROVED SOURCE CHECKOUT')
    if args.device == 'mps':
        import torch
        if not torch.backends.mps.is_available():
            parser.error('REQUESTED MPS DEVICE IS UNAVAILABLE; NO DEVICE FALLBACK')
    config['python'] = sys.executable
    config['lock_file'] = str(output / '.kbot-heavy-job.lock')
    for role in ['vision', 'reasoning']:
        config[role]['cli'] = str(paths['codex'])
    metadata = cli_metadata(config['reasoning'])
    # Validate the actual frozen command-line vocabulary without inference.
    help_command = invocation_command(config['reasoning'], source, source / 'runtime/config/pipeline.json',
                                      output / 'unused.json', [])
    help_command[-1:] = ['--help']
    checked = subprocess.run(help_command, capture_output=True, text=True, timeout=20)
    if checked.returncode:
        parser.error('CODEX CLI DOES NOT ACCEPT THE FROZEN FLAGS; USE A COMPATIBLE CLI. NO FLAGS WERE REMOVED.')
    spice = Path('/opt/homebrew/bin/ngspice')
    if not spice.is_file():
        parser.error('FROZEN SPICE MODULE REQUIRES /opt/homebrew/bin/ngspice (SEE DOCKER NOTES ON LINUX)')
    before = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(source).parts}
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc',
                    'UI-RUNS', 'readiness-records', 'server-logs', '.venv', '.env'))
    target_yolo = output / 'runtime' / config['detector']['weights']
    target_yolo.parent.mkdir(parents=True, exist_ok=True)
    target_yolo.symlink_to(paths['yolo'])
    write(output / 'runtime/config/sam.json', sam)
    config['sam']['sha256'] = digest(output / 'runtime/config/sam.json')
    write(output / 'runtime/config/pipeline.json', config)
    frozen = dict(original)
    frozen['python_version'] = sys.version
    frozen['package_versions'] = versions
    frozen['bundle_files'] = {relative: digest(output / relative)
                             for relative in original['bundle_files'] if (output / relative).is_file()}
    # Replace the old machine-specific executable/site-package paths. Imported
    # source and binaries are recorded at their actual new locations.
    external = {str(Path(sys.executable).resolve()): digest(Path(sys.executable).resolve()),
                str(paths['codex']): digest(paths['codex']), str(spice): digest(spice)}
    ultra = Path(importlib.util.find_spec('ultralytics').origin).parent
    for old in original['external_source_files']:
        marker = '/site-packages/ultralytics/'
        if marker in old:
            path = ultra / old.split(marker, 1)[1]
            external[str(path)] = digest(path)
    frozen['external_source_files'] = external
    write(output / 'CONFIG_FREEZE.json', frozen)
    receipt = {'status': 'DERIVED_ENVIRONMENT_CONFIGURED_NOT_END_TO_END_VALIDATED',
               'source': str(source), 'output': str(output), 'source_files_before': before,
               'source_CONFIG_FREEZE_sha256': digest(source / 'CONFIG_FREEZE.json'),
               'derived_CONFIG_FREEZE_sha256': digest(output / 'CONFIG_FREEZE.json'),
               'device': args.device, 'cli': metadata,
               'changed_runtime_files': ['runtime/config/pipeline.json', 'runtime/config/sam.json', 'CONFIG_FREEZE.json'],
               'model_inference': False, 'model_download': False,
               'warning': 'NEW PATHS, INTERPRETER AND DEPENDENCY MANIFEST. NOT THE ORIGINAL MAC REHEARSAL ENVIRONMENT.'}
    write(output / 'provenance/DERIVED_ENVIRONMENT.json', receipt)
    after = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
             if p.is_file() and '.git' not in p.relative_to(source).parts}
    if before != after:
        raise RuntimeError('SOURCE CHANGED DURING CONFIGURATION; DO NOT USE THIS DERIVED RUNTIME')
    print('CONFIGURED A SEPARATE DERIVED RUNTIME; NO INFERENCE WAS RUN: ' + str(output))
    print('NEXT: ' + str(output / 'check-ready'))


if __name__ == '__main__':
    main()
