"""Read-only artifact/credential readiness; never infer, install or download."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import socket
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'runtime/config/pipeline.json').read_text())
    cloud = any(config[role].get('provider') in ('openai', 'codex_cli') for role in ('vision', 'reasoning'))
    subscription = any(config[role].get('provider') == 'codex_cli' for role in ('vision', 'reasoning'))
    frozen = json.loads((root / 'CONFIG_FREEZE.json').read_text())
    for relative, expected in frozen['bundle_files'].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('Frozen demo file missing/changed: ' + relative)
    for package, expected in frozen['package_versions'].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError('Installed package changed: ' + package)
    for path, expected in frozen['external_source_files'].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise RuntimeError('External runtime file missing/changed: ' + path)
    if sys.version != frozen['python_version']:
        raise RuntimeError('Pinned Python version changed')
    sys.path.insert(0, str(root / 'runtime/src'))
    from kbot_pipeline.io import check_models, api
    from kbot_pipeline._sam_backend import validate_lock
    checkpoint = root / 'runtime' / config['detector']['weights']
    if digest(checkpoint) != config['detector']['sha256']:
        raise RuntimeError('Original detector checksum changed')
    check_models(config)  # Cloud: credential presence only; local: installed metadata.
    if not cloud and api(config['endpoint'], '/api/version')['version'] != frozen['ollama_server_version']:
        raise RuntimeError('Local Ollama server version changed')
    validate_lock(json.loads((root / 'runtime/config/sam.json').read_text()))
    external_denied = False
    if not cloud:
        with socket.socket() as probe:
            probe.settimeout(2)
            try:
                probe.connect(('1.1.1.1', 443))
            except PermissionError:
                external_denied = True
    if not cloud and not external_denied:
        raise RuntimeError('Expected external-network restriction is absent')
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d-%H%M%S-%f')
    directory = root / 'readiness-records'
    directory.mkdir(exist_ok=True)
    record = {'checked_hong_kong': stamp, 'ready': True,
              'frozen_files_and_versions_match': True,
              'local_detector_checkpoint_matches': True,
              'local_language_model_digests_match': not cloud,
              'openai_credentials_present': cloud and not subscription,
              'managed_chatgpt_cli_login_verified': subscription,
              'openai_model_access_verified': False if cloud else None,
              'SAM_source_and_checkpoint_hashes_match': True,
              'external_socket_denied': external_denied, 'model_inference': False,
              'inference_network': 'Official Codex CLI manages network and ChatGPT authentication.' if subscription else 'Official OpenAI HTTPS enabled; requests restricted by transport allowlist.' if cloud else 'Local Ollama only.',
              'installation_or_download': False}
    path = directory / (stamp + '.json')
    path.write_text(json.dumps(record, indent=2) + '\n')
    print('READY: frozen local detector/SAM artifacts and managed ChatGPT CLI login verified; inference was not tested.' if subscription else 'READY: frozen local detector/SAM artifacts and private OpenAI credential presence verified; cloud access is not tested.' if cloud else 'READY: frozen files, local models and offline restrictions verified.')
    print('No model inference, installation or downloads. Record: ' + str(path))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('NOT READY: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
