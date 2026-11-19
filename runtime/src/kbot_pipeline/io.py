"""Small local I/O surface. A failed stage is recorded, never retried or replaced."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def effective_model(config, role):
    if config[role].get('provider') == 'codex_cli':
        from .codex_backend import effective_model as cli_model
        return cli_model(config[role])
    if config[role].get('provider') == 'openai':
        from .openai_backend import selected_model
        return selected_model(config[role])
    return config[role]['model']


def api(endpoint, route, payload=None, *, timeout=15):
    url = urlparse(endpoint)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('This pipeline uses local Ollama only')
    request = urllib.request.Request(endpoint.rstrip('/') + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode(errors='replace')
        raise RuntimeError(f'Ollama HTTP {error.code}: {detail}') from error


def check_models(config):
    if any(config[role].get('provider', 'ollama') not in ('openai', 'ollama', 'codex_cli') for role in ('vision', 'reasoning')):
        raise ValueError('Unknown inference provider.')
    cli_roles = [role for role in ('vision', 'reasoning') if config[role].get('provider') == 'codex_cli']
    if cli_roles:
        from .codex_backend import cli_metadata
        cli_metadata(config[cli_roles[0]])
    cloud_roles = [role for role in ('vision', 'reasoning') if config[role].get('provider') == 'openai']
    if cloud_roles:
        from .openai_backend import api_key, selected_model
        api_key()  # Presence only; no inference, network request or credential log.
        for role in cloud_roles:
            if not selected_model(config[role]):
                raise ValueError('Configured cloud model is missing for ' + role)
    local_roles = [role for role in ('vision', 'reasoning') if role not in cloud_roles + cli_roles]
    if not local_roles:
        return
    tags = {m['name']: m for m in api(config['endpoint'], '/api/tags')['models']}
    for role, vision in [('vision', True), ('reasoning', False)]:
        if role not in local_roles:
            continue
        model = config[role]
        if tags.get(model['model'], {}).get('digest') != model['digest']:
            raise ValueError(f"Configured {role} checkpoint is absent or changed: {model['model']}")
        info = api(config['endpoint'], '/api/show', {'model': model['model']})
        required={'completion'}|({'vision'} if vision else set())|({'thinking'} if model.get('think') else set())
        if not required.issubset(info.get('capabilities', [])):
            raise ValueError(f'Wrong model capability for {role}')


def chat(config, role, messages, schema, output, *, retain_model=False):
    """Exactly one request. A caller may own a bounded reading-stage model session."""
    import jsonschema
    model = config[role]
    if model.get('provider') == 'codex_cli':
        from .codex_backend import chat as cli_chat
        return cli_chat(config, role, messages, schema, output)
    if model.get('provider') == 'openai':
        from .openai_backend import chat as openai_chat
        return openai_chat(config, role, messages, schema, output)
    if model.get('provider', 'ollama') != 'ollama':
        raise ValueError('Unknown inference provider for ' + role)
    payload = {'model': model['model'], 'messages': messages, 'format': schema,
        'stream': False, 'keep_alive': '5m' if retain_model else 0, 'options': model['options']}
    if 'think' in model:
        payload['think'] = model['think']
    save(Path(output) / f'{role}-request.json', payload)
    raw = None
    try:
        raw = api(config['endpoint'], '/api/chat', payload, timeout=model['timeout'])
        save(Path(output) / f'{role}-raw.json', raw)
        if not raw.get('done') or raw.get('done_reason') in ('length', 'max_tokens'):
            raise ValueError(f'{role} output was incomplete; no automatic retry')
        value = json.loads(raw['message']['content'])
        jsonschema.validate(value, schema)
        return value
    finally:
        # Cleanup does not generate tokens or act as an inference fallback.
        if not retain_model:
            api(config['endpoint'], '/api/generate', {'model': model['model'], 'keep_alive': 0}, timeout=30)
