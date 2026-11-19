"""One-shot OpenAI Responses transport; original circuit schemas remain authoritative."""
from __future__ import annotations

import base64
import copy
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

ENDPOINT = 'https://api.openai.com/v1/responses'
KEY_FILE = Path.home() / '.config/kbot/openai.env'


def api_key():
    """Read a runtime secret only; never include it in configuration/artifacts."""
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key and KEY_FILE.is_file():
        if KEY_FILE.stat().st_mode & 0o077:
            raise ValueError('Private OpenAI credentials file must have mode 600.')
        for line in KEY_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith('export '):
                line = line[7:].strip()
            if line.startswith('OPENAI_API_KEY='):
                key = line.split('=', 1)[1].strip()
                if len(key) >= 2 and key[0] == key[-1] and key[0] in ('"', "'"):
                    key = key[1:-1]
                break
    if not key or any(c.isspace() for c in key):
        raise ValueError('OPENAI_API_KEY is missing or invalid. Configure it privately; no automatic fallback.')
    return key


def selected_model(settings):
    return os.environ.get('OPENAI_MODEL', '').strip() or settings['model']


def transport_schema(value):
    """Map exclusive branches/constants for transport; validate original on return."""
    if isinstance(value, list):
        return [transport_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {('anyOf' if key == 'oneOf' else key): transport_schema(item)
              for key, item in value.items() if key != 'const'}
    if 'const' in value:
        result['enum'] = [copy.deepcopy(value['const'])]
    if 'enum' in result and 'type' not in result:
        kinds = []
        for item in result['enum']:
            kind = 'null' if item is None else 'boolean' if isinstance(item, bool) else 'string' if isinstance(item, str) else 'integer' if isinstance(item, int) else 'number' if isinstance(item, float) else None
            if kind and kind not in kinds:
                kinds.append(kind)
        if kinds:
            result['type'] = kinds[0] if len(kinds) == 1 else kinds
    return result


def request_payload(settings, role, messages, schema):
    inputs = []
    for message in messages:
        if message.get('role') not in ('system', 'developer', 'user'):
            raise ValueError('Unsupported input message role.')
        if not isinstance(message.get('content'), str):
            raise ValueError('OpenAI input text must be a string.')
        content = [{'type': 'input_text', 'text': message['content']}]
        for encoded in message.get('images', []):
            image = base64.b64decode(encoded, validate=True)
            if image.startswith(b'\x89PNG\r\n\x1a\n'):
                mime = 'image/png'
            elif image.startswith(b'\xff\xd8\xff'):
                mime = 'image/jpeg'
            elif image.startswith(b'RIFF') and image[8:12] == b'WEBP':
                mime = 'image/webp'
            else:
                raise ValueError('Unsupported OpenAI image input.')
            content.append({'type': 'input_image', 'image_url': 'data:' + mime + ';base64,' + encoded, 'detail': 'high'})
        inputs.append({'role': message['role'], 'content': content})
    payload = {'model': selected_model(settings), 'store': False, 'input': inputs,
               'max_output_tokens': settings.get('max_output_tokens', 8192),
               'text': {'format': {'type': 'json_schema', 'name': 'circuit_' + role,
                                   'strict': True, 'schema': transport_schema(schema)}}}
    if settings.get('reasoning_effort'):
        payload['reasoning'] = {'effort': settings['reasoning_effort']}
    if 'temperature' in settings:
        payload['temperature'] = settings['temperature']
    return payload


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValueError('OpenAI redirect refused; no alternate endpoint is allowed.')


def chat(config, role, messages, schema, output):
    import jsonschema
    from .io import save
    settings = config[role]
    endpoint = settings.get('endpoint', ENDPOINT)
    parsed = urlparse(endpoint)
    if endpoint != ENDPOINT or parsed.scheme != 'https' or parsed.hostname != 'api.openai.com':
        raise ValueError('Only the official OpenAI Responses HTTPS endpoint is allowed.')
    payload = request_payload(settings, role, messages, schema)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / f'{role}-request.json', payload)
    save(output / f'{role}-provider.json', {'provider': 'openai', 'endpoint': endpoint,
                                          'requested_model': payload['model'], 'store': False})
    secret = api_key()
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=settings.get('timeout', 240)) as response:
            raw = json.load(response)
    except urllib.error.HTTPError as error:
        # Provider error messages can echo a rejected credential fragment.
        detail = re.sub(r'sk-[A-Za-z0-9_\-*]+', '[REDACTED]', error.read(16384).decode(errors='replace').replace(secret, '[REDACTED]'))
        save(output / f'{role}-raw.json', {'provider': 'openai', 'http_status': error.code,
                                          'error_body_redacted': detail})
        raise RuntimeError(f'OpenAI HTTP {error.code}; request failed without retry or fallback.') from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError('OpenAI request failed; no automatic retry or fallback.') from None
    save(output / f'{role}-raw.json', raw)
    save(output / f'{role}-provider.json', {'provider': 'openai', 'endpoint': endpoint,
        'requested_model': payload['model'], 'returned_model': raw.get('model'),
        'response_id': raw.get('id'), 'status': raw.get('status'), 'usage': raw.get('usage'), 'store': False})
    if raw.get('status') != 'completed':
        raise ValueError(f'{role} OpenAI output was incomplete or failed; no automatic retry.')
    text = []
    for item in raw.get('output', []):
        for part in item.get('content', []):
            if part.get('type') == 'refusal':
                raise ValueError(f'{role} OpenAI request was refused; no automatic retry.')
            if part.get('type') == 'output_text':
                text.append(part.get('text', ''))
    if len(text) != 1 or not text[0].strip():
        raise ValueError(f'{role} OpenAI returned no single structured output; no automatic retry.')
    value = json.loads(text[0])
    jsonschema.validate(value, schema)
    return value
