# -*- coding: utf-8 -*-
"""
@description: Multi-provider LLM client for medical data generation.

Supports OpenAI, DeepSeek, Doubao (ByteDance), and MiniMax via OpenAI-compatible API.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from urllib import error, request

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


def load_env_file(path: Optional[str] = None) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend([
        Path.cwd() / '.env',
        Path(__file__).resolve().parents[1] / '.env',
    ])
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        for raw_line in candidate.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


class _HTTPChatCompletionResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        content = ''
        choices = payload.get('choices') or []
        if choices:
            message = choices[0].get('message') or {}
            raw_content = message.get('content', '')
            if isinstance(raw_content, list):
                content = ''.join(
                    item.get('text', '') if isinstance(item, dict) else str(item)
                    for item in raw_content
                )
            else:
                content = str(raw_content or '')
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    def model_dump(self) -> Dict[str, Any]:
        return self._payload


class _HTTPChatCompletions:
    def __init__(self, api_key: str, base_url: Optional[str]):
        self.api_key = api_key
        self.base_url = (base_url or 'https://api.openai.com/v1').rstrip('/')

    def create(self, model: str, messages, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512):
        endpoint = self.base_url
        if not endpoint.endswith('/chat/completions'):
            endpoint = endpoint + '/chat/completions'
        payload = json.dumps({
            'model': model,
            'messages': list(messages),
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
        }).encode('utf-8')
        req = request.Request(
            endpoint,
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                response_payload = json.loads(resp.read().decode('utf-8'))
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='ignore')
            raise RuntimeError(f'LLM request failed: {exc.code} {detail}') from exc
        except error.URLError as exc:
            raise RuntimeError(f'LLM request failed: {exc}') from exc
        return _HTTPChatCompletionResponse(response_payload)


class _HTTPChat:
    def __init__(self, api_key: str, base_url: Optional[str]):
        self.completions = _HTTPChatCompletions(api_key=api_key, base_url=base_url)


class HTTPCompatibleClient:
    def __init__(self, api_key: str, base_url: Optional[str]):
        self.chat = _HTTPChat(api_key=api_key, base_url=base_url)


# Provider configurations
PROVIDER_CONFIGS = {
    'openai': {
        'env_key': 'OPENAI_API_KEY',
        'base_url': None,
        'default_model': 'gpt-4o',
    },
    'deepseek': {
        'env_key': 'DEEPSEEK_API_KEY',
        'base_url': 'https://api.deepseek.com',
        'default_model': 'deepseek-reasoner',
    },
    'doubao': {
        'env_key': 'DOUBAO_API_KEY',
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
        'default_model': 'ep-20240623141021-r77gl',
    },
    'minimax': {
        'env_key': 'MINIMAX_API_KEY',
        'base_url': 'https://api.minimax.io/v1',
        'default_model': 'MiniMax-M2.7',
    },
}

_DETECTION_ORDER = ['openai', 'deepseek', 'minimax', 'doubao']

load_env_file()


def detect_provider():
    for provider in _DETECTION_ORDER:
        config = PROVIDER_CONFIGS[provider]
        if os.environ.get(config['env_key']):
            return provider
    return None


def create_llm_client(provider=None, api_key=None, base_url=None, model=None):
    if provider is None:
        provider = detect_provider()
        if provider is None:
            raise ValueError(
                'No LLM provider detected. Set one of: '
                + ', '.join(c['env_key'] for c in PROVIDER_CONFIGS.values())
            )

    provider = provider.lower()
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(PROVIDER_CONFIGS.keys())}"
        )

    config = PROVIDER_CONFIGS[provider]
    resolved_key = api_key or os.environ.get(config['env_key'])
    if not resolved_key:
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set {config['env_key']} environment variable or pass api_key parameter."
        )

    resolved_url = base_url or os.environ.get(f"{provider.upper()}_BASE_URL") or config['base_url']
    resolved_model = model or os.environ.get(f"{provider.upper()}_MODEL") or config['default_model']

    if OpenAI is not None:
        kwargs = {'api_key': resolved_key}
        if resolved_url:
            kwargs['base_url'] = resolved_url
        client = OpenAI(**kwargs)
    else:
        client = HTTPCompatibleClient(api_key=resolved_key, base_url=resolved_url)
    return client, resolved_model
