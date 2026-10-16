import os
from typing import Protocol

import requests


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    URL = 'https://api.anthropic.com/v1/messages'

    def __init__(self, model, api_key=None, timeout=60, temperature=None, session=None):
        self.model, self.timeout, self.temperature = model, timeout, temperature
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise RuntimeError('Set ANTHROPIC_API_KEY (see .env.example)')
        self._http = session or requests

    def complete(self, system, user, max_tokens=1024):
        body = {'model': self.model, 'max_tokens': max_tokens, 'system': system,
                'messages': [{'role': 'user', 'content': user}]}
        if self.temperature is not None:
            body['temperature'] = self.temperature
        r = self._http.post(
            self.URL, json=body, timeout=self.timeout,
            headers={'x-api-key': self.api_key, 'anthropic-version': '2023-06-01',
                     'content-type': 'application/json'})
        r.raise_for_status()
        return ''.join(b.get('text', '') for b in r.json()['content'] if b.get('type') == 'text')
