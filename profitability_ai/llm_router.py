"""Роутер LLM с автоматическим переключением моделей при лимитах/ошибках.

Не требует сторонних пакетов (только стандартная библиотека). Состояние «отдыха»
моделей хранится в процессе, поэтому исчерпавшая лимит модель не дёргается повторно
на каждый вопрос, пока не пройдёт время ожидания.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import llm_config as cfg

ENV_FILE = Path(__file__).parent / '.env'


class AllModelsFailed(Exception):
    """Ни одна модель цепочки не смогла ответить."""

    def __init__(self, log: list):
        self.log = log
        super().__init__('; '.join(log) if log else 'нет доступных моделей (не заданы ключи)')


@dataclass
class LLMReply:
    text: str
    model_label: str           # «Claude Sonnet (claude-sonnet-5-5)»
    switched: list = field(default_factory=list)   # какие модели пропущены и почему


def _read_env_file() -> dict:
    out = {}
    try:
        for line in ENV_FILE.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def get_key(name: str) -> str:
    """Ключ: код (llm_config.API_KEYS) → переменная окружения → .env."""
    return (cfg.API_KEYS.get(name) or os.environ.get(name) or _read_env_file().get(name) or '').strip()


# --------------------------------------------------------------------------
# Вызовы провайдеров (HTTP)
# --------------------------------------------------------------------------
class _HTTPFail(Exception):
    def __init__(self, status: int | None, kind: str, retry_after: float | None = None, detail: str = ''):
        self.status, self.kind, self.retry_after, self.detail = status, kind, retry_after, detail
        super().__init__(f'{kind} ({status}) {detail}'.strip())


def _post(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json', **headers}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=cfg.REQUEST_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', 'ignore')[:300]
        except Exception:
            pass
        ra = None
        try:
            ra = float(e.headers.get('retry-after')) if e.headers.get('retry-after') else None
        except (TypeError, ValueError):
            ra = None
        low = body.lower()
        if e.code in (429, 402) or 'quota' in low or 'rate limit' in low or 'rate_limit' in low or 'resource_exhausted' in low:
            kind = 'limit'
        elif e.code in (401, 403, 404):
            kind = 'auth'
        elif e.code >= 500 or e.code in (408, 409, 529):
            kind = 'server'
        else:
            kind = 'bad_request'
        raise _HTTPFail(e.code, kind, ra, body) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise _HTTPFail(None, 'server', None, str(e)[:120]) from None
    except json.JSONDecodeError:
        raise _HTTPFail(None, 'server', None, 'невалидный JSON') from None


def _call_anthropic(m: dict, key: str, system: str, user: str, max_tokens: int) -> str:
    data = _post('https://api.anthropic.com/v1/messages',
                 {'x-api-key': key, 'anthropic-version': '2023-06-01'},
                 {'model': m['model'], 'max_tokens': max_tokens, 'system': system,
                  'messages': [{'role': 'user', 'content': user}]})
    return ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text').strip()


def _call_openai(m: dict, key: str, system: str, user: str, max_tokens: int) -> str:
    base = m.get('base_url', 'https://api.openai.com/v1').rstrip('/')
    data = _post(f'{base}/chat/completions', {'Authorization': f'Bearer {key}'},
                 {'model': m['model'], 'max_tokens': max_tokens, 'temperature': 0.2,
                  'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]})
    try:
        return (data['choices'][0]['message']['content'] or '').strip()
    except (KeyError, IndexError, TypeError):
        raise _HTTPFail(None, 'server', None, 'пустой ответ') from None


_CALLERS = {'anthropic': _call_anthropic, 'openai': _call_openai}


# --------------------------------------------------------------------------
# Роутер
# --------------------------------------------------------------------------
class LLMRouter:
    def __init__(self, chain: list | None = None):
        self.chain = list(chain if chain is not None else cfg.MODEL_CHAIN)
        self._cooldown: dict = {}      # индекс модели -> (до какого времени, причина)
        self._lock = threading.Lock()

    @staticmethod
    def label(m: dict) -> str:
        return f"{m['name']} ({m['model']})"

    def configured(self) -> list:
        """Модели цепочки, для которых задан ключ."""
        return [m for m in self.chain if get_key(m['key'])]

    def enabled(self) -> bool:
        return bool(cfg.USE_LLM and self.configured())

    def status(self) -> list:
        """Для интерфейса: [(название, состояние)] — без раскрытия ключей."""
        now, rows = time.time(), []
        for i, m in enumerate(self.chain):
            if not get_key(m['key']):
                rows.append((self.label(m), 'нет ключа'))
                continue
            until, why = self._cooldown.get(i, (0, ''))
            rows.append((self.label(m), f'пауза {int(until - now)} с ({why})' if until > now else 'готова'))
        return rows

    def _penalize(self, i: int, err: _HTTPFail):
        if err.kind == 'limit':
            secs, why = err.retry_after or cfg.COOLDOWN_RATE_LIMIT_S, 'лимит'
        elif err.kind == 'auth':
            secs, why = cfg.COOLDOWN_AUTH_S, 'ключ/модель недоступны'
        else:
            secs, why = cfg.COOLDOWN_SERVER_S, 'сбой сервиса'
        with self._lock:
            self._cooldown[i] = (time.time() + min(secs, 3600), why)

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMReply:
        """Пробует модели по порядку; при лимите/ошибке переходит к следующей."""
        max_tokens = max_tokens or cfg.MAX_TOKENS
        skipped, tried = [], 0
        for i, m in enumerate(self.chain):
            key = get_key(m['key'])
            if not key:
                continue
            until, why = self._cooldown.get(i, (0, ''))
            if until > time.time():
                skipped.append(f"{self.label(m)} — пауза ({why})")
                continue
            tried += 1
            try:
                text = _CALLERS[m['provider']](m, key, system, user, max_tokens)
                if not text:
                    raise _HTTPFail(None, 'server', None, 'пустой ответ')
                return LLMReply(text, self.label(m), skipped)
            except _HTTPFail as err:
                if err.kind == 'bad_request':
                    # ошибка запроса (не лимит): пробуем следующую, но модель на паузу не ставим надолго
                    skipped.append(f"{self.label(m)} — ошибка запроса ({err.status})")
                    continue
                self._penalize(i, err)
                reason = {'limit': 'достигнут лимит', 'auth': 'ключ/модель недоступны'}.get(err.kind, 'сбой сервиса')
                skipped.append(f"{self.label(m)} — {reason}")
        raise AllModelsFailed(skipped)


_ROUTER: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Общий роутер процесса (чтобы «паузы» моделей сохранялись между вопросами)."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LLMRouter()
    return _ROUTER
