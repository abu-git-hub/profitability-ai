"""Конфигурация LLM: ключи и цепочка моделей — всё задаётся здесь, в коде.

Как это работает
----------------
* ``MODEL_CHAIN`` — список моделей по приоритету. Сначала используется первая;
  если у неё закончился лимит/квота (429, 402, 5xx, таймаут, неверный ключ и т. п.),
  запрос автоматически уходит на следующую.
* Ключ каждого провайдера берётся в таком порядке:
    1) ``API_KEYS`` ниже (прямо в коде);
    2) переменная окружения (``ANTHROPIC_API_KEY`` …);
    3) файл ``.env`` рядом с приложением (строки ``ИМЯ=значение``).
* Модель без ключа просто пропускается. Если ключей нет вообще — аналитик работает
  в локальном режиме (формулы + шаблоны), приложение остаётся полностью рабочим.

ВАЖНО: если вписываете ключи прямо в этот файл — не публикуйте его в открытом
репозитории (GitHub и т. п.). Для публичного репозитория используйте ``.env``
(он уже в .gitignore) или переменные окружения.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# 1. КЛЮЧИ (вписать сюда). Пустая строка = взять из окружения / .env
# --------------------------------------------------------------------------
API_KEYS = {
    'ANTHROPIC_API_KEY': '',   # https://console.anthropic.com
    'GEMINI_API_KEY': '',      # https://aistudio.google.com/apikey
    'GROQ_API_KEY': '',        # https://console.groq.com/keys
    'OPENAI_API_KEY': '',      # https://platform.openai.com/api-keys
    'OPENROUTER_API_KEY': '',  # https://openrouter.ai/keys
}

# --------------------------------------------------------------------------
# 2. ЦЕПОЧКА МОДЕЛЕЙ по приоритету (меняйте порядок / названия моделей как нужно)
#    provider: 'anthropic'  — Anthropic Messages API
#              'openai'     — любой OpenAI-совместимый API (OpenAI, Gemini, Groq, OpenRouter …)
# --------------------------------------------------------------------------
MODEL_CHAIN = [
    dict(name='Claude Sonnet', provider='anthropic', model='claude-sonnet-5-5',
         key='ANTHROPIC_API_KEY'),
    dict(name='Claude Haiku', provider='anthropic', model='claude-haiku-5-5',
         key='ANTHROPIC_API_KEY'),
    dict(name='Gemini Flash', provider='openai', model='gemini-2.5-flash',
         base_url='https://generativelanguage.googleapis.com/v1beta/openai',
         key='GEMINI_API_KEY'),
    dict(name='Groq Llama', provider='openai', model='llama-3.3-70b-versatile',
         base_url='https://api.groq.com/openai/v1',
         key='GROQ_API_KEY'),
    dict(name='OpenAI', provider='openai', model='gpt-4o-mini',
         base_url='https://api.openai.com/v1',
         key='OPENAI_API_KEY'),
    dict(name='OpenRouter', provider='openai', model='meta-llama/llama-3.3-70b-instruct:free',
         base_url='https://openrouter.ai/api/v1',
         key='OPENROUTER_API_KEY'),
]

# --------------------------------------------------------------------------
# 3. Параметры
# --------------------------------------------------------------------------
USE_LLM = True               # False — принудительно только локальный режим
REQUEST_TIMEOUT_S = 40       # таймаут одного запроса к модели
MAX_TOKENS = 1500
COOLDOWN_RATE_LIMIT_S = 60   # «отдых» модели после лимита (если API не вернул Retry-After)
COOLDOWN_SERVER_S = 30       # после 5xx / таймаута / сетевой ошибки
COOLDOWN_AUTH_S = 3600       # после неверного ключа / недоступной модели
