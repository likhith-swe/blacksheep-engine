import json
import os
import threading
from flask import request, has_request_context

_thread_local = threading.local()

_candidate_paths = [
    os.path.join(os.path.dirname(__file__), '..', '..', 'locales'),
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'locales'),
]
_locales_dir = None
for p in _candidate_paths:
    if os.path.exists(os.path.join(p, 'languages.json')):
        _locales_dir = os.path.abspath(p)
        break

if not _locales_dir:
    _locales_dir = os.path.abspath(_candidate_paths[0])
    os.makedirs(_locales_dir, exist_ok=True)

# Load language registry
_languages = {}
lang_json_path = os.path.join(_locales_dir, 'languages.json')
if os.path.exists(lang_json_path):
    with open(lang_json_path, 'r', encoding='utf-8') as f:
        _languages = json.load(f)

# Load translation files
_translations = {}
if os.path.exists(_locales_dir):
    for filename in os.listdir(_locales_dir):
        if filename.endswith('.json') and filename != 'languages.json':
            locale_name = filename[:-5]
            with open(os.path.join(_locales_dir, filename), 'r', encoding='utf-8') as f:
                _translations[locale_name] = json.load(f)


def set_locale(locale: str):
    """Set locale for current thread. Call at the start of background threads."""
    _thread_local.locale = locale


def get_locale() -> str:
    if has_request_context():
        raw = request.headers.get('Accept-Language', 'en')
        return raw if raw in _translations else 'en'
    return getattr(_thread_local, 'locale', 'en')


def t(key: str, **kwargs) -> str:
    locale = get_locale()
    messages = _translations.get(locale, _translations.get('en', {}))

    value = messages
    for part in key.split('.'):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value is None:
        value = _translations.get('en', {})
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

    if value is None:
        return key

    if kwargs:
        for k, v in kwargs.items():
            value = value.replace(f'{{{k}}}', str(v))

    return value


def get_language_instruction() -> str:
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get('en', {}))
    return lang_config.get('llmInstruction', 'Please respond in English.')
