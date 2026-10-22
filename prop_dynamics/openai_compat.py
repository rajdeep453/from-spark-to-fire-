

import os
import threading
from typing import Optional


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_CHAT_SEMAPHORE = None
_CHAT_SEMAPHORE_LIMIT = None


def _get_chat_semaphore():
    global _CHAT_SEMAPHORE, _CHAT_SEMAPHORE_LIMIT
    raw_limit = os.getenv("OPENAI_CONCURRENCY_LIMIT", "").strip()
    if not raw_limit:
        return None
    limit = max(1, int(raw_limit))
    if _CHAT_SEMAPHORE is None or _CHAT_SEMAPHORE_LIMIT != limit:
        _CHAT_SEMAPHORE = threading.BoundedSemaphore(limit)
        _CHAT_SEMAPHORE_LIMIT = limit
    return _CHAT_SEMAPHORE


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value or value.startswith("your_"):
        return None
    return value


def _normalize_proxy(proxy: Optional[str]) -> Optional[str]:
    proxy = _clean_optional(proxy)
    if not proxy:
        return None
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def configure_openai(config=None):
    
    api_key = _clean_optional(getattr(config, "api_key", None)) or os.getenv("OPENAI_API_KEY")
    base_url = (
        _clean_optional(getattr(config, "base_url", None))
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    proxy = (
        _normalize_proxy(getattr(config, "proxy", None))
        or _normalize_proxy(os.getenv("OPENAI_PROXY"))
    )

    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    import openai

    if api_key:
        openai.api_key = api_key
    if base_url:
        if hasattr(openai, "base_url"):
            openai.base_url = base_url
        if hasattr(openai, "api_base"):
            openai.api_base = base_url

    if hasattr(openai, "OpenAI"):
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client_kwargs["max_retries"] = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
        client_kwargs["timeout"] = float(os.getenv("OPENAI_TIMEOUT", "30"))
        client = openai.OpenAI(**client_kwargs)

        class _ChatCompletion:
            @staticmethod
            def create(**kwargs):
                semaphore = _get_chat_semaphore()
                if semaphore is None:
                    return client.chat.completions.create(**kwargs).model_dump()
                with semaphore:
                    return client.chat.completions.create(**kwargs).model_dump()

        class _Embedding:
            @staticmethod
            def create(**kwargs):
                return client.embeddings.create(**kwargs).model_dump()

        openai.ChatCompletion = _ChatCompletion
        openai.Embedding = _Embedding

    return openai
