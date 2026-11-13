

import os
import threading
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_CHAT_SEMAPHORE = None
_CHAT_SEMAPHORE_LIMIT = None
_GUARD_AUDIT_LOCK = threading.Lock()
_GUARD_BLOCKED_PLACEHOLDER = "[BLOCKED_BY_GUARDRAIL]"
_SCREEN_AGENT_OUTPUT = ContextVar("screen_agent_output", default=False)


@contextmanager
def guard_agent_output():
    """Mark a model call as an agent response that must pass through GUARD."""
    token = _SCREEN_AGENT_OUTPUT.set(True)
    try:
        yield
    finally:
        _SCREEN_AGENT_OUTPUT.reset(token)


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
        guard_client = None
        if os.getenv("GUARD_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}:
            from prop_dynamics.guard.client import GuardClient

            guard_client = GuardClient.from_env()

        def screen_completion(result):
            if guard_client is None or not _SCREEN_AGENT_OUTPUT.get():
                return result

            choices = result.get("choices")
            if not isinstance(choices, list):
                raise RuntimeError("GUARD cannot screen a completion without choices")
            for choice in choices:
                message = choice.get("message") if isinstance(choice, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    decision = None
                    forwarded = _GUARD_BLOCKED_PLACEHOLDER
                    audit = {
                        "enabled": True,
                        "decision": "block",
                        "can_send": False,
                        "reason": "completion content was not text",
                    }
                else:
                    decision = guard_client.screen("model_output", "next_agent", content)
                    forwarded = decision.message if decision.allowed else _GUARD_BLOCKED_PLACEHOLDER
                    audit = decision.audit

                if isinstance(message, dict):
                    message["content"] = forwarded
                _write_guard_audit(result.get("model"), content, audit)
                if isinstance(choice, dict):
                    choice["guard"] = audit
            return result

        def _write_guard_audit(model_name, content, audit):
            path = os.getenv("GUARD_AUDIT_PATH", "outputs/guard_audit.jsonl")
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model_name,
                "output_sha256": hashlib.sha256(
                    (content if isinstance(content, str) else "").encode("utf-8")
                ).hexdigest(),
                "decision": audit,
            }
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with _GUARD_AUDIT_LOCK:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        class _ChatCompletion:
            @staticmethod
            def create(**kwargs):
                semaphore = _get_chat_semaphore()
                if semaphore is None:
                    result = client.chat.completions.create(**kwargs).model_dump()
                else:
                    with semaphore:
                        result = client.chat.completions.create(**kwargs).model_dump()
                return screen_completion(result)

        class _Embedding:
            @staticmethod
            def create(**kwargs):
                return client.embeddings.create(**kwargs).model_dump()

        openai.ChatCompletion = _ChatCompletion
        openai.Embedding = _Embedding

    return openai
