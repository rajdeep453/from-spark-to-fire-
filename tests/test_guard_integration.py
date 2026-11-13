import json
import sys
import types
from types import SimpleNamespace

import pytest

from prop_dynamics.guard.client import GuardClientError
from prop_dynamics.openai_compat import configure_openai, guard_agent_output


class _FakeCompletion:
    def model_dump(self):
        return {
            "model": "test-model",
            "choices": [{"message": {"content": "model generated text"}}],
            "usage": {"total_tokens": 3},
        }


class _FakeGuard:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.calls = []

    def screen(self, sender, receiver, message):
        self.calls.append((sender, receiver, message))
        return SimpleNamespace(
            allowed=self.allowed,
            message=message if self.allowed else "",
            audit={"enabled": True, "decision": "allow" if self.allowed else "block"},
        )


def _configure_with_fake_guard(monkeypatch, tmp_path, fake_guard):
    class _Completions:
        def create(self, **kwargs):
            return _FakeCompletion()

    class _OpenAIClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())
            self.embeddings = SimpleNamespace()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _OpenAIClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("GUARD_ENABLED", "1")
    monkeypatch.setenv("GUARD_API_KEY", "x" * 40)
    monkeypatch.setenv("GUARD_AUDIT_PATH", str(tmp_path / "guard-audit.jsonl"))
    monkeypatch.setattr("prop_dynamics.guard.client.GuardClient.from_env", lambda: fake_guard)
    return configure_openai(SimpleNamespace(api_key="test", base_url=None, proxy=None))


def test_openai_compat_screens_and_audits_allowed_output(monkeypatch, tmp_path):
    fake_guard = _FakeGuard(allowed=True)
    openai = _configure_with_fake_guard(monkeypatch, tmp_path, fake_guard)

    with guard_agent_output():
        result = openai.ChatCompletion.create(model="test-model", messages=[])

    assert result["choices"][0]["message"]["content"] == "model generated text"
    assert result["choices"][0]["guard"]["decision"] == "allow"
    assert fake_guard.calls == [("model_output", "next_agent", "model generated text")]
    audit = json.loads((tmp_path / "guard-audit.jsonl").read_text(encoding="utf-8"))
    assert audit["decision"]["decision"] == "allow"
    assert audit["output_sha256"]


def test_openai_compat_replaces_blocked_output_before_return(monkeypatch, tmp_path):
    fake_guard = _FakeGuard(allowed=False)
    openai = _configure_with_fake_guard(monkeypatch, tmp_path, fake_guard)

    with guard_agent_output():
        result = openai.ChatCompletion.create(model="test-model", messages=[])

    assert result["choices"][0]["message"]["content"] == "[BLOCKED_BY_GUARDRAIL]"
    assert result["choices"][0]["guard"]["decision"] == "block"


def test_openai_compat_does_not_forward_when_guard_fails(monkeypatch, tmp_path):
    class _UnavailableGuard:
        def screen(self, sender, receiver, message):
            raise GuardClientError("service unavailable")

    openai = _configure_with_fake_guard(monkeypatch, tmp_path, _UnavailableGuard())

    with pytest.raises(GuardClientError):
        with guard_agent_output():
            openai.ChatCompletion.create(model="test-model", messages=[])

    assert not (tmp_path / "guard-audit.jsonl").exists()
