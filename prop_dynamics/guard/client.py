"""Small fail-closed HTTP client for screening experiment handoffs with GUARD."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GuardClientError(RuntimeError):
    """Raised when an enabled GUARD service cannot safely screen a message."""


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    message: str
    audit: Dict[str, Any]


class GuardClient:
    """Call GUARD's authenticated message endpoint and fail closed on errors."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("GUARD base_url must be a non-empty string")
        if not isinstance(api_key, str) or len(api_key) < 32:
            raise ValueError("GUARD API key must contain at least 32 characters")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("GUARD timeout must be a positive number")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)

    @classmethod
    def from_env(cls) -> Optional["GuardClient"]:
        enabled = os.getenv("GUARD_ENABLED", "0").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        api_key = os.getenv("GUARD_API_KEY", "")
        if not api_key:
            raise GuardClientError("GUARD_ENABLED is set but GUARD_API_KEY is missing")
        try:
            timeout = float(os.getenv("GUARD_REQUEST_TIMEOUT", "300"))
        except ValueError as exc:
            raise GuardClientError("GUARD_REQUEST_TIMEOUT must be numeric") from exc
        try:
            return cls(
                os.getenv("GUARD_API_URL", "http://127.0.0.1:8000"),
                api_key,
                timeout,
            )
        except ValueError as exc:
            raise GuardClientError(str(exc)) from exc

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Do not put response bodies or credentials in experiment logs.
            raise GuardClientError(f"GUARD returned HTTP {exc.code} for {path}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GuardClientError(f"GUARD request failed for {path}: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardClientError(f"GUARD returned invalid JSON for {path}") from exc
        if not isinstance(value, dict):
            raise GuardClientError(f"GUARD returned an invalid response for {path}")
        return value

    def _request(self, method: str, path: str) -> Dict[str, Any]:
        request = Request(
            self.base_url + path,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise GuardClientError(f"GUARD returned HTTP {exc.code} for {path}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise GuardClientError(f"GUARD request failed for {path}: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GuardClientError(f"GUARD returned invalid JSON for {path}") from exc
        if not isinstance(value, dict):
            raise GuardClientError(f"GUARD returned an invalid response for {path}")
        return value

    def clear_facts(self) -> Dict[str, Any]:
        """Delete all facts to isolate task contexts between runs."""
        return self._request("DELETE", "/facts/all")

    def add_facts(self, facts: Iterable[str], source: str) -> Dict[str, Any]:
        clean_facts = [fact.strip() for fact in facts if isinstance(fact, str) and fact.strip()]
        if not clean_facts:
            return {"added": [], "skipped_duplicates": [], "total": None}
        return self._post("/facts", {"facts": clean_facts, "source": source})

    def screen(self, sender: str, receiver: str, message: str) -> GuardDecision:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        if not message.strip():
            return GuardDecision(True, message, {"enabled": True, "decision": "empty"})
        if len(message) > 8000:
            return GuardDecision(
                False,
                "",
                {
                    "enabled": True,
                    "decision": "block",
                    "can_send": False,
                    "reason": "message exceeds GUARD's 8000-character API limit",
                },
            )

        started = time.perf_counter()
        response = self._post(
            "/check",
            {"sender": sender, "receiver": receiver, "message": message},
        )
        allowed = response.get("can_send")
        if not isinstance(allowed, bool):
            raise GuardClientError("GUARD response is missing boolean can_send")
        claims = response.get("claims", [])
        if not isinstance(claims, list):
            raise GuardClientError("GUARD response claims must be a list")
        if allowed:
            final_message = response.get("final_message")
            if not isinstance(final_message, str):
                raise GuardClientError("GUARD allowed a message but returned no final_message")
        else:
            # A rejected response is never forwarded, even if the API payload
            # accidentally includes partial content.
            final_message = ""
        audit = {
            "enabled": True,
            "decision": "allow" if allowed else "block",
            "can_send": allowed,
            "claims": claims,
            "feedback": response.get("feedback"),
            "latency_seconds": round(time.perf_counter() - started, 3),
        }
        return GuardDecision(allowed, final_message, audit)


def trusted_task_facts(task: Any) -> list[str]:
    """Extract only benchmark-provided source/answer fields as trusted facts."""
    task_id = str(getattr(task, "id", "task"))
    name = str(getattr(task, "name", task_id))
    facts = []
    for field in ("data_source", "resource_url"):
        value = getattr(task, field, None)
        if isinstance(value, str) and value.strip():
            facts.append(f"Trusted source for {name}: {value.strip()}")
    truth = getattr(task, "ground_truth", None)
    if truth is not None and str(truth).strip():
        facts.append(f"Benchmark ground truth for {name}: {str(truth).strip()}")
    return facts


_last_prepared_task: Optional[str] = None


def prepare_task_facts(client: Optional[GuardClient], task: Any) -> None:
    """Seed the GUARD fact store with the current task's trusted facts.

    Clears facts from a previous task so that residual entries cannot
    contaminate the classifier.  Each experiment repeat therefore starts
    with only the current task's benchmark facts.
    """
    global _last_prepared_task
    if client is None:
        return
    task_id = str(getattr(task, "id", "task"))
    facts = trusted_task_facts(task)
    if _last_prepared_task is not None and _last_prepared_task != task_id:
        # Task changed — purge prior facts.
        try:
            client.clear_facts()
        except GuardClientError:
            pass
    _last_prepared_task = task_id
    if facts:
        client.add_facts(facts, source=f"benchmark:{task_id}")

