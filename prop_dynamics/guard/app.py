"""Authenticated HTTP API for the project's message-level defense middleware."""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator

from .explain import explain_claim
from .fact_store import FactStore

logger = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class _RequestBodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Bound request bodies whether clients send Content-Length or stream them."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except ValueError:
                await self._reject(send, 400, "invalid Content-Length")
                return
            if declared_length < 0:
                await self._reject(send, 400, "invalid Content-Length")
                return
            if declared_length > self.max_bytes:
                await self._reject(send, 413, "request body is too large")
                return

        total_bytes = 0
        response_started = False

        async def limited_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive():
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > self.max_bytes:
                    raise _RequestBodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, limited_send)
        except _RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(send, 413, "request body is too large")

    @staticmethod
    async def _reject(send, status_code: int, detail: str):
        body = (f'{{"detail":"{detail}"}}').encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactsIn(_InputModel):
    facts: List[StrictStr] = Field(min_length=1, max_length=100)
    source: StrictStr = Field(default="manual", max_length=500)

    @field_validator("facts")
    @classmethod
    def validate_facts(cls, facts: List[str]) -> List[str]:
        cleaned = [" ".join(fact.split()) for fact in facts]
        if any(not fact for fact in cleaned):
            raise ValueError("facts must not contain empty strings")
        if any(len(fact) > 8000 for fact in cleaned):
            raise ValueError("each fact must be at most 8000 characters")
        return cleaned

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        source = " ".join(source.split())
        if not source:
            raise ValueError("source must not be empty")
        return source


class ClaimIn(_InputModel):
    claim: StrictStr = Field(min_length=1, max_length=2000)
    top_k: StrictInt = Field(default=3, ge=1, le=10)

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, claim: str) -> str:
        claim = claim.strip()
        if not claim:
            raise ValueError("claim must not be empty")
        return claim


class MessageIn(_InputModel):
    sender: StrictStr = Field(default="user", min_length=1, max_length=200)
    receiver: StrictStr = Field(default="agent", min_length=1, max_length=200)
    message: StrictStr = Field(min_length=1, max_length=8000)

    @field_validator("sender", "receiver", "message")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


def _default_graph():
    from prop_dynamics.defense.config import DefenseConfig
    from prop_dynamics.defense.genealogy_graph import GenealogyGraph

    config = DefenseConfig()
    graph = GenealogyGraph(similarity_threshold=config.similarity_threshold)
    graph.set_local_model(config.embedding_model)
    return graph


def _default_nli_classifier():
    from prop_dynamics.defense.config import DefenseConfig
    from prop_dynamics.defense.classifier_nli import NLIClassifier

    return NLIClassifier(model_name=DefenseConfig().nli_model)


def _default_guard_factory(graph):
    """Build the project's existing four-stage message middleware."""
    from prop_dynamics.defense.config import DefenseConfig, setup_openai
    from prop_dynamics.defense.middleware import DefenseMiddleware

    strategy = os.getenv("GUARD_STRATEGY", "balanced").strip().lower()
    if strategy not in {"speed", "balanced", "strict"}:
        raise ValueError("GUARD_STRATEGY must be speed, balanced, or strict")
    config = DefenseConfig(strategy=strategy)
    api_key = os.getenv("OPENAI_API_KEY")
    openai_module = setup_openai(config) if api_key else None
    guard = DefenseMiddleware(config, graph=graph, openai_module=openai_module)
    guard.set_hub_agents(
        [name.strip() for name in os.getenv("GUARD_HUB_AGENTS", "").split(",") if name.strip()]
    )
    guard.verifier.set_knowledge_base(graph)
    return guard


def create_app(
    db_path: str = "guard_facts.db",
    graph=None,
    nli_model=None,
    guard_factory: Optional[Callable[[Any], Any]] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Create the API without opening the database or loading ML models.

    Models and the fact store are initialized during application startup.
    ``nli_model`` accepts an object implementing the project's
    ``classify(claim, graph, top_k=...)`` interface, which permits offline tests
    and local alternatives without changing the API code.
    """
    configured_key = api_key if api_key is not None else os.getenv("GUARD_API_KEY")
    state = {"graph": graph, "store": None, "classifier": nli_model, "guard": None}
    graph_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        current_graph = state["graph"]
        if current_graph is None:
            current_graph = _default_graph()
            state["graph"] = current_graph
        state["store"] = FactStore(db_path, current_graph)
        try:
            if state["classifier"] is None:
                state["classifier"] = _default_nli_classifier()
            app.state.graph = current_graph
            app.state.store = state["store"]
            app.state.classifier = state["classifier"]
            yield
        finally:
            state["store"].close()
            state["store"] = None
            state["guard"] = None

    app = FastAPI(title="From Spark to Fire GUARD", version="1.0.0", lifespan=lifespan)
    app.state.graph = None
    app.state.store = None
    app.state.classifier = None

    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=_MAX_REQUEST_BYTES)

    def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
        if not isinstance(configured_key, str) or not configured_key.strip():
            raise HTTPException(status_code=503, detail="API authentication is not configured")
        parts = authorization.split() if authorization else []
        provided = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else ""
        if (
            configured_key != configured_key.strip()
            or len(configured_key) < 32
            or len(configured_key.encode("utf-8")) < 32
        ):
            raise HTTPException(status_code=503, detail="configured API key must be at least 32 characters")
        if not secrets.compare_digest(provided.encode("utf-8"), configured_key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health():
        store = state["store"]
        return {"status": "ok" if store is not None else "starting"}

    @app.post("/facts", dependencies=[Depends(require_api_key)])
    def add_facts(body: FactsIn):
        added, skipped = [], []
        with graph_lock:
            store = state["store"]
            for row in store.add_many(body.facts, source=body.source):
                (added if row["created"] else skipped).append(row)
            total = store.count()
        return {"added": added, "skipped_duplicates": skipped, "total": total}

    @app.get("/facts", dependencies=[Depends(require_api_key)])
    def list_facts(
        limit: int = Query(default=100, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        with graph_lock:
            store = state["store"]
            facts = store.list(limit=limit, offset=offset)
            total = store.count()
        return {"total": total, "limit": limit, "offset": offset, "facts": facts}

    @app.delete("/facts/{fact_id}", dependencies=[Depends(require_api_key)])
    def delete_fact(fact_id: int):
        if fact_id < 1:
            raise HTTPException(status_code=422, detail="fact_id must be positive")
        with graph_lock:
            store = state["store"]
            if not store.delete(fact_id):
                raise HTTPException(status_code=404, detail="fact not found")
            total = store.count()
        return {"deleted": fact_id, "total": total}

    @app.post("/explain", dependencies=[Depends(require_api_key)])
    def explain(body: ClaimIn):
        try:
            with graph_lock:
                return explain_claim(body.claim, state["graph"], state["classifier"], top_k=body.top_k)
        except Exception as exc:
            logger.error("GUARD claim explanation failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=503, detail="claim explanation unavailable")

    @app.post("/check", dependencies=[Depends(require_api_key)])
    def check(body: MessageIn):
        try:
            with graph_lock:
                if state["guard"] is None:
                    factory = guard_factory or _default_guard_factory
                    state["guard"] = factory(state["graph"])
                result = state["guard"].process_message(body.sender, body.receiver, body.message)
        except Exception as exc:
            logger.error("GUARD message check failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="message check failed")

        claims = []
        for atom in list(result.send_buffer) + list(result.rollback_queue):
            claims.append(
                {
                    "text": atom.content,
                    "color": atom.color.value,
                    "tag": atom.tag,
                    "reason": atom.reason,
                    "evidence": list(atom.evidence or []),
                    "source_fact_ids": list(atom.sources or []),
                }
            )
        return {
            "can_send": result.can_send,
            # Do not offer a partial message that a caller could accidentally
            # forward after GUARD has requested rollback.
            "final_message": result.final_message if result.can_send else None,
            "feedback": result.rollback_feedback,
            "claims": claims,
        }

    return app


__all__ = ["ClaimIn", "FactsIn", "MessageIn", "create_app"]
