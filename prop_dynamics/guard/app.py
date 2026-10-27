"""FastAPI app: persistent trusted facts + explain + full message check."""

from typing import Any, Callable, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .explain import explain_claim
from .fact_store import FactStore


class FactsIn(BaseModel):
    facts: List[str] = Field(min_length=1, max_length=100)


class ClaimIn(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(3, ge=1, le=10)


class MessageIn(BaseModel):
    sender: str = "user"
    receiver: str = "agent"
    message: str = Field(min_length=1, max_length=8000)


def _default_graph():
    from prop_dynamics.defense.config import DefenseConfig
    from prop_dynamics.defense.genealogy_graph import GenealogyGraph

    graph = GenealogyGraph(similarity_threshold=DefenseConfig().similarity_threshold)
    graph.set_local_model()
    return graph


def _default_nli_model():
    from prop_dynamics.defense.classifier_nli import NLIClassifier

    return NLIClassifier().model


def _default_guard_factory(graph):
    """Build the real DefenseMiddleware (needs an LLM API key)."""
    from prop_dynamics.defense.config import DefenseConfig, setup_openai
    from prop_dynamics.defense.middleware import DefenseMiddleware

    config = DefenseConfig(strategy="balanced")
    guard = DefenseMiddleware(config, graph=graph, openai_module=setup_openai(config))
    # An empty GenealogyGraph is falsy (it defines __len__), so the middleware's
    # "graph or GenealogyGraph(...)" may have swapped in its own graph. Force ours.
    guard.graph = graph
    guard.verifier.set_knowledge_base(graph)
    return guard


def create_app(
    db_path: str = "guard_facts.db",
    graph=None,
    nli_model=None,
    guard_factory: Optional[Callable[[Any], Any]] = None,
) -> FastAPI:
    graph = graph if graph is not None else _default_graph()
    nli_model = nli_model if nli_model is not None else _default_nli_model()
    store = FactStore(db_path, graph)
    factory = guard_factory or _default_guard_factory
    holder = {"guard": None}

    app = FastAPI(title="Message Guard", version="0.1")
    app.state.store = store
    app.state.graph = graph

    @app.get("/health")
    def health():
        return {"status": "ok", "trusted_facts": store.count()}

    @app.post("/facts")
    def add_facts(body: FactsIn):
        added, skipped = [], []
        for text in body.facts:
            try:
                row = store.add(text)
            except ValueError:
                raise HTTPException(status_code=422, detail="facts must not be empty")
            (added if row["created"] else skipped).append(row)
        return {"added": added, "skipped_duplicates": skipped, "total": store.count()}

    @app.get("/facts")
    def list_facts():
        facts = store.list()
        return {"total": len(facts), "facts": facts}

    @app.delete("/facts/{fact_id}")
    def delete_fact(fact_id: int):
        if not store.delete(fact_id):
            raise HTTPException(status_code=404, detail=f"fact {fact_id} not found")
        return {"deleted": fact_id, "total": store.count()}

    @app.post("/explain")
    def explain(body: ClaimIn):
        return explain_claim(body.claim, graph, nli_model, top_k=body.top_k)

    @app.post("/check")
    def check(body: MessageIn):
        try:
            if holder["guard"] is None:
                holder["guard"] = factory(graph)
            result = holder["guard"].process_message(body.sender, body.receiver, body.message)
        except Exception as exc:  # LLM/API problems should not crash the server
            raise HTTPException(status_code=502, detail=f"check failed: {exc}")
        claims = []
        for atom in list(result.send_buffer) + list(result.rollback_queue):
            claims.append(
                {
                    "text": atom.content,
                    "color": atom.color.value,
                    "tag": atom.tag,
                    "reason": atom.reason,
                    "evidence": list(atom.evidence or []),
                }
            )
        return {
            "can_send": result.can_send,
            "final_message": result.final_message,
            "feedback": result.rollback_feedback,
            "claims": claims,
        }

    return app
