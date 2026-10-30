"""GUARD message screening and fact-store integration for Propagation Dynamics."""

from .explain import explain_claim, explain_result
from .fact_store import FactStore


def create_app(*args, **kwargs):
    """Create the optional HTTP app without importing FastAPI at package load."""
    from .app import create_app as app_factory

    return app_factory(*args, **kwargs)


__all__ = ["FactStore", "create_app", "explain_claim", "explain_result"]
