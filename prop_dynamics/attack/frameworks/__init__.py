"""Frameworks."""

from .base import BaseFramework, Agent, Message, ExecutionLog, MessageType, ArtifactType, Artifact
from .llm import call_llm_with_retry
from .langchain_adapter import LangChainAdapter
from .metagpt_adapter import MetaGPTAdapter
from .crewai_adapter import CrewAIAdapter
from .langgraph_adapter import LangGraphAdapter
from .autogen_adapter import AutoGenAdapter
from .camel_adapter import CamelAdapter

FRAMEWORKS = {
    "langchain": {"adapter": LangChainAdapter, "topology": "chain"},
    "metagpt": {"adapter": MetaGPTAdapter, "topology": "chain"},
    "crewai": {"adapter": CrewAIAdapter, "topology": "star"},
    "langgraph": {"adapter": LangGraphAdapter, "topology": "star"},
    "autogen": {"adapter": AutoGenAdapter, "topology": "mesh"},
    "camel": {"adapter": CamelAdapter, "topology": "mesh"},
}


def create_adapter(framework_name: str, config):
    if framework_name not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework_name}. Available: {list(FRAMEWORKS.keys())}")
    return FRAMEWORKS[framework_name]["adapter"](config)


def get_topology(framework_name: str) -> str:
    if framework_name not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework_name}")
    return FRAMEWORKS[framework_name]["topology"]


__all__ = [
    "BaseFramework",
    "Agent",
    "Message",
    "ExecutionLog",
    "MessageType",
    "ArtifactType",
    "Artifact",
    "call_llm_with_retry",
    "LangChainAdapter",
    "MetaGPTAdapter",
    "CrewAIAdapter",
    "LangGraphAdapter",
    "AutoGenAdapter",
    "CamelAdapter",
    "FRAMEWORKS",
    "create_adapter",
    "get_topology",
]
