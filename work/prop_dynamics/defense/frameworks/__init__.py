

from .base import BaseDefenseFramework, Agent, Message, ExecutionLog, call_llm
try:
    from .hybrid_adapter import HybridFramework
except ModuleNotFoundError:
    HybridFramework = None
from .metagpt_defense import MetaGPTDefense
from .langgraph_defense import LangGraphDefense
from .autogen_defense import AutoGenDefense

FRAMEWORKS = {
    "metagpt": {"adapter": MetaGPTDefense, "topology": "chain"},
    "langgraph": {"adapter": LangGraphDefense, "topology": "star"},
    "autogen": {"adapter": AutoGenDefense, "topology": "mesh"},
}
if HybridFramework is not None:
    FRAMEWORKS["hybrid"] = {"adapter": HybridFramework, "topology": "hybrid"}

__all__ = [
    "BaseDefenseFramework", "Agent", "Message", "ExecutionLog", "call_llm",
    "HybridFramework", "MetaGPTDefense", "LangGraphDefense", "AutoGenDefense",
    "FRAMEWORKS",
]
