"""Base framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class MessageType(Enum):
    CONVERSATION = "conversation"
    ARTIFACT = "artifact"
    INJECTION = "injection"


class ArtifactType(Enum):
    PRD = "prd"
    DESIGN = "design"
    CODE = "code"
    REVIEW = "review"
    ANSWER = "answer"


@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str = ""
    is_hub: bool = False


@dataclass
class Message:
    sender: str
    content: str
    round: int
    msg_type: MessageType = MessageType.CONVERSATION
    artifact_type: Optional[ArtifactType] = None
    receivers: List[str] = field(default_factory=list)
    tokens: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Artifact:
    artifact_type: ArtifactType
    content: str
    publisher: str
    subscribers: List[str] = field(default_factory=list)


@dataclass
class ExecutionLog:
    task_id: str
    framework: str
    topology_type: str
    messages: List[Message] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    final_output: str = ""
    total_rounds: int = 0
    total_tokens: int = 0
    injection_config: Optional[object] = None
    debug: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> dict:
        data = {
            "task_id": self.task_id,
            "framework": self.framework,
            "topology_type": self.topology_type,
            "messages": [
                {"sender": m.sender, "content": m.content[:10000], "round": m.round, 
                 "msg_type": m.msg_type.value}
                for m in self.messages
            ],
            "final_output": self.final_output,
            "total_rounds": self.total_rounds,
            "total_tokens": self.total_tokens,
        }
        if self.injection_config:
            data["infection_marker"] = self.injection_config.infection_marker
            data["strategy"] = self.injection_config.strategy
        if self.debug:
            data["debug"] = self.debug
        return data


class BaseFramework(ABC):   
    def __init__(self):
        self.agents: List[Agent] = []
        self.topology_type: str = ""
        self.max_rounds: int = 1
    
    @abstractmethod
    def get_agents(self) -> List[Agent]:
        pass
    
    @abstractmethod
    def run_task(self, task, injection_config=None) -> ExecutionLog:
        pass
    
    def get_entry_agent(self) -> str:
        return self.agents[0].name if self.agents else ""
    
    def get_hub_agent(self) -> Optional[str]:
        for agent in self.agents:
            if agent.is_hub:
                return agent.name
        return None
    
    def get_leaf_agent(self) -> Optional[str]:
        for agent in reversed(self.agents):
            if not agent.is_hub:
                return agent.name
        return None
