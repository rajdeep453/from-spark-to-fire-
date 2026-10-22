

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import re
import os
import time


BLOCKED_BY_GUARDRAIL = "[BLOCKED_BY_GUARDRAIL]"

LIMITED_BY_UPSTREAM = {
    "Architect": "PRD input is unavailable because the upstream PM handoff was stopped by the guardrail. Architecture design cannot be completed.",
    "Engineer": "Required PRD or architecture input is unavailable because an upstream handoff was stopped by the guardrail. A reliable implementation cannot be completed.",
    "QA": "Implementation output is unavailable because an upstream handoff was stopped by the guardrail. QA review and final deliverables cannot be completed.",
    "Supervisor": "Worker inputs are unavailable because an upstream handoff was stopped by the guardrail. Reliable synthesis cannot be completed.",
    "Analyst": "Supervisor instruction is unavailable because the upstream handoff was stopped by the guardrail. Analysis cannot be completed.",
    "Solver": "Supervisor instruction is unavailable because the upstream handoff was stopped by the guardrail. Solution work cannot be completed.",
    "Validator": "Supervisor instruction is unavailable because the upstream handoff was stopped by the guardrail. Validation cannot be completed.",
    "Critic": "Prior discussion input is unavailable because an upstream handoff was stopped by the guardrail. Critique cannot be completed.",
    "Executor": "Prior discussion input is unavailable because an upstream handoff was stopped by the guardrail. Execution details cannot be completed.",
    "Summarizer": "Prior discussion input is unavailable because an upstream handoff was stopped by the guardrail. Final synthesis cannot be completed.",
}


def is_blocked_placeholder(content: str) -> bool:
    return (content or "").strip() == BLOCKED_BY_GUARDRAIL


def role_limited_output(agent_name: str) -> str:
    return LIMITED_BY_UPSTREAM.get(
        agent_name,
        "Required upstream input is unavailable because a handoff was stopped by the guardrail. This role cannot reliably continue.",
    )


def is_limited_output(content: str) -> bool:
    stripped = (content or "").strip()
    return (
        stripped in set(LIMITED_BY_UPSTREAM.values())
        or any(message in stripped for message in LIMITED_BY_UPSTREAM.values())
        or stripped.endswith("This role cannot reliably continue.")
    )


def handoff_has_unavailable_dependency(content: str) -> bool:
    lines = [
        line.strip()
        for line in (content or "").splitlines()
        if line.strip() and not (line.strip().startswith("===") and line.strip().endswith("==="))
    ]
    return any(is_blocked_placeholder(line) or is_limited_output(line) for line in lines)


def estimate_content_tokens(content: str) -> int:
    return len(re.findall(r"\w+|[^\s\w]", content or ""))


class MessageType(Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    RESULT = "result"
    INJECTION = "injection"


@dataclass
class Agent:
    name: str
    role: str
    system_prompt: str = ""
    is_hub: bool = False


@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    round: int
    msg_type: MessageType = MessageType.CONVERSATION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    blocked: bool = False
    tags: List[str] = field(default_factory=list)
    content_tokens_est: int = 0
    sender_completion_tokens: int = 0
    
    def to_dict(self) -> dict:
        return {
            "sender": self.sender, "receiver": self.receiver,
            "content": self.content[:5000], "round": self.round,
            "msg_type": self.msg_type.value, "blocked": self.blocked, "tags": self.tags,
            "content_tokens_est": self.content_tokens_est,
            "sender_completion_tokens": self.sender_completion_tokens,
        }


@dataclass
class ExecutionLog:
    task_id: str
    framework: str
    topology_type: str
    messages: List[Message] = field(default_factory=list)
    final_output: str = ""
    total_rounds: int = 0
    total_tokens: int = 0
    defense_stats: Dict[str, Any] = field(default_factory=dict)
    injection_config: Optional[Any] = None
    
    def add_message(self, msg: Message):
        self.messages.append(msg)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "framework": self.framework,
            "topology_type": self.topology_type,
            "messages": [m.to_dict() for m in self.messages],
            "final_output": self.final_output[:5000],
            "total_rounds": self.total_rounds, "defense_stats": self.defense_stats,
        }


REWRITE_PROMPT = """Your message had issues:

{feedback}

Original: {original}

Please rewrite addressing the issues."""


def call_llm(openai_module, model: str, messages: list, 
             temperature: float = 0.3, max_tokens: int = 1000,
             return_usage: bool = False):
    retries = int(os.getenv("FRAMEWORK_API_RETRIES", os.getenv("OPENAI_MAX_RETRIES", "3")))
    base_delay = float(os.getenv("API_RETRY_BASE_SECONDS", "2"))
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = openai_module.ChatCompletion.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            usage = resp.get("usage", {}) or {}
            tokens = usage.get("total_tokens", 0)
            content = resp["choices"][0]["message"]["content"].strip()
            if return_usage:
                return content, tokens, usage
            return content, tokens
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    print("[Framework] Request failed.")
    raise last_exc


class BaseDefenseFramework(ABC):
    def __init__(self):
        self.agents: List[Agent] = []
        self.topology_type: str = ""
        self.max_rounds: int = 5
        self.middleware = None
        self._openai = None
        self.model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_rewrite_retries: int = 1
        self._agent_map: Dict[str, Agent] = {}
        self.tokens_framework: int = 0
        self.agent_token_usage: Dict[str, Dict[str, int]] = {}
        self._last_agent_call: Dict[str, Any] = {}
    
    def set_middleware(self, middleware):
        self.middleware = middleware
        hub_agents = [a.name for a in self.agents if a.is_hub]
        middleware.set_hub_agents(hub_agents)
    
    def set_openai(self, openai_module):
        self._openai = openai_module
    
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
    
    def _call_agent(self, agent_name: str, context: str) -> str:
        if self._openai is None:
            return f"[{agent_name} placeholder response]"
        agent = self._agent_map.get(agent_name)
        if not agent:
            return f"[Unknown agent: {agent_name}]"
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": context}
        ]
        try:
            content, tokens, usage = call_llm(
                self._openai, self.model, messages, return_usage=True
            )
            self.tokens_framework += tokens
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            if completion_tokens == 0:
                completion_tokens = estimate_content_tokens(content)
            agent_usage = self.agent_token_usage.setdefault(
                agent_name,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            agent_usage["calls"] += 1
            agent_usage["prompt_tokens"] += prompt_tokens
            agent_usage["completion_tokens"] += completion_tokens
            agent_usage["total_tokens"] += int(tokens or 0)
            self._last_agent_call[agent_name] = {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": int(tokens or 0),
            }
            return content
        except Exception:
            if os.getenv("FRAMEWORK_FAIL_FAST", "0").lower() in {"1", "true", "yes"}:
                raise
            return "[Request unavailable]"
    
    def _call_rewrite(self, agent_name: str, original: str, feedback: str) -> str:
        if self._openai is None:
            return original
        agent = self._agent_map.get(agent_name)
        if not agent:
            return original
        messages = [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": REWRITE_PROMPT.format(feedback=feedback, original=original)}
        ]
        try:
            content, tokens = call_llm(self._openai, self.model, messages)
            self.tokens_framework += tokens
            return content
        except Exception:
            return original
    
    def _send_message(self, sender: str, receiver: str, message: str,
                      log: ExecutionLog, round_num: int) -> Tuple[str, List[str]]:
        def make_message(sender_name: str, receiver_name: str, content: str,
                         blocked: bool = False, tags: List[str] = None) -> Message:
            last = self._last_agent_call.get(sender_name, {})
            completion_tokens = 0
            if last.get("content") == content:
                completion_tokens = int(last.get("completion_tokens", 0) or 0)
            return Message(
                sender_name,
                receiver_name,
                content,
                round_num,
                blocked=blocked,
                tags=tags or [],
                content_tokens_est=estimate_content_tokens(content),
                sender_completion_tokens=completion_tokens,
            )

        if self.middleware is None:
            log.add_message(make_message(sender, receiver, message))
            return message, []

        if hasattr(self.middleware, "process_message_with_context"):
            decision = self.middleware.process_message_with_context(
                sender=sender,
                receiver=receiver,
                message=message,
                log=log,
                round_num=round_num,
                agents=self.agents,
                topology_type=self.topology_type,
            )
            final_message = decision.get("message", message)
            tags = decision.get("tags", [])
            blocked = bool(decision.get("blocked", False))
            log.add_message(make_message(sender, receiver, final_message, blocked=blocked, tags=tags))
            return final_message, tags

        def build_tags(send_buffer, rollback_queue):
            tags = []
            for atom in send_buffer:
                color = getattr(atom, "color", None)
                color_value = getattr(color, "value", None)
                if color_value == "green":
                    tags.append("PASS")
                elif atom.tag:
                    tags.append(atom.tag)
                else:
                    tags.append("YELLOW")
            for atom in rollback_queue:
                color = getattr(atom, "color", None)
                color_value = getattr(color, "value", None)
                if color_value == "red":
                    tags.append("RED")
            return tags
        
        current_msg = message
        for attempt in range(self.max_rewrite_retries + 1):
            result = self.middleware.process_message(sender, receiver, current_msg)
            
            if result.can_send:
                tags = build_tags(result.send_buffer, result.rollback_queue)
                log.add_message(make_message(sender, receiver, result.final_message, tags=tags))
                return result.final_message, tags
            else:
                if attempt < self.max_rewrite_retries:
                    current_msg = self._call_rewrite(sender, current_msg, result.rollback_feedback)
                else:
                    forced = self.middleware.force_finalize(result, sender)
                    tags = build_tags(forced.send_buffer, result.rollback_queue)
                    log.add_message(make_message(sender, receiver, forced.final_message, tags=tags))
                    return forced.final_message, tags
        
        log.add_message(make_message(sender, receiver, message))
        return message, []

    def _finalize_output(self, sender: str, message: str) -> str:
        if self.middleware is None:
            return message
        if hasattr(self.middleware, "process_message_with_context"):
            return message
        result = self.middleware.process_message(sender, "FINAL", message)
        if result.can_send:
            return result.final_message
        forced = self.middleware.force_finalize(result, sender)
        return forced.final_message
