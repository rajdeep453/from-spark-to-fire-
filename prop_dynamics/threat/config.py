"""Configuration"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ThreatConfig:
    """Experiment configuration."""
    framework: str = "langchain"
    
    n_tasks: int = 10
    n_repeats: int = 3
    
    max_rounds_crewai: int = 3      # Manager -> Workers -> Manager cycles
    max_rounds_langgraph: int = 3   # Supervisor <-> Worker cycles
    max_rounds_autogen: int = 3     # GroupChat rounds
    max_rounds_camel: int = 3       # Debate rounds
    
    injection_target: str = "entry"  # entry, hub, leaf
    intervention_round: Optional[int] = None
    
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini") # default model
    temperature: float = 0.3
    
    output_dir: str = "outputs/threat"
    
    log_debug_prompts: bool = False
    
    api_key: Optional[str] = None
    proxy: Optional[str] = None
    
    # Random seed
    seed: int = 42
    
    def get_output_path(self, filename: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)


def stable_seed(base_seed: int, task_id: str, repeat: int = 0) -> int:
    """Derive a repeatable 64-bit seed independent of Python hash randomization."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError("repeat must be a non-negative integer")
    material = f"{base_seed}\0{task_id}\0{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def setup_openai(config: ThreatConfig):
    from prop_dynamics.openai_compat import configure_openai
    return configure_openai(config)
