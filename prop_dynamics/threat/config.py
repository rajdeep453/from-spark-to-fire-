"""Configuration"""

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
    
    api_key: Optional[str] = 'your_api_key'
    proxy: Optional[str] = 'your_proxy_url'
    
    # Random seed
    seed: int = 42
    
    def get_output_path(self, filename: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)


def setup_openai(config: ThreatConfig):
    from prop_dynamics.openai_compat import configure_openai
    return configure_openai(config)
