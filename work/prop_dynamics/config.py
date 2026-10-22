"""
unified configuration module
"""

import os
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class ExperimentConfig:
    """experiment configuration"""
    topology: str = "decentralized"  # centralized, layered, decentralized
    N: int = 5  # Network size
    
    num_seeds: int = 1  # Number of initial infected nodes
    edge_prob: float = 1.0  # Edge probability
    
    horizontal_prob: float = 0.3  # Horizontal edge probability
    skip_prob: float = 0.0  # Skip edge probability
    reciprocity: float = 0.5  # Fraction of bidirectional edges
    
    T_rounds: int = 5  # Rounds per experiment
    R_repeats: int = 10  # Number of repeated trials
    
    infection_function: str = "prod"  # prod or pois
    
    use_uniform_persona: bool = True  # True or False
    
    early_stop_on_full_infection: bool = False # True or False
    skip_infected_agents: bool = False # True or False
    
    gamma: float = 0.0 # not relative to this experiment
    checker_node: Optional[int] = None # not relative to this experiment
    
    edge_weight_mode: str = "uniform"
    
    seed: int = 42
    
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    output_dir: str = "outputs"
    log_file: str = "experiment_log.jsonl"
    
    api_key: Optional[str] = 'your_api_key'
    # proxy: Optional[str] = 'your_proxy_url'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.topology in ["centralized", "layered", "layered_horizontal", "decentralized"]
        assert self.N >= 2
        assert self.T_rounds >= 1
        assert self.R_repeats >= 1
        assert 0.0 <= self.gamma <= 1.0
        assert self.infection_function in ["prod", "pois"]
        assert self.edge_weight_mode in ["uniform", "normalized"]
        assert 1 <= self.num_seeds <= self.N
        assert 0.0 < self.edge_prob <= 1.0
    
    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "topology": self.topology,
            "N": self.N,
            "num_seeds": self.num_seeds,
            "edge_prob": self.edge_prob,
            "T_rounds": self.T_rounds,
            "R_repeats": self.R_repeats,
            "infection_function": self.infection_function,
            "use_uniform_persona": self.use_uniform_persona,
            "early_stop_on_full_infection": self.early_stop_on_full_infection,
            "skip_infected_agents": self.skip_infected_agents,
            "gamma": self.gamma,
            "checker_node": self.checker_node,
            "edge_weight_mode": self.edge_weight_mode,
            "seed": self.seed,
            "model": self.model,
            "output_dir": self.output_dir,
            "log_file": self.log_file,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Create config from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
    
    def save_json(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_log_path(self) -> str:
        """Get full log path."""
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, self.log_file)


V1_TOPOLOGY_CONFIGS = [
    ExperimentConfig(topology="centralized", N=5, T_rounds=5, R_repeats=20),
    ExperimentConfig(topology="layered", N=5, T_rounds=5, R_repeats=20),
    ExperimentConfig(topology="decentralized", N=5, T_rounds=5, R_repeats=20),
]

# V2_SCALE_CONFIGS = [
#     ExperimentConfig(topology="decentralized", N=3, T_rounds=5, R_repeats=20),
#     ExperimentConfig(topology="decentralized", N=5, T_rounds=5, R_repeats=20),
#     ExperimentConfig(topology="decentralized", N=10, T_rounds=5, R_repeats=20),
# ]

# V3_GAMMA_CONFIGS = [
#     ExperimentConfig(topology="layered", N=3, T_rounds=5, R_repeats=20, 
#                      use_uniform_persona=False, gamma=g, checker_node=2)
#     for g in [0.0, 0.25, 0.5, 0.75, 1.0]
# ]
