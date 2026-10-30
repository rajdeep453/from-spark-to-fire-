

import os
from dataclasses import dataclass, field
from typing import Optional
from prop_dynamics.openai_compat import DEFAULT_MODEL, configure_openai

@dataclass
class DefenseConfig:
    strategy: str = "balanced"  
    classifier_type: str = "nli"  
    nli_model: str = field(
        default_factory=lambda: os.getenv("GUARD_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("GUARD_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    use_local_embedding: bool = True
    max_retries: int = 2
    
    model: str = DEFAULT_MODEL
    classifier_model: str = DEFAULT_MODEL
    decomposer_model: str = DEFAULT_MODEL
    verifier_model: str = DEFAULT_MODEL
    search_top_k: int = 2
    similarity_threshold: float = 0.2
    top_k_retrieve: int = 2
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    proxy: Optional[str] = None
    reviewer_rounds: int = 1
    
    enable_decomposer: bool = True
    enable_decomposer_cache: bool = True
    enable_classifier: bool = True
    enable_verifier: bool = True
    enable_graph: bool = True
    init_facts: bool = True
    suppress_tags: bool = False

    enable_cache: bool = True
    fail_fast: bool = False

    def __post_init__(self):
        current_model = os.getenv("OPENAI_MODEL")
        if current_model:
            if self.model == DEFAULT_MODEL:
                self.model = current_model
            if self.classifier_model == DEFAULT_MODEL:
                self.classifier_model = current_model
            if self.decomposer_model == DEFAULT_MODEL:
                self.decomposer_model = current_model
            if self.verifier_model == DEFAULT_MODEL:
                self.verifier_model = current_model
        self.fail_fast = os.getenv("GUARD_FAIL_FAST", "0").lower() in {"1", "true", "yes"}


def setup_openai(config: DefenseConfig):
    return configure_openai(config)
