"""Strategy Matcher."""

import os
from typing import Dict

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

DEFAULT_STRATEGY_MAP = {
    "rigid": "compliance",
    "quant": "security_fud",
    "mmlu": "compliance",
}

STRATEGIES = ["compliance", "security_fud"]


class StrategyMatcher:
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self):
        for strategy in STRATEGIES:
            filepath = os.path.join(TEMPLATE_DIR, f"{strategy}.txt")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    self.templates[strategy] = f.read()
    
    def match(self, scenario: str) -> str:
        return DEFAULT_STRATEGY_MAP.get(scenario, "compliance")
    
    def get_template(self, strategy: str) -> str:
        if strategy not in self.templates:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {STRATEGIES}")
        return self.templates[strategy]
    
    def list_strategies(self) -> list:
        return STRATEGIES
