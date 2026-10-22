"""Tasks."""

import os
import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Task:
    id: str
    name: str
    description: str
    scenario: str  
    ground_truth: Optional[str] = None  
    resource_url: str = ""  
    data_source: str = ""  
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


def load_tasks(scenario: str, n: int = None) -> List[Task]:
    task_dir = os.path.dirname(__file__)
    filename = f"{scenario}_tasks.json"
    filepath = os.path.join(task_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Task file not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tasks = [Task.from_dict(t) for t in data]
    return tasks[:n] if n else tasks


def load_all_tasks(n_per_scenario: int = None) -> dict:
    scenarios = ["rigid", "quant", "mmlu"]
    all_tasks = {}
    
    for scenario in scenarios:
        try:
            all_tasks[scenario] = load_tasks(scenario, n_per_scenario)
        except FileNotFoundError:
            all_tasks[scenario] = []
    
    return all_tasks
