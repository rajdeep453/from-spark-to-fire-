"""Task loading utilities for defense experiments."""

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    id: str
    name: str
    description: str
    scenario: str
    ground_truth: str
    resource_url: str = ""
    data_source: str = ""
    facts: List[str] = None
    
    def __post_init__(self):
        if self.facts is None:
            self.facts = []


def load_tasks(scenario: str, n_tasks: int = 10, tasks_dir: str = None) -> List[Task]:
    if tasks_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tasks_dir = os.path.join(base_dir, "..", "attack", "tasks")
    
    task_file = os.path.join(tasks_dir, f"{scenario}_tasks.json")
    if not os.path.exists(task_file):
        raise FileNotFoundError("Task source unavailable.")
    
    with open(task_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tasks = []
    for item in data[:n_tasks]:
        tasks.append(Task(
            id=item.get("id", ""),
            name=item.get("name", ""),
            description=item.get("description", ""),
            scenario=item.get("scenario", scenario),
            ground_truth=item.get("ground_truth", ""),
            resource_url=item.get("resource_url", ""),
            data_source=item.get("data_source", ""),
            facts=item.get("facts", []),
        ))
    return tasks
