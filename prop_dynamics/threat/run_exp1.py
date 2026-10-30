"""Experiment I"""
import json
import os
from tqdm import tqdm
from datetime import datetime

from .config import ThreatConfig, stable_seed
from .base_framework import Task
from .payload import create_injection_config
from .detector import InfectionDetector, compute_metrics
from .langchain_adapter import LangChainAdapter
from .metagpt_adapter import MetaGPTAdapter
from .autogen_adapter import AutoGenAdapter
from .crewai_adapter import CrewAIAdapter
from .langgraph_adapter import LangGraphAdapter
from .camel_adapter import CamelAdapter

# Topology classification
FRAMEWORKS = {
    # Linear (chain)
    "langchain": {"adapter": LangChainAdapter, "topology": "chain"},
    "metagpt": {"adapter": MetaGPTAdapter, "topology": "chain"},
    # Centralized (star)
    "crewai": {"adapter": CrewAIAdapter, "topology": "star"},
    "langgraph": {"adapter": LangGraphAdapter, "topology": "star"},
    # Decentralized (mesh)
    "autogen": {"adapter": AutoGenAdapter, "topology": "mesh"},
    "camel": {"adapter": CamelAdapter, "topology": "mesh"},
}


def load_tasks(path: str, n: int = None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = [Task(**t) for t in data]
    return tasks[:n] if n else tasks


def create_adapter(fw_name: str, config: ThreatConfig):
    if fw_name in FRAMEWORKS:
        return FRAMEWORKS[fw_name]["adapter"](config)
    raise ValueError(f"Unknown framework: {fw_name}")


def run_exp1(
    n_tasks: int = 10,
    n_repeats: int = 3,
    frameworks: list = None,
    output_dir: str = "outputs/threat/exp1",
    save_logs: bool = True
):
    if frameworks is None:
        frameworks = list(FRAMEWORKS.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    tasks = load_tasks(os.path.join(os.path.dirname(__file__), "tasks.json"), n_tasks)
    
    all_results, all_logs = [], []
    
    for fw_name in frameworks:
        print(f"\n=== Framework: {fw_name} ===")
        config = ThreatConfig(framework=fw_name, n_tasks=n_tasks, n_repeats=n_repeats)
        base_seed = config.seed
        adapter = create_adapter(fw_name, config)
        entry_agent = adapter.get_entry_agent()
        fw_results = []
        
        for task in tqdm(tasks, desc=fw_name):
            for r in range(n_repeats):
                seed = stable_seed(base_seed, task.id, r)
                config.seed = seed
                injection = create_injection_config(task, entry_agent, seed)
                log = adapter.run_task(task, injection)
                
                if save_logs:
                    all_logs.append(log.to_dict())
                
                detector = InfectionDetector(injection.phantom_lib)
                result = detector.scan_log(log)
                result.update({"task_id": task.id, "repeat": r, "framework": fw_name})
                fw_results.append(result)
        
        metrics = compute_metrics(fw_results)
        metrics["framework"] = fw_name
        all_results.append(metrics)
        print(f"{fw_name}: infection_rate={metrics['avg_infection_rate']:.3f}, "
              f"final_rate={metrics['final_infection_rate']:.3f}")
    
    if save_logs and all_logs:
        with open(os.path.join(output_dir, "exp1_logs.jsonl"), "w", encoding="utf-8") as f:
            for log in all_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "exp1_results.json"), "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "config": {"n_tasks": n_tasks, "n_repeats": n_repeats, "base_seed": base_seed},
                   "results": all_results}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return all_results


if __name__ == "__main__":
    run_exp1(n_tasks=10, n_repeats=3, frameworks=["camel","crewai","langgraph","langchain","metagpt","autogen"])
