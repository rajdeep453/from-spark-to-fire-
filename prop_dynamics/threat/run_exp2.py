"""Experiment II"""
import json
import os
from tqdm import tqdm
from datetime import datetime

from .config import ThreatConfig, stable_seed
from .payload import create_injection_config
from .detector import InfectionDetector, compute_metrics
from .crewai_adapter import CrewAIAdapter
from .langgraph_adapter import LangGraphAdapter
from .run_exp1 import load_tasks

STAR_FRAMEWORKS = {
    "crewai": {
        "adapter": CrewAIAdapter,
        "hub": "Manager",
        "leaf": "Developer"
    },
    "langgraph": {
        "adapter": LangGraphAdapter,
        "hub": "Supervisor",
        "leaf": "Coder"
    }
}

def run_exp2(
    n_tasks: int = 10,
    n_repeats: int = 3,
    frameworks: list = None,
    output_dir: str = "outputs/threat/exp2",
    save_logs: bool = True
):
    if frameworks is None:
        frameworks = list(STAR_FRAMEWORKS.keys())
    
    os.makedirs(output_dir, exist_ok=True)
    tasks = load_tasks(os.path.join(os.path.dirname(__file__), "tasks.json"), n_tasks)
    
    all_results, all_logs = [], []
    
    for fw_name in frameworks:
        fw_info = STAR_FRAMEWORKS[fw_name]
        config = ThreatConfig(framework=fw_name, n_tasks=n_tasks, n_repeats=n_repeats)
        base_seed = config.seed
        
        for group in ["hub", "leaf"]:
            target_agent = fw_info[group]
            print(f"\n=== {fw_name} - {group} (inject at {target_agent}) ===")
            adapter = fw_info["adapter"](config)
            group_results = []
            
            for task in tqdm(tasks, desc=f"{fw_name}-{group}"):
                for r in range(n_repeats):
                    seed = stable_seed(base_seed, task.id, r)
                    config.seed = seed
                    injection = create_injection_config(task, target_agent, seed)
                    log = adapter.run_task(task, injection)
                    
                    if save_logs:
                        log_dict = log.to_dict()
                        log_dict["group"] = group
                        all_logs.append(log_dict)
                    
                    detector = InfectionDetector(injection.phantom_lib)
                    result = detector.scan_log(log)
                    result.update({
                        "task_id": task.id, "repeat": r,
                        "framework": fw_name, "group": group
                    })
                    group_results.append(result)
            
            metrics = compute_metrics(group_results)
            metrics.update({"framework": fw_name, "group": group, "target": target_agent})
            all_results.append(metrics)
            print(f"{fw_name}-{group}: coverage={metrics['avg_infection_rate']:.3f}, "
                  f"final_rate={metrics['final_infection_rate']:.3f}")
    
    impact_factors = {}
    for fw_name in frameworks:
        hub_m = next((r for r in all_results if r["framework"] == fw_name and r["group"] == "hub"), None)
        leaf_m = next((r for r in all_results if r["framework"] == fw_name and r["group"] == "leaf"), None)
        if hub_m and leaf_m:
            leaf_rate = leaf_m["avg_infection_rate"]
            impact = hub_m["avg_infection_rate"] / leaf_rate if leaf_rate > 0 else float('inf')
            impact_factors[fw_name] = impact
            print(f"\n{fw_name} Impact Factor: {impact:.2f}")
    
    if save_logs and all_logs:
        with open(os.path.join(output_dir, "exp2_logs.jsonl"), "w", encoding="utf-8") as f:
            for log in all_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
    
    with open(os.path.join(output_dir, "exp2_results.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {"n_tasks": n_tasks, "n_repeats": n_repeats, "frameworks": frameworks,
                       "base_seed": base_seed},
            "results": all_results,
            "impact_factors": impact_factors
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return all_results, impact_factors


if __name__ == "__main__":
    run_exp2(n_tasks=10, n_repeats=3)
