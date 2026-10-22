"""Metrics Computation."""

from typing import List, Dict
import numpy as np


def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {}
    
    n = len(results)
    infection_rates = [r["infection_rate"] for r in results]
    final_infections = [1 if r["final_infected"] else 0 for r in results]
    n_infected_list = [r["n_infected"] for r in results]
    
    return {
        "n_runs": n,
        "avg_infection_rate": np.mean(infection_rates),
        "std_infection_rate": np.std(infection_rates),
        "final_infection_rate": np.mean(final_infections),  # ASR
        "avg_infected_agents": np.mean(n_infected_list),
        "max_infected_agents": max(n_infected_list),
        "min_infected_agents": min(n_infected_list)
    }


def compute_topology_metrics(results_by_framework: Dict[str, List[Dict]]) -> Dict:
    metrics = {}
    
    for fw_name, results in results_by_framework.items():
        if results:
            metrics[fw_name] = compute_metrics(results)
            if "topology" in results[0]:
                metrics[fw_name]["topology"] = results[0]["topology"]
    
    return metrics


def compute_strategy_metrics(results_by_strategy: Dict[str, List[Dict]]) -> Dict:
    metrics = {}
    
    for strategy, results in results_by_strategy.items():
        if results:
            metrics[strategy] = compute_metrics(results)
    
    return metrics


def compute_scenario_metrics(results_by_scenario: Dict[str, List[Dict]]) -> Dict:
    metrics = {}
    
    for scenario, results in results_by_scenario.items():
        if results:
            metrics[scenario] = compute_metrics(results)
    
    return metrics


def format_results_table(metrics: Dict, title: str = "Results") -> str:
    lines = [f"\n=== {title} ==="]
    lines.append(f"{'Name':<15} {'ASR':>8} {'AvgInf':>8} {'StdInf':>8} {'Runs':>6}")
    lines.append("-" * 50)
    
    for name, m in metrics.items():
        lines.append(
            f"{name:<15} "
            f"{m.get('final_infection_rate', 0):.3f}    "
            f"{m.get('avg_infection_rate', 0):.3f}    "
            f"{m.get('std_infection_rate', 0):.3f}    "
            f"{m.get('n_runs', 0):>6}"
        )
    
    return "\n".join(lines)
