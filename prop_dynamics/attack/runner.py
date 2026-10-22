"""Runners."""

import json
import os
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from .config import AttackConfig
from .tasks import load_tasks, Task
from .injector import PayloadGenerator
from .detector import InfectionDetector, create_detector
from .metrics import compute_metrics
from .frameworks import FRAMEWORKS, create_adapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_single_task(
    adapter,
    task: Task,
    payload_gen: PayloadGenerator,
    target_agent: str,
    strategy: Optional[str] = None,
    use_baseline: bool = False
) -> Dict:
    if use_baseline:
        injection = payload_gen.generate_baseline(task, target_agent)
    else:
        injection = payload_gen.generate(task, target_agent, strategy)
    
    log = adapter.run_task(task, injection)
    
    detector = create_detector(injection)
    result = detector.scan_log(log)
    
    result["task_id"] = task.id
    result["framework"] = log.framework
    result["topology"] = log.topology_type
    result["strategy"] = injection.strategy
    result["target_agent"] = target_agent
    
    return result, log


def run_experiment(
    scenario: str,
    frameworks: List[str],
    strategy: Optional[str] = None,
    n_tasks: int = 10,
    n_repeats: int = 3,
    output_dir: str = "outputs/attack",
    save_logs: bool = True
) -> Dict[str, List[Dict]]:
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = load_tasks(scenario, n_tasks)
    payload_gen = PayloadGenerator()
    
    results_by_fw: Dict[str, List[Dict]] = {fw: [] for fw in frameworks}
    all_logs = []
    
    for fw_name in frameworks:
        logging.info(f"Framework: {fw_name}")
        config = AttackConfig(
            framework=fw_name,
            scenario=scenario,
            n_tasks=n_tasks,
            n_repeats=n_repeats,
            output_dir=output_dir
        )
        
        adapter = create_adapter(fw_name, config)
        target_agent = adapter.get_hub_agent() or adapter.get_entry_agent()
        
        for task in tqdm(tasks, desc=fw_name):
            for r in range(n_repeats):
                result, log = run_single_task(
                    adapter, task, payload_gen, target_agent, strategy
                )
                result["repeat"] = r
                results_by_fw[fw_name].append(result)
                
                if save_logs:
                    all_logs.append(log.to_dict())
        
        metrics = compute_metrics(results_by_fw[fw_name])
        logging.info(f"{fw_name}: ASR={metrics['final_infection_rate']:.3f}, AvgInf={metrics['avg_infection_rate']:.3f}")
    
    if save_logs and all_logs:
        log_path = os.path.join(output_dir, "logs.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            for log in all_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
        logging.info(f"Logs saved to {log_path}")
    
    return results_by_fw


def run_strategy_comparison(
    scenario: str,
    framework: str,
    strategies: List[str],
    n_tasks: int = 10,
    n_repeats: int = 3,
    output_dir: str = "outputs/attack"
) -> Dict[str, List[Dict]]:
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = load_tasks(scenario, n_tasks)
    payload_gen = PayloadGenerator()
    
    config = AttackConfig(
        framework=framework,
        scenario=scenario,
        output_dir=output_dir
    )
    
    adapter = create_adapter(framework, config)
    target_agent = adapter.get_hub_agent() or adapter.get_entry_agent()
    
    results_by_strategy: Dict[str, List[Dict]] = {s: [] for s in strategies}
    
    for strategy in strategies:
        logging.info(f"Strategy: {strategy}")
        
        for task in tqdm(tasks, desc=strategy):
            for r in range(n_repeats):
                result, _ = run_single_task(
                    adapter, task, payload_gen, target_agent, strategy
                )
                result["repeat"] = r
                results_by_strategy[strategy].append(result)
        
        metrics = compute_metrics(results_by_strategy[strategy])
        logging.info(f"{strategy}: ASR={metrics['final_infection_rate']:.3f}")
    
    return results_by_strategy


def run_baseline_comparison(
    scenario: str,
    framework: str,
    n_tasks: int = 10,
    n_repeats: int = 3,
    output_dir: str = "outputs/attack"
) -> Dict[str, List[Dict]]:
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = load_tasks(scenario, n_tasks)
    payload_gen = PayloadGenerator()
    
    config = AttackConfig(
        framework=framework,
        scenario=scenario,
        output_dir=output_dir
    )
    
    adapter = create_adapter(framework, config)
    target_agent = adapter.get_hub_agent() or adapter.get_entry_agent()
    
    results = {"strategic": [], "baseline": []}
    
    for condition in ["strategic", "baseline"]:
        logging.info(f"Condition: {condition}")
        use_baseline = (condition == "baseline")
        
        for task in tqdm(tasks, desc=condition):
            for r in range(n_repeats):
                result, _ = run_single_task(
                    adapter, task, payload_gen, target_agent,
                    use_baseline=use_baseline
                )
                result["repeat"] = r
                results[condition].append(result)
        
        metrics = compute_metrics(results[condition])
        logging.info(f"{condition}: ASR={metrics['final_infection_rate']:.3f}")
    
    return results
