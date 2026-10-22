"""
Mega Attack Experiment
"""

import json
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prop_dynamics.attack.config import AttackConfig, setup_openai
from prop_dynamics.attack.tasks import load_tasks
from prop_dynamics.attack.injector import PayloadGenerator
from prop_dynamics.attack.detector import create_detector
from prop_dynamics.attack.frameworks import create_adapter, get_topology

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FRAMEWORKS_LIST = ["langchain", "metagpt", "crewai", "langgraph", "autogen", "camel"]
SCENARIOS = ["rigid", "mmlu", "quant"]
STRATEGIES = ["baseline", "compliance", "security_fud"]


def run_single_experiment(args):
    fw_name, scenario, strategy, task, repeat_idx, n_tasks, n_repeats, output_dir = args
    
    config = AttackConfig(
        framework=fw_name,
        scenario=scenario,
        n_tasks=n_tasks,
        n_repeats=n_repeats,
        output_dir=output_dir
    )
    adapter = create_adapter(fw_name, config)
    target_agent = adapter.get_hub_agent() or adapter.get_entry_agent()
    
    payload_gen = PayloadGenerator()
    if strategy == "baseline":
        injection = payload_gen.generate_baseline(task, target_agent)
    else:
        injection = payload_gen.generate(task, target_agent, strategy)
    
    log = adapter.run_task(task, injection)
    
    detector = create_detector(injection)
    scan_result = detector.scan_log(log)
    
    result = {
        "framework": fw_name,
        "topology": get_topology(fw_name),
        "scenario": scenario,
        "strategy": strategy,
        "task_id": task.id,
        "repeat": repeat_idx,
        "final_infected": scan_result["final_infected"],
        "final_reason": scan_result["final_reason"],
        "process_infected": scan_result["process_infected"],
        "infection_rate": scan_result["infection_rate"],
        "infected_agents": scan_result["infected_agents"],
        "rebuttal_agents": scan_result["rebuttal_agents"],
        "true_infected_agents": scan_result["true_infected_agents"],
        "n_agents": scan_result["n_total"],
    }
    
    return result, log.to_dict()


def run_mega_experiment(
    n_tasks: int = 3,
    n_repeats: int = 3,
    output_dir: str = "outputs/attack/mega_exp",
    max_workers: int = 4
):
    default_config = AttackConfig()
    setup_openai(default_config)
    os.makedirs(output_dir, exist_ok=True)
    
    all_args = []
    tasks_cache = {}
    
    for fw_name in FRAMEWORKS_LIST:
        for scenario in SCENARIOS:
            if scenario not in tasks_cache:
                tasks_cache[scenario] = load_tasks(scenario, n_tasks)
            tasks = tasks_cache[scenario]
            
            for strategy in STRATEGIES:
                for task in tasks:
                    for r in range(n_repeats):
                        all_args.append((
                            fw_name, scenario, strategy, task, r,
                            n_tasks, n_repeats, output_dir
                        ))
    
    total = len(all_args)
    all_logs = []
    failed_experiments = []
    
    logging.info(f"Running {total} experiments with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_experiment, args): args for args in all_args}
        
        with tqdm(total=total, desc="Mega Experiment") as pbar:
            for future in as_completed(futures):
                args = futures[future]
                try:
                    result, log = future.result()
                    all_logs.append(log)
                    pbar.update(1)
                    pbar.set_postfix({
                        "fw": result["framework"][:4],
                        "sc": result["scenario"][:4],
                        "st": result["strategy"][:4],
                        "inf": int(result["final_infected"])
                    })
                except Exception as e:
                    failed_experiments.append({
                        "framework": args[0],
                        "scenario": args[1],
                        "strategy": args[2],
                        "task_id": args[3].id,
                        "repeat": args[4],
                        "error": str(e)
                    })
                    logging.error(f"Failed: {args[0]}/{args[1]}/{args[2]} - {e}")
                    pbar.update(1)
    
    logs_path = os.path.join(output_dir, "logs.jsonl")
    with open(logs_path, "w", encoding="utf-8") as f:
        for log in all_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    
    if failed_experiments:
        failed_path = os.path.join(output_dir, "failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_experiments, f, indent=2, ensure_ascii=False)
        logging.warning(f"{len(failed_experiments)} experiments failed, saved to {failed_path}")
    
    logging.info(f"Completed {len(all_logs)}/{total} experiments, logs saved to {logs_path}")
    return all_logs


def main():
    n_tasks = 20
    n_repeats = 3
    workers = 20
    output_dir = "outputs/attack/mega_exp"
    run_mega_experiment(n_tasks, n_repeats, output_dir, workers)


if __name__ == "__main__":
    main()
