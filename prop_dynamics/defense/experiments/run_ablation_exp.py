import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
from collections import defaultdict
from threading import Lock

_file_lock = Lock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prop_dynamics.defense.config import DefenseConfig, setup_openai
from prop_dynamics.defense.middleware import DefenseMiddleware
from prop_dynamics.defense.local_retriever import LocalRetriever
from prop_dynamics.defense.frameworks.metagpt_defense import MetaGPTDefense
from prop_dynamics.defense.frameworks.langgraph_defense import LangGraphDefense
from prop_dynamics.defense.frameworks.autogen_defense import AutoGenDefense
from prop_dynamics.defense.tasks import load_tasks
from prop_dynamics.attack.injector import PayloadGenerator
from prop_dynamics.attack.detector import InfectionDetector


def create_framework(fw_name: str, openai_module):
    if fw_name == "metagpt":
        fw = MetaGPTDefense()
    elif fw_name == "langgraph":
        fw = LangGraphDefense()
    elif fw_name == "autogen":
        fw = AutoGenDefense()
    else:
        raise ValueError("Invalid framework.")
    fw.set_openai(openai_module)
    return fw

ABLATION_CONFIGS = {
    "no_atomization": {
        "enable_decomposer": False,
        "enable_classifier": True,
        "enable_verifier": True,
        "enable_graph": True,
        "init_facts": False,
        "suppress_tags": False,
        "strategy": "strict"
    },
    "no_detection": {
        "enable_decomposer": True,
        "enable_classifier": False,
        "enable_verifier": False,
        "enable_graph": True,
        "init_facts": False,
        "suppress_tags": False,
        "strategy": "strict"
    },
    "no_blocking": {
        "enable_decomposer": True,
        "enable_classifier": True,
        "enable_verifier": True,
        "enable_graph": True,
        "init_facts": False,
        "suppress_tags": True,
        "strategy": "strict"
    },
    "none": {
        "enable_decomposer": False,
        "enable_classifier": False,
        "enable_verifier": False,
        "enable_graph": False,
        "init_facts": False,
        "suppress_tags": True,
        "strategy": "none"
    },
}

FRAMEWORKS = ["metagpt", "langgraph", "autogen"]
SCENARIOS = ["quant", "rigid", "mmlu"]
ATTACK_STRATEGIES = ["baseline", "compliance", "security_fud"]


def run_single_ablation(args) -> Dict:
    import time
    start_time = time.time()
    
    (ablation_name, framework_name, scenario, attack_strategy, 
     task, repeat_idx, openai_module, shared_retriever, output_dir) = args
    ablation_cfg = ABLATION_CONFIGS[ablation_name]
    
    framework = create_framework(framework_name, openai_module)
    
    config = DefenseConfig(
        strategy=ablation_cfg["strategy"],
        enable_decomposer=ablation_cfg["enable_decomposer"],
        enable_classifier=ablation_cfg["enable_classifier"],
        enable_verifier=ablation_cfg["enable_verifier"],
        enable_graph=ablation_cfg["enable_graph"],
        init_facts=ablation_cfg.get("init_facts", False),
        suppress_tags=ablation_cfg.get("suppress_tags", False),
    )
    
    middleware = None
    if ablation_name != "none":
        middleware = DefenseMiddleware(config, openai_module=openai_module)
        if shared_retriever is not None:
            middleware.verifier.set_external_search_func(shared_retriever.search)
        framework.set_middleware(middleware)
    
    payload_gen = PayloadGenerator()
    target_agent = framework.get_hub_agent() or framework.get_entry_agent()
    injection = payload_gen.generate(task, target_agent, attack_strategy)
    
    log = framework.run_task(task, injection)
    
    log_entry = log.to_dict()
    log_entry["ablation"] = ablation_name
    log_entry["framework"] = framework_name
    log_entry["scenario"] = scenario
    log_entry["attack_strategy"] = attack_strategy
    log_entry["config"] = ablation_cfg
    
    log_file = os.path.join(output_dir, "logs.jsonl")
    with _file_lock:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    detector = InfectionDetector(injection.infection_marker, scenario)
    scan_result = detector.scan_log(log)
    
    # Accuracy is not a paper metric.
    task_correct = False
    if log.final_output and hasattr(task, 'ground_truth'):
        task_correct = str(task.ground_truth).lower() in log.final_output.lower()
    
    stats = log.defense_stats or {}
    execution_time = round(time.time() - start_time, 2)
    
    result = {
        "ablation": ablation_name,
        "framework": framework_name,
        "scenario": scenario,
        "attack_strategy": attack_strategy,
        "task_id": task.id,
        "repeat": repeat_idx,
        "final_infected": scan_result["final_infected"],
        "infection_rate": scan_result["infection_rate"],
        "s_i_t": scan_result.get("s_i_t", []),
        "agent_names": scan_result.get("agent_names", []),
        "total_atoms": stats.get("total_atoms", 0),
        "green_atoms": stats.get("green", 0),
        "yellow_atoms": stats.get("yellow", 0),
        "red_atoms": stats.get("red", 0),
        "blocked_atoms": stats.get("blocked_atoms", 0),
        "api_tokens": stats.get("api_tokens", 0),
        "tokens_framework": framework.tokens_framework,
        "tokens_total": stats.get("api_tokens", 0) + framework.tokens_framework,
        "execution_time": execution_time,
        "task_correct": task_correct,
        "enable_decomposer": ablation_cfg["enable_decomposer"],
        "enable_classifier": ablation_cfg["enable_classifier"],
        "enable_verifier": ablation_cfg["enable_verifier"],
        "enable_graph": ablation_cfg["enable_graph"],
    }
    
    import gc
    del log, framework, middleware, detector, scan_result, payload_gen, injection
    gc.collect()
    
    return result


def run_ablation_experiment(
    n_tasks: int = 5,
    n_repeats: int = 3,
    output_dir: str = "outputs/defense/ablation_exp",
    max_workers: int = 20,
    frameworks: List[str] = None,
    scenarios: List[str] = None,
    attack_strategies: List[str] = None,
):
    config = DefenseConfig()
    openai_module = setup_openai(config)
    os.makedirs(output_dir, exist_ok=True)
    
    frameworks = frameworks or FRAMEWORKS
    scenarios = scenarios or SCENARIOS
    attack_strategies = attack_strategies or ATTACK_STRATEGIES
    
    tasks_by_scenario = {}
    for scenario in scenarios:
        tasks_by_scenario[scenario] = load_tasks(scenario, n_tasks)
    
    shared_retriever = None
    kb_dir = os.path.join("outputs", "defense", "kb")
    if os.path.exists(os.path.join(kb_dir, "index.faiss")):
        shared_retriever = LocalRetriever(kb_dir, openai_module=openai_module)
    
    all_args = []
    for ablation_name in ABLATION_CONFIGS.keys():
        for fw_name in frameworks:
            for scenario in scenarios:
                for attack_strat in attack_strategies:
                    for task in tasks_by_scenario[scenario]:
                        for r in range(n_repeats):
                            all_args.append((
                                ablation_name, fw_name, scenario, attack_strat,
                                task, r, openai_module, shared_retriever, output_dir
                            ))
    
    total = len(all_args)
    all_results = []
    
    results_path = os.path.join(output_dir, "ablation_results.jsonl")
    
    print(f"Ablation: {list(ABLATION_CONFIGS.keys())} | Total: {total} | Workers: {max_workers}")
    
    try:
        from prop_dynamics.defense.classifier_nli import _get_shared_model
        _get_shared_model("cross-encoder/nli-deberta-v3-small")
    except Exception:
        pass
    
    import signal
    import threading
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        print("\nStopping...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    
    try:
        for args in all_args:
            if stop_event.is_set():
                break
            f = executor.submit(run_single_ablation, args)
            futures[f] = args
        
        with tqdm(total=total, desc="Ablation Exp") as pbar:
            for future in as_completed(futures):
                if stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
                try:
                    result = future.result(timeout=300)  # 5-minute timeout per task
                    all_results.append(result)
                    
                    with open(results_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "abl": result["ablation"][:6], 
                        "inf": int(result["final_infected"])
                    })
                except Exception:
                    print("\nTask failed.")
                    pbar.update(1)
    finally:
        executor.shutdown(wait=True)
    
    summary = compute_ablation_summary(all_results)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults: {results_path}")
    print(f"Summary: {summary_path}")
    
    print(f"\n{'Variant':<15} {'n':>5} {'BICR':>12} {'Tokens':>10} {'Time(s)':>10}")
    for name, data in summary["by_ablation"].items():
        n = data["n"]
        bicr = data["bicr"] * 100
        bicr_std = data["bicr_std"] * 100
        tokens = data["avg_tokens_total"]
        time_s = data["avg_time"]
        time_std = data["std_time"]
        print(f"{name:<15} {n:>5} {bicr:>5.1f}±{bicr_std:<5.1f} {tokens:>10.0f} {time_s:>4.1f}±{time_std:<4.1f}")
    
    return all_results


def compute_ablation_summary(results: List[Dict]) -> Dict:
    import numpy as np
    
    summary = {"by_ablation": {}, "by_framework": {}, "by_scenario": {}, 
               "by_attack": {}, "by_ablation_framework": {}, "overall": {}}
    
    def compute_stats(items: List[Dict]) -> Dict:
        n = len(items)
        if n == 0:
            return {"n": 0, "bicr": 0, "bicr_std": 0, "accuracy": 0, "accuracy_std": 0}
        
        infections = [r["final_infected"] for r in items]
        accuracies = [r["task_correct"] for r in items]
        tokens_total = [r["tokens_total"] for r in items]
        tokens_defense = [r["api_tokens"] for r in items]
        times = [r["execution_time"] for r in items]
        blocked = [r["blocked_atoms"] for r in items]
        
        infection_rate = np.mean(infections)
        return {
            "n": n,
            "infection_rate": infection_rate,
            "infection_std": np.std(infections),
            "bicr": 1 - infection_rate,
            "bicr_std": np.std(infections),  # Same as infection std
            "accuracy": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "avg_tokens_total": np.mean(tokens_total),
            "std_tokens_total": np.std(tokens_total),
            "avg_tokens_defense": np.mean(tokens_defense),
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "avg_blocked": np.mean(blocked),
        }
    
    by_ablation = defaultdict(list)
    for r in results:
        by_ablation[r["ablation"]].append(r)
    for name, items in by_ablation.items():
        summary["by_ablation"][name] = compute_stats(items)
    
    by_framework = defaultdict(list)
    for r in results:
        by_framework[r["framework"]].append(r)
    for name, items in by_framework.items():
        summary["by_framework"][name] = compute_stats(items)
    
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["scenario"]].append(r)
    for name, items in by_scenario.items():
        summary["by_scenario"][name] = compute_stats(items)
    
    by_attack = defaultdict(list)
    for r in results:
        by_attack[r["attack_strategy"]].append(r)
    for name, items in by_attack.items():
        summary["by_attack"][name] = compute_stats(items)
    
    by_cross = defaultdict(list)
    for r in results:
        key = f"{r['ablation']}_{r['framework']}"
        by_cross[key].append(r)
    for key, items in by_cross.items():
        summary["by_ablation_framework"][key] = compute_stats(items)
    
    n = len(results)
    summary["overall"] = {
        "total_experiments": n,
        "ablation_variants": list(by_ablation.keys()),
        "frameworks": list(by_framework.keys()),
        "scenarios": list(by_scenario.keys()),
        "attack_strategies": list(by_attack.keys()),
    }
    
    return summary


def main():
    import sys
    try:
        run_ablation_experiment(
            n_tasks=10,       
            n_repeats=1,   
            max_workers=10,  
            frameworks=["metagpt", "langgraph", "autogen"],
            scenarios=["quant", "rigid", "mmlu"],
            attack_strategies=["compliance"],
            output_dir="outputs/defense/ablation_exp_full3"
        )
    except Exception:
        print("\nExecution failed.")
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()

# python -m prop_dynamics.defense.experiments.run_ablation_exp
