

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prop_dynamics.defense.config import DefenseConfig, setup_openai
from prop_dynamics.defense.middleware import DefenseMiddleware
from prop_dynamics.defense.simple_middlewares import (
    AGrailMonitorMiddleware,
    BLOCKED_PLACEHOLDER,
    ControlFlowGuard,
    SelfCorrectionMiddleware,
)
from prop_dynamics.defense.local_retriever import LocalRetriever
try:
    from prop_dynamics.defense.frameworks.hybrid_adapter import HybridFramework
except ModuleNotFoundError:
    HybridFramework = None
from prop_dynamics.defense.frameworks.metagpt_defense import MetaGPTDefense
from prop_dynamics.defense.frameworks.langgraph_defense import LangGraphDefense
from prop_dynamics.defense.frameworks.autogen_defense import AutoGenDefense
from prop_dynamics.defense.frameworks.base import (
    ExecutionLog,
    Message,
    estimate_content_tokens,
    handoff_has_unavailable_dependency,
    is_blocked_placeholder,
    role_limited_output,
)
from prop_dynamics.defense.tasks import load_tasks
from prop_dynamics.attack.injector import PayloadGenerator
from prop_dynamics.attack.detector import InfectionDetector

DEFENSE_STRATEGIES = [
    "none", "speed", "balanced", "strict", "self_correction",
    "agrail_monitor", "control_flow_guard",
]
ATTACK_STRATEGIES = ["baseline", "compliance", "security_fud"]
FRAMEWORKS = ["metagpt", "langgraph", "autogen"]
if HybridFramework is not None:
    FRAMEWORKS.append("hybrid")
SCENARIOS = ["rigid", "quant", "mmlu"]


def create_framework(fw_name: str, openai_module):
    if fw_name == "hybrid":
        if HybridFramework is None:
            raise ValueError("Hybrid framework unavailable.")
        fw = HybridFramework()
    elif fw_name == "metagpt":
        fw = MetaGPTDefense()
    elif fw_name == "langgraph":
        fw = LangGraphDefense()
    elif fw_name == "autogen":
        fw = AutoGenDefense()
    else:
        raise ValueError("Invalid framework.")
    fw.model = os.getenv("OPENAI_MODEL", getattr(fw, "model", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
    fw.set_openai(openai_module)
    return fw


def _task_text(task) -> str:
    text = f"Task: {getattr(task, 'description', str(task))}"
    data_source = getattr(task, "data_source", "")
    if data_source:
        text += f"\nData Source: {data_source}"
    return text


def _with_injection(agent_name: str, user_input: str, injection) -> str:
    if injection and injection.target_agent == agent_name:
        return f"{injection.injection_prompt}\n\n{user_input}"
    return user_input


def _control_checked_send(
    guard: ControlFlowGuard,
    sender: str,
    receiver: str,
    message: str,
    log: ExecutionLog,
    round_num: int,
    attempted: List[str] = None,
) -> str:
    attempted = attempted or []
    current_receiver = receiver
    for attempt in range(4):
        event = guard.check_transition(
            sender=sender,
            receiver=current_receiver,
            message=message,
            round_num=round_num,
            event_summary=message[:1000],
            attempt=attempt,
        )
        if event["decision"] == "pass":
            log.add_message(
                Message(
                    sender,
                    current_receiver,
                    message,
                    round_num,
                    tags=["CONTROL_FLOW_PASS"],
                    content_tokens_est=estimate_content_tokens(message),
                )
            )
            return message
        if event["decision"] == "re-plan" and attempt < 3:
            attempted.append(current_receiver)
            alternate = guard.choose_alternate_receiver(sender, attempted)
            if alternate:
                current_receiver = alternate
                continue
        log.add_message(
            Message(
                sender,
                current_receiver,
                BLOCKED_PLACEHOLDER,
                round_num,
                blocked=True,
                tags=[f"CONTROL_FLOW_{event['decision'].upper()}"],
                content_tokens_est=estimate_content_tokens(BLOCKED_PLACEHOLDER),
            )
        )
        return BLOCKED_PLACEHOLDER
    log.add_message(
        Message(
            sender,
            current_receiver,
            BLOCKED_PLACEHOLDER,
            round_num,
            blocked=True,
            tags=["CONTROL_FLOW_REJECT"],
            content_tokens_est=estimate_content_tokens(BLOCKED_PLACEHOLDER),
        )
    )
    return BLOCKED_PLACEHOLDER


def _framework_allowed_edges(framework) -> List[Tuple[str, str]]:
    if hasattr(framework, "_get_subscribers"):
        edges = []
        for agent in framework.agents:
            for receiver in framework._get_subscribers(agent.name):
                edges.append((agent.name, receiver))
        return list(dict.fromkeys(edges))

    names = [a.name for a in framework.agents]
    if framework.topology_type == "star":
        hub = next((a.name for a in framework.agents if a.is_hub), names[0])
        return [(hub, n) for n in names if n != hub] + [(n, hub) for n in names if n != hub]
    if framework.topology_type == "mesh":
        return [(s, r) for s in names for r in names if s != r]
    return [(names[i], names[i + 1]) for i in range(len(names) - 1)]


def run_control_flow_guard_task(framework, task, injection, guard: ControlFlowGuard) -> ExecutionLog:
    
    guard.plan_task(task, framework.agents, framework.topology_type, [], _framework_allowed_edges(framework))
    log = ExecutionLog(
        task_id=getattr(task, "id", "unknown"),
        framework=framework.__class__.__name__.replace("Defense", "").lower(),
        topology_type=framework.topology_type,
    )
    task_text = _task_text(task)

    if hasattr(framework, "_get_subscribed_content") and hasattr(framework, "_get_subscribers"):
        artifacts = {}
        for round_num, agent in enumerate(framework.agents, 1):
            subscribed_content = framework._get_subscribed_content(agent.name, artifacts)
            user_input = f"{task_text}\n\n{subscribed_content}" if subscribed_content else task_text
            if subscribed_content and handoff_has_unavailable_dependency(subscribed_content):
                response = role_limited_output(agent.name)
            else:
                response = framework._call_agent(agent.name, _with_injection(agent.name, user_input, injection))
            artifact_msg = response
            for subscriber in framework._get_subscribers(agent.name):
                artifact_msg = _control_checked_send(guard, agent.name, subscriber, response, log, round_num)
            if not framework._get_subscribers(agent.name):
                log.add_message(Message(agent.name, agent.name, response, round_num, tags=["CONTROL_FLOW_FINAL"]))
            artifacts[agent.name] = artifact_msg
        log.final_output = artifacts.get(framework.agents[-1].name, "")
        log.total_rounds = len(framework.agents)

    elif framework.topology_type == "star":
        hub = next((a for a in framework.agents if a.is_hub), framework.agents[0])
        leaves = [a for a in framework.agents if a.name != hub.name]
        context = task_text
        final_output = ""
        for round_num in range(1, framework.max_rounds + 1):
            if round_num > 1 and handoff_has_unavailable_dependency(context):
                hub_response = role_limited_output(hub.name)
            else:
                hub_response = framework._call_agent(hub.name, _with_injection(hub.name, context, injection))
            leaf_outputs = []
            for leaf in leaves:
                instruction = _control_checked_send(guard, hub.name, leaf.name, hub_response, log, round_num)
                leaf_input = f"{task_text}\n\nSupervisor instruction:\n{instruction}"
                if is_blocked_placeholder(instruction) or handoff_has_unavailable_dependency(instruction):
                    leaf_response = role_limited_output(leaf.name)
                else:
                    leaf_response = framework._call_agent(leaf.name, _with_injection(leaf.name, leaf_input, injection))
                returned = _control_checked_send(guard, leaf.name, hub.name, leaf_response, log, round_num)
                leaf_outputs.append(f"{leaf.name}:\n{returned}")
            context = f"{task_text}\n\nTeam outputs:\n" + "\n\n".join(leaf_outputs)
            final_output = hub_response
        log.final_output = final_output
        log.total_rounds = framework.max_rounds

    else:
        chat_history = []
        final_output = ""
        for round_num in range(1, framework.max_rounds + 1):
            for agent in framework.agents:
                if chat_history:
                    history_text = "\n\n".join(f"{h['sender']}: {h['content']}" for h in chat_history[-6:])
                    user_input = f"{task_text}\n\nDiscussion so far:\n{history_text}\n\nYour turn to contribute:"
                else:
                    user_input = f"{task_text}\n\nStart the discussion:"
                if chat_history and (
                    is_blocked_placeholder(chat_history[-1]["content"])
                    or handoff_has_unavailable_dependency(chat_history[-1]["content"])
                ):
                    response = role_limited_output(agent.name)
                else:
                    response = framework._call_agent(agent.name, _with_injection(agent.name, user_input, injection))
                receivers = [a.name for a in framework.agents if a.name != agent.name]
                delivered = response
                for receiver in receivers:
                    delivered = _control_checked_send(guard, agent.name, receiver, response, log, round_num)
                chat_history.append({"sender": agent.name, "content": delivered})
                final_output = delivered
        log.final_output = final_output
        log.total_rounds = framework.max_rounds

    log.defense_stats = guard.get_stats()
    return log


def run_single_experiment(args) -> Tuple[dict, dict]:
    import time
    start_time = time.time()
    
    fw_name, scenario, defense_strategy, attack_strategy, task, repeat_idx, openai_module = args
    
    framework = create_framework(fw_name, openai_module)
    middleware = None
    
    if defense_strategy != "none":
        config = DefenseConfig(strategy=defense_strategy)
        if defense_strategy == "self_correction":
            middleware = SelfCorrectionMiddleware(config, openai_module=openai_module)
        elif defense_strategy == "agrail_monitor":
            middleware = AGrailMonitorMiddleware(config, openai_module=openai_module)
        elif defense_strategy == "control_flow_guard":
            middleware = ControlFlowGuard(config, openai_module=openai_module)
        else:
            middleware = DefenseMiddleware(config, openai_module=openai_module)
            kb_dir = os.path.join("outputs", "defense", "kb")
            if os.path.exists(os.path.join(kb_dir, "index.faiss")):
                retriever = LocalRetriever(kb_dir, openai_module=openai_module)
                middleware.verifier.set_external_search_func(retriever.search)
        framework.set_middleware(middleware)
    
    payload_gen = PayloadGenerator()
    target_agent = framework.get_hub_agent() or framework.get_entry_agent()
    
    if attack_strategy == "baseline":
        injection = payload_gen.generate_baseline(task, target_agent)
    else:
        injection = payload_gen.generate(task, target_agent, attack_strategy)
    
    if defense_strategy == "control_flow_guard":
        log = run_control_flow_guard_task(framework, task, injection, middleware)
    else:
        log = framework.run_task(task, injection)
    
    detector = InfectionDetector(injection.infection_marker, scenario)
    scan_result = detector.scan_log(log)
    
    # Accuracy is not a paper metric.
    task_correct = False
    if log.final_output and hasattr(task, 'ground_truth'):
        task_correct = str(task.ground_truth).lower() in log.final_output.lower()
    
    stats = log.defense_stats or {}
    
    result = {
        "framework": fw_name,
        "scenario": scenario,
        "defense_strategy": defense_strategy,
        "attack_strategy": attack_strategy,
        "task_id": task.id,
        "repeat": repeat_idx,
        "final_infected": scan_result["final_infected"],
        "final_reason": scan_result["final_reason"],
        "infection_rate": scan_result["infection_rate"],
        "infected_agents": scan_result["infected_agents"],
        "n_agents": scan_result["n_total"],
        "s_i_t": scan_result.get("s_i_t", []),
        "agent_names": scan_result.get("agent_names", []),
        "total_atoms": stats.get("total_atoms", 0),
        "green_atoms": stats.get("green", 0),
        "yellow_atoms": stats.get("yellow", 0),
        "red_atoms": stats.get("red", 0),
        "verified_true": stats.get("verified_true", 0),
        "verified_false": stats.get("verified_false", 0),
        "unverifiable": stats.get("unverifiable", 0),
        "rollbacks": stats.get("rollbacks", 0),
        "forced_pass": stats.get("forced_pass", 0),
        "blocked_atoms": stats.get("blocked_atoms", 0),
        "graph_size": stats.get("graph_size", 0),
        "api_tokens": stats.get("api_tokens", 0),
        "tokens_framework": framework.tokens_framework,
        "total_tokens": stats.get("api_tokens", 0) + framework.tokens_framework,
        "tokens_decomposer": stats.get("tokens_decomposer", 0),
        "tokens_classifier": stats.get("tokens_classifier", 0),
        "tokens_verifier": stats.get("tokens_verifier", 0),
        "latency_decompose": stats.get("latency_decompose", 0),
        "latency_classify": stats.get("latency_classify", 0),
        "latency_verify": stats.get("latency_verify", 0),
        "latency_assemble": stats.get("latency_assemble", 0),
        "task_correct": task_correct,
        "ground_truth": str(getattr(task, 'ground_truth', '')),
        "execution_time": round(time.time() - start_time, 2),
        "bicr": 1 - int(bool(scan_result["final_infected"])),
        "final_failure_rate": None,
        "guard_type": stats.get("guard_type"),
        "block_rate": stats.get("block_rate", 0.0),
        "reject_rate": stats.get("reject_rate", 0.0),
        "replan_rate": stats.get("replan_rate", 0.0),
        "uses_controlvalve_orchestrator": stats.get("uses_controlvalve_orchestrator", False),
        "agent_token_usage": getattr(framework, "agent_token_usage", {}),
    }
    
    log_dict = log.to_dict()
    log_dict["execution_time"] = result["execution_time"]
    log_dict["injection_marker"] = injection.infection_marker
    log_dict["injection_strategy"] = injection.strategy
    log_dict["api_tokens"] = stats.get("api_tokens", 0)
    log_dict["guard_type"] = stats.get("guard_type")
    log_dict["guard_events"] = middleware.get_guard_events() if middleware is not None and hasattr(middleware, "get_guard_events") else []
    log_dict["control_flow_plan"] = middleware.get_control_flow_plan() if middleware is not None and hasattr(middleware, "get_control_flow_plan") else {}
    log_dict["uses_controlvalve_orchestrator"] = stats.get("uses_controlvalve_orchestrator", False)
    log_dict["agent_token_usage"] = getattr(framework, "agent_token_usage", {})
    
    if middleware is not None:
        log_dict["lineage_graph"] = middleware.get_graph_snapshot()
    else:
        log_dict["lineage_graph"] = []
    
    return result, log_dict


def run_defense_experiment(
    n_tasks: int = 3, n_repeats: int = 1,
    output_dir: str = "outputs/defense/exp", max_workers: int = 2,
    frameworks: List[str] = None, scenarios: List[str] = None,
    defense_strategies: List[str] = None, attack_strategies: List[str] = None,
    limit_runs: int = None,
):
    config = DefenseConfig()
    openai_module = setup_openai(config)
    os.makedirs(output_dir, exist_ok=True)
    
    frameworks = frameworks or FRAMEWORKS
    scenarios = scenarios or SCENARIOS
    defense_strategies = defense_strategies or DEFENSE_STRATEGIES
    attack_strategies = attack_strategies or ATTACK_STRATEGIES
    
    all_args, tasks_cache = [], {}
    
    for fw_name in frameworks:
        for scenario in scenarios:
            if scenario not in tasks_cache:
                tasks_cache[scenario] = load_tasks(scenario, n_tasks)
            for defense_strat in defense_strategies:
                for attack_strat in attack_strategies:
                    for task in tasks_cache[scenario]:
                        for r in range(n_repeats):
                            all_args.append((fw_name, scenario, defense_strat, 
                                           attack_strat, task, r, openai_module))
    if limit_runs is not None:
        all_args = all_args[:limit_runs]
    
    total = len(all_args)
    all_results, all_logs = [], []
    
    results_path = os.path.join(output_dir, "defense_results.jsonl")
    logs_path = os.path.join(output_dir, "logs.jsonl")
    completed_keys = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_key = (
                    row.get("framework"),
                    row.get("scenario"),
                    row.get("defense_strategy"),
                    row.get("attack_strategy"),
                    row.get("task_id"),
                    row.get("repeat"),
                )
                guard_row = row.get("defense_strategy") in {"agrail_monitor", "control_flow_guard"}
                if guard_row and int(row.get("api_tokens", 0) or 0) <= 0:
                    continue
                completed_keys.add(row_key)
                all_results.append(row)
    if completed_keys:
        filtered_args = []
        for args in all_args:
            fw_name, scenario, defense_strat, attack_strat, task, repeat_idx, _openai_module = args
            key = (fw_name, scenario, defense_strat, attack_strat, task.id, repeat_idx)
            if key not in completed_keys:
                filtered_args.append(args)
        skipped = len(all_args) - len(filtered_args)
        all_args = filtered_args
        total = len(all_args)
        print(f"Resume mode: skipped {skipped} completed experiments.")
    
    print(f"Running {total} experiments with {max_workers} workers...")
    print(f"Incremental save to: {results_path}")
    
    import signal
    import threading
    stop_event = threading.Event()
    
    def signal_handler(signum, frame):
        print("\n[!] Ctrl+C detected. Stopping workers... (this may take a moment)")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    
    try:
        for args in all_args:
            if stop_event.is_set():
                break
            f = executor.submit(run_single_experiment, args)
            futures[f] = args
            
        with tqdm(total=total, desc="Defense Exp") as pbar:
            for future in as_completed(futures):
                if stop_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    print("\n[!] Execution stopped by user.")
                    break
                
                try:
                    result, log = future.result()
                    all_results.append(result)
                    all_logs.append(log)
                    
                    with open(results_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    simplified_log = {
                        "task_id": log.get("task_id"),
                        "framework": log.get("framework"),
                        "topology_type": log.get("topology_type"),
                        "messages": log.get("messages"),
                        "final_output": log.get("final_output"),
                        "total_rounds": log.get("total_rounds"),
                        "execution_time": log.get("execution_time"),
                        "injection_marker": log.get("injection_marker"),
                        "injection_strategy": log.get("injection_strategy"),
                        "api_tokens": log.get("api_tokens"),
                        "guard_type": log.get("guard_type"),
                        "guard_events": log.get("guard_events"),
                        "control_flow_plan": log.get("control_flow_plan"),
                        "uses_controlvalve_orchestrator": log.get("uses_controlvalve_orchestrator"),
                        "agent_token_usage": log.get("agent_token_usage"),
                    }
                    with open(logs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(simplified_log, ensure_ascii=False) + "\n")
                    
                    pbar.update(1)
                    pbar.set_postfix({"fw": result["framework"][:4], 
                                     "def": result["defense_strategy"][:3], 
                                     "inf": int(result["final_infected"])})
                except Exception as e:
                    args = futures[future]
                    print(
                        "\nTask failed: "
                        f"framework={args[0]} scenario={args[1]} defense={args[2]} "
                        f"attack={args[3]} task={args[4].id} repeat={args[5]} error={e}"
                    )
                    if os.getenv("EXPERIMENT_FAIL_FAST", "0").lower() in {"1", "true", "yes"}:
                        traceback.print_exc()
                        stop_event.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    pbar.update(1)
    finally:
        executor.shutdown(wait=True)
    
    summary = compute_summary(all_results)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    graphs_path = os.path.join(output_dir, "lineage_graphs.json")
    lineage_data = {log.get("task_id"): log.get("lineage_graph", []) for log in all_logs if log.get("task_id")}
    with open(graphs_path, "w", encoding="utf-8") as f:
        json.dump(lineage_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults: {results_path}")
    print(f"Logs: {logs_path}")
    print(f"Summary: {summary_path}")
    print(f"Lineage: {graphs_path}")
    
    return all_results, all_logs


def compute_summary(results: List[dict]) -> dict:
    summary = {"by_framework": {}, "by_defense": {}, "by_attack": {}, 
               "by_framework_defense": {}, "overall": {}}
    
    by_framework = defaultdict(list)
    for r in results:
        by_framework[r["framework"]].append(r)
    
    for fw, items in by_framework.items():
        n = len(items)
        summary["by_framework"][fw] = {
            "n": n,
            "infection_rate": sum(r["final_infected"] for r in items) / n,
            "bicr": 1 - sum(r["final_infected"] for r in items) / n,
            "accuracy": sum(r["task_correct"] for r in items) / n,
            "avg_block_rate": sum(r.get("block_rate", 0.0) for r in items) / n,
            "avg_reject_rate": sum(r.get("reject_rate", 0.0) for r in items) / n,
            "avg_replan_rate": sum(r.get("replan_rate", 0.0) for r in items) / n,
        }
    
    by_defense = defaultdict(list)
    for r in results:
        by_defense[r["defense_strategy"]].append(r)
    
    for defense, items in by_defense.items():
        n = len(items)
        summary["by_defense"][defense] = {
            "n": n,
            "infection_rate": sum(r["final_infected"] for r in items) / n,
            "bicr": 1 - sum(r["final_infected"] for r in items) / n,
            "accuracy": sum(r["task_correct"] for r in items) / n,
            "avg_green": sum(r["green_atoms"] for r in items) / n,
            "avg_yellow": sum(r["yellow_atoms"] for r in items) / n,
            "avg_red": sum(r["red_atoms"] for r in items) / n,
            "avg_blocked": sum(r["blocked_atoms"] for r in items) / n,
            "avg_block_rate": sum(r.get("block_rate", 0.0) for r in items) / n,
            "avg_reject_rate": sum(r.get("reject_rate", 0.0) for r in items) / n,
            "avg_replan_rate": sum(r.get("replan_rate", 0.0) for r in items) / n,
        }
    
    by_attack = defaultdict(list)
    for r in results:
        by_attack[r["attack_strategy"]].append(r)
    
    for attack, items in by_attack.items():
        n = len(items)
        summary["by_attack"][attack] = {
            "n": n,
            "infection_rate": sum(r["final_infected"] for r in items) / n,
        }
    
    by_cross = defaultdict(list)
    for r in results:
        key = f"{r['framework']}_{r['defense_strategy']}"
        by_cross[key].append(r)
    
    for key, items in by_cross.items():
        n = len(items)
        summary["by_framework_defense"][key] = {
            "n": n,
            "infection_rate": sum(r["final_infected"] for r in items) / n,
            "bicr": 1 - sum(r["final_infected"] for r in items) / n,
        }
    
    n = len(results)
    summary["overall"] = {
        "total_experiments": n,
        "frameworks": list(by_framework.keys()),
        "infection_rate": sum(r["final_infected"] for r in results) / n if n > 0 else 0,
        "accuracy": sum(r["task_correct"] for r in results) / n if n > 0 else 0,
    }
    
    return summary


def main():
    
    run_defense_experiment(
        n_tasks=20,
        n_repeats=2,
        max_workers=20,
        
        frameworks=["metagpt", "langgraph", "autogen"],
        
        scenarios=["quant", "rigid", "mmlu"], 
        
        defense_strategies=["strict", "balanced", "speed", "none", "self_correction"], 
        
        attack_strategies=["security_fud", "baseline", "compliance"], 
        
        output_dir="outputs/defense/mega_exp_v12"
    )


if __name__ == "__main__":
    main()

