"""
Mega Experiment Analysis
"""

import json
import os
from datetime import datetime

from prop_dynamics.attack.config import INFECTION_MARKERS
from prop_dynamics.attack.detector import InfectionDetector
from prop_dynamics.attack.frameworks.base import ExecutionLog, Message, MessageType
from prop_dynamics.attack.frameworks import get_topology


def load_logs(log_path: str) -> list:
    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            logs.append(json.loads(line))
    return logs


def _infer_scenario(log_dict: dict) -> str:
    task_id = log_dict.get("task_id", "")
    if "_" in task_id:
        prefix = task_id.split("_", 1)[0]
        if prefix in INFECTION_MARKERS:
            return prefix
    marker = (log_dict.get("infection_marker") or "").lower()
    for scenario, info in INFECTION_MARKERS.items():
        if info["marker"].lower() in marker:
            return scenario
    return "unknown"


def _normalize_final_output(final_output) -> str:
    if not final_output:
        return ""
    if isinstance(final_output, dict):
        return "\n".join(str(v) for v in final_output.values())
    if isinstance(final_output, list):
        return "\n".join(str(v) for v in final_output)
    return str(final_output)


def _to_message_type(msg_type: str) -> MessageType:
    if msg_type == "artifact":
        return MessageType.ARTIFACT
    if msg_type == "injection":
        return MessageType.INJECTION
    return MessageType.CONVERSATION


def _build_execution_log(log_dict: dict) -> ExecutionLog:
    messages = []
    for msg in log_dict.get("messages", []):
        messages.append(
            Message(
                sender=msg.get("sender", ""),
                content=msg.get("content", ""),
                round=msg.get("round", 0),
                msg_type=_to_message_type(msg.get("msg_type", "conversation"))
            )
        )
    return ExecutionLog(
        task_id=log_dict.get("task_id", ""),
        framework=log_dict.get("framework", ""),
        topology_type=log_dict.get("topology_type", ""),
        messages=messages,
        final_output=_normalize_final_output(log_dict.get("final_output", "")),
    )


def analyze_logs(log_path: str, output_dir: str) -> list:
    logs = load_logs(log_path)
    results = []

    for log_dict in logs:
        scenario = _infer_scenario(log_dict)
        marker = log_dict.get("infection_marker")
        if not marker and scenario in INFECTION_MARKERS:
            marker = INFECTION_MARKERS[scenario]["marker"]

        detector = InfectionDetector(infection_marker=marker or "", scenario=scenario)
        exec_log = _build_execution_log(log_dict)
        scan_result = detector.scan_log(exec_log)

        result = {
            "framework": log_dict.get("framework", ""),
            "topology": get_topology(log_dict.get("framework", "")),
            "scenario": scenario,
            "strategy": log_dict.get("strategy", "unknown"),
            "task_id": log_dict.get("task_id", ""),
            "repeat": log_dict.get("repeat", 0),
            "final_infected": scan_result["final_infected"],
            "final_reason": scan_result["final_reason"],
            "process_infected": scan_result["process_infected"],
            "infection_rate": scan_result["infection_rate"],
            "infected_agents": scan_result["infected_agents"],
            "rebuttal_agents": scan_result["rebuttal_agents"],
            "true_infected_agents": scan_result["true_infected_agents"],
            "n_agents": scan_result["n_total"],
        }
        results.append(result)

    _save_results(results, output_dir)
    _generate_summary(results, output_dir)
    return results


def _save_results(results: list, output_dir: str) -> None:
    frameworks = sorted({r["framework"] for r in results})
    scenarios = sorted({r["scenario"] for r in results})
    strategies = sorted({r["strategy"] for r in results})

    results_path = os.path.join(output_dir, "mega_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "frameworks": frameworks,
                    "scenarios": scenarios,
                    "strategies": strategies,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Results saved to {results_path}")


def _generate_summary(results: list, output_dir: str) -> None:
    groups = {}
    for r in results:
        key = (r["framework"], r["scenario"], r["strategy"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    summary = []
    for (fw, sc, st), group in groups.items():
        final_asr = sum(1 for r in group if r["final_infected"]) / len(group)
        process_asr = sum(1 for r in group if r["process_infected"]) / len(group)
        avg_inf = sum(r["infection_rate"] for r in group) / len(group)
        summary.append({
            "framework": fw,
            "topology": get_topology(fw),
            "scenario": sc,
            "strategy": st,
            "final_asr": round(final_asr, 3),
            "process_asr": round(process_asr, 3),
            "avg_infection_rate": round(avg_inf, 3),
            "n_samples": len(group),
        })

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    csv_path = os.path.join(output_dir, "mega_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Topology,Framework,Scenario,Baseline,Compliance,Security_FUD\n")

        pivot = {}
        for s in summary:
            row_key = (s["topology"], s["framework"], s["scenario"])
            if row_key not in pivot:
                pivot[row_key] = {}
            pivot[row_key][s["strategy"]] = s["final_asr"]

        for (topo, fw, sc), strats in sorted(pivot.items()):
            baseline = strats.get("baseline", 0)
            compliance = strats.get("compliance", 0)
            security_fud = strats.get("security_fud", 0)
            f.write(f"{topo},{fw},{sc},{baseline:.1%},{compliance:.1%},{security_fud:.1%}\n")
    print(f"CSV saved to {csv_path}")


def main():
    log_path = "outputs/attack/mega_exp/logs.jsonl"
    output_dir = "outputs/attack/mega_exp"
    results = analyze_logs(log_path, output_dir)
    print(f"Analysis complete. Records: {len(results)}")


if __name__ == "__main__":
    main()
