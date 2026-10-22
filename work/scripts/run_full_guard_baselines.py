

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_env_file(path: Path) -> set[str]:
    loaded_keys: set[str] = set()
    if not path.exists():
        return loaded_keys
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip().lstrip("\ufeff")
        os.environ[clean_key] = value.strip()
        loaded_keys.add(clean_key)
    return loaded_keys


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def preflight_openai_json(openai_module, model: str) -> int:
    response = openai_module.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": 'Return {"ok": true}.'},
        ],
        temperature=0,
        max_tokens=20,
    )
    content = response["choices"][0]["message"]["content"].strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        content = match.group(1).strip()
    usage = response.get("usage", {}) or {}
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    parsed = json.loads(content)
    if parsed.get("ok") is not True or total_tokens <= 0:
        raise RuntimeError(
            f"OpenAI preflight returned invalid JSON or zero token usage for model {model}."
        )
    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-tasks", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--frameworks", default="metagpt,langgraph,autogen")
    parser.add_argument("--scenarios", default="quant,rigid,mmlu")
    parser.add_argument("--defenses", default="agrail_monitor,control_flow_guard")
    parser.add_argument("--attacks", default="baseline,compliance,security_fud")
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    loaded_keys = load_env_file(Path(args.env_file))
    if "OPENAI_BASE_URL" not in loaded_keys and "OPENAI_API_BASE" not in loaded_keys:
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_BASE", None)
    os.environ.setdefault("OPENAI_MODEL", "gpt-4o-mini")
    os.environ.setdefault("OPENAI_TIMEOUT", "60")
    os.environ.setdefault("OPENAI_MAX_RETRIES", "0")
    os.environ.setdefault("GUARD_FAIL_FAST", "1")
    os.environ.setdefault("FRAMEWORK_FAIL_FAST", "1")
    os.environ.setdefault("EXPERIMENT_FAIL_FAST", "1")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in the env file or process env.")

    from prop_dynamics.defense.experiments.run_defense_exp import (
        compute_summary,
        run_defense_experiment,
    )
    from prop_dynamics.defense.config import DefenseConfig, setup_openai

    config = DefenseConfig()
    expected_model = os.environ["OPENAI_MODEL"]
    if config.model != expected_model:
        raise RuntimeError(
            f"DefenseConfig model mismatch: config.model={config.model}, OPENAI_MODEL={expected_model}"
        )
    preflight_tokens = preflight_openai_json(setup_openai(config), config.model)
    print(f"PREFLIGHT_OK model={config.model} tokens={preflight_tokens}")
    if args.preflight_only:
        return

    output_dir = args.output_dir
    if not output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/defense/full_4omini_guard_baselines_{ts}"

    results, logs = run_defense_experiment(
        n_tasks=args.n_tasks,
        n_repeats=args.n_repeats,
        output_dir=output_dir,
        max_workers=args.max_workers,
        frameworks=parse_csv(args.frameworks),
        scenarios=parse_csv(args.scenarios),
        defense_strategies=parse_csv(args.defenses),
        attack_strategies=parse_csv(args.attacks),
        limit_runs=args.limit_runs or None,
    )

    summary = compute_summary(results)
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print("OUTPUT_DIR=" + output_dir)
    print("RESULTS=" + str(Path(output_dir) / "defense_results.jsonl"))
    print("LOGS=" + str(Path(output_dir) / "logs.jsonl"))
    print("SUMMARY=" + str(summary_path))
    print("N_RESULTS=" + str(len(results)))
    print("N_LOGS=" + str(len(logs)))
    print(json.dumps(summary.get("overall", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
