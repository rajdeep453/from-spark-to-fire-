#!/usr/bin/env bash
# Small, free-tier-safe versions of all four experiments. Run from repo root after sourcing .env
set -e
python check_api.py
echo "== 1/4 Model Utility (propagation dynamics) =="
python - <<'PY'
import sys; sys.path.insert(0,'.')
from prop_dynamics.run_v1_exp import run_v1_experiments
from prop_dynamics.runner import load_sample_questions
run_v1_experiments(questions=load_sample_questions(2), topologies=["centralized","layered","decentralized"],
                   N=5, T_rounds=4, R_repeats=2, output_dir="outputs/v1_topology")
PY
echo "== 2/4 Vulnerability validation =="
python - <<'PY'
import sys; sys.path.insert(0,'.')
from prop_dynamics.threat.run_exp1 import run_exp1
run_exp1(n_tasks=3, n_repeats=1, frameworks=["langchain","crewai","langgraph","autogen","metagpt","camel"])
PY
echo "== 3/4 Attack =="
python - <<'PY'
import sys; sys.path.insert(0,'.')
import prop_dynamics.attack.experiments.run_mega_exp as m
m.SCENARIOS=["quant","mmlu"]; m.STRATEGIES=["baseline","compliance","security_fud"]
m.run_mega_experiment(n_tasks=2, n_repeats=1, max_workers=2)
PY
python -m prop_dynamics.attack.experiments.mega_exp_analysis || true
echo "== 4/4 Defense =="
python - <<'PY'
import sys; sys.path.insert(0,'.')
from prop_dynamics.defense.experiments.run_defense_exp import run_defense_experiment
run_defense_experiment(n_tasks=2, n_repeats=1, max_workers=2, frameworks=["langgraph"], scenarios=["quant"],
    defense_strategies=["none","balanced","strict"], attack_strategies=["compliance"])
PY
