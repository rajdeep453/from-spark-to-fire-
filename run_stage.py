"""
Cross-platform stage runner (Windows / Linux / Mac).  Usage:
    python run_stage.py p1        # Model Utility   (~675 calls)
    python run_stage.py threat    # Vulnerability   (~380 calls)
    python run_stage.py attack    # Attack          (~1512 calls, one scenario at a time, partial results survive)
    python run_stage.py defense   # Defense         (~1200 calls, resumable)
    python run_stage.py analyze   # builds attack summary tables
Add  --tiny  for a ~20x smaller smoke test.
Pass --env env_groq.txt (or env_local.txt) or set the variables yourself. Reads OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL from the environment (see env_groq.txt / env_local.txt).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["p1", "threat", "attack", "defense", "analyze"])
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--env", default=None, help="env file to load, e.g. env_groq.txt or env_local.txt")
ap.add_argument("--workers", type=int, default=int(os.getenv("OPENAI_CONCURRENCY_LIMIT", "1")))
ap.add_argument("--scenarios", default="rigid,quant,mmlu", help="attack/defense scenarios, comma separated")
a = ap.parse_args()
if a.env:
    for line in open(a.env, encoding="utf-8-sig"):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); os.environ[k.strip()] = v.strip()
T = a.tiny
t0 = time.time()
print(f"model={os.getenv('OPENAI_MODEL')} base={os.getenv('OPENAI_BASE_URL')} workers={a.workers} tiny={T}")

if a.stage == "p1":
    from prop_dynamics.run_v1_exp import run_v1_experiments
    from prop_dynamics.runner import load_sample_questions
    run_v1_experiments(questions=load_sample_questions(1 if T else 3),
                       topologies=["centralized", "layered", "decentralized"],
                       N=3 if T else 5, T_rounds=2 if T else 5, R_repeats=1 if T else 3,
                       output_dir="outputs/v1_topology")

elif a.stage == "threat":
    from prop_dynamics.threat.run_exp1 import run_exp1
    from prop_dynamics.threat.run_exp2 import run_exp2
    fws = ["camel", "crewai", "langgraph", "langchain", "metagpt", "autogen"]
    run_exp1(n_tasks=1 if T else 5, n_repeats=1, frameworks=fws[:2] if T else fws)
    run_exp2(n_tasks=1 if T else 5, n_repeats=1)

elif a.stage == "attack":
    import prop_dynamics.attack.experiments.run_mega_exp as m
    base = "outputs/attack/mega_exp"
    m.STRATEGIES = ["baseline", "compliance", "security_fud"]
    if T: m.FRAMEWORKS_LIST = ["langchain", "crewai"]
    for sc in a.scenarios.split(","):
        out = os.path.join(base, sc)
        if os.path.exists(os.path.join(out, "logs.jsonl")):
            print(f"[skip] {sc} already done ({out})"); continue
        m.SCENARIOS = [sc]
        m.run_mega_experiment(n_tasks=1 if T else 3, n_repeats=1, output_dir=out, max_workers=a.workers)
    # merge per-scenario logs
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "logs.jsonl"), "w", encoding="utf-8") as fo:
        for sc in ["rigid", "quant", "mmlu"]:
            p = os.path.join(base, sc, "logs.jsonl")
            if os.path.exists(p): fo.write(open(p, encoding="utf-8").read())
    from prop_dynamics.attack.experiments.mega_exp_analysis import analyze_logs
    analyze_logs(os.path.join(base, "logs.jsonl"), base)

elif a.stage == "analyze":
    from prop_dynamics.attack.experiments.mega_exp_analysis import analyze_logs
    base = "outputs/attack/mega_exp"
    analyze_logs(os.path.join(base, "logs.jsonl"), base)

elif a.stage == "defense":
    from prop_dynamics.defense.experiments.run_defense_exp import run_defense_experiment
    run_defense_experiment(n_tasks=1 if T else 3, n_repeats=1, max_workers=a.workers,
        frameworks=["langgraph"] if T else ["langgraph", "autogen", "metagpt"],
        scenarios=["quant"] if T else ["quant", "mmlu"],
        defense_strategies=["none", "balanced"] if T else ["none", "balanced", "strict"],
        attack_strategies=["compliance"], output_dir="outputs/defense/exp")

print(f"done in {(time.time()-t0)/60:.1f} min")
