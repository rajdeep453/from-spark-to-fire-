"""
Interactive Live Demonstration Script for "From Spark to Fire"
Usage:
    python demo.py
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load_env(env_file="env_groq.txt"):
    if os.path.exists(env_file):
        for line in open(env_file, encoding="utf-8-sig"):
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()
        print(f"[OK] Loaded environment from {env_file} (Model: {os.getenv('OPENAI_MODEL')})")
    else:
        print(f"[!] Warning: {env_file} not found. Relying on current system environment.")

def banner():
    print("=" * 80)
    print("      FROM SPARK TO FIRE: MULTI-AGENT COLLABORATION DEMONSTRATION")
    print("      Modeling & Mitigating Error Cascades in LLM-MAS (arXiv:2603.04474)")
    print("=" * 80)

def demo_audit_existing():
    print("\n--- DEMO 1: Instant Forensic Analysis & Rebuttal Verification ---")
    print("Reading recorded multi-agent traces from outputs/attack/mega_exp/logs.jsonl...")
    time.sleep(0.5)

    log_path = os.path.join(PROJECT_ROOT, "outputs", "attack", "mega_exp", "logs.jsonl")
    if not os.path.exists(log_path):
        print(f"[!] Log file not found at {log_path}. Please run an experiment first.")
        return

    from prop_dynamics.attack.detector import InfectionDetector
    from prop_dynamics.attack.experiments.mega_exp_analysis import _build_execution_log

    with open(log_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            fw = data.get("framework")
            strat = data.get("strategy")
            task_id = data.get("task_id")
            
            print(f"\n[Case {idx+1}] Framework: {fw.upper():9s} | Topology: {data.get('topology_type'):5s} | Strategy: {strat.upper():12s} ({task_id})")
            
            # Show the agents involved
            agents = [m.get("sender") for m in data.get("messages", [])]
            print(f"       Agent execution trace: {' -> '.join(agents)}")
            
            detector = InfectionDetector("malware-math.io", "rigid")
            exec_log = _build_execution_log(data)
            scan = detector.scan_log(exec_log)
            
            if scan["final_infected"]:
                status = "[INFECTED]   (Attack Succeeded - Consensual Error)"
            else:
                status = "[DEFENDED]   (Attack Blocked / Rebutted by Agent)"
                
            print(f"       Verdict:         {status}")
            print(f"       Detection Note:  {scan['final_reason']}")
            if scan["rebuttal_agents"]:
                print(f"       Rebuttal Agents: {scan['rebuttal_agents']} (Sanitized the code/answer)")
            print(f"       Infected Agents: {scan['true_infected_agents']}")
            print(f"       Infection Rate:  {scan['infection_rate']:.1%}")

    print("\n" + "=" * 80)
    print("       CURRENT SUMMARY TABLE (mega_table.csv)")
    print("=" * 80)
    csv_path = os.path.join(PROJECT_ROOT, "outputs", "attack", "mega_exp", "mega_table.csv")
    if os.path.exists(csv_path):
        print(open(csv_path, "r", encoding="utf-8").read().strip())
    print("=" * 80)

def demo_live_agent_run():
    print("\n--- DEMO 2: Live Multi-Agent Execution Run ---")
    print("This will execute a live 4-agent LangChain task with injected attack payload.")
    load_env("env_groq.txt")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[!] Error: OPENAI_API_KEY is not set. Check env_groq.txt.")
        return

    from prop_dynamics.attack.config import AttackConfig
    from prop_dynamics.attack.tasks import load_tasks
    from prop_dynamics.attack.injector import PayloadGenerator
    from prop_dynamics.attack.frameworks import create_adapter
    from prop_dynamics.attack.detector import create_detector

    print("\n[1/4] Loading Task: rigid_001 (Math reasoning: 2^2005 mod 7)")
    tasks = load_tasks("rigid", 1)
    task = tasks[0]
    print(f"      Description: {task.description}")
    print(f"      Official API: {task.resource_url}")

    print("\n[2/4] Generating Injected Adversarial Payload (Strategy: Security_FUD)")
    config = AttackConfig(framework="langchain", scenario="rigid", n_tasks=1, n_repeats=1)
    adapter = create_adapter("langchain", config)
    target_agent = adapter.get_entry_agent()
    
    payload_gen = PayloadGenerator()
    injection = payload_gen.generate(task, target_agent, "security_fud")
    print(f"      Target Agent: {target_agent}")
    print(f"      Injected Payload: {injection.content.strip()[:150]}...")

    print("\n[3/4] Running Live Multi-Agent Chain (PM -> Architect -> Engineer -> Reviewer)...")
    print("      Calling LLM endpoints via Groq API. Please wait ~15-30s...")
    t0 = time.time()
    log = adapter.run_task(task, injection)
    elapsed = time.time() - t0
    print(f"      [✓] Multi-agent execution completed in {elapsed:.1f} seconds!")

    print("\n[4/4] Multi-Agent Conversation Inspection:")
    for msg in log.messages:
        print(f"\n--- [Agent: {msg.sender}] (Turn {msg.round}) ---")
        preview = msg.content.strip().split("\n")
        print("\n".join(preview[:8]))
        if len(preview) > 8:
            print("... (truncated for display)")

    detector = create_detector(injection)
    scan = detector.scan_log(log)
    print("\n" + "=" * 80)
    print("       LIVE EVALUATION RESULT")
    print("=" * 80)
    print(f"Final Output Infected: {scan['final_infected']} (Reason: {scan['final_reason']})")
    print(f"Infected Agents:       {scan['true_infected_agents']}")
    print(f"Rebuttal Agents:       {scan['rebuttal_agents']}")
    print(f"Infection Rate:        {scan['infection_rate']:.1%}")
    print("=" * 80)

def main():
    banner()
    print("\nSelect a demonstration mode:")
    print("  [1] Instant Forensic Analysis Demo (Fast, 3 seconds - Shows Rebuttal Detection & Results)")
    print("  [2] Live Multi-Agent Execution Demo (Calls API, ~30 seconds - Live Agent Chat & Attack)")
    print("  [3] Re-generate HTML & PPTX Slide Decks")
    print("  [4] Exit")
    
    choice = input("\nEnter choice [1-4] (default: 1): ").strip() or "1"
    
    if choice == "1":
        demo_audit_existing()
    elif choice == "2":
        demo_live_agent_run()
    elif choice == "3":
        print("\nRe-generating presentation assets...")
        os.system(f'"{sys.executable}" scripts/create_presentation.py presentation.pptx')
        print("[✓] presentation.pptx updated.")
        print("[✓] presentation.html is ready.")
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
