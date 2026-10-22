# RUNBOOK: two terminals, same model family (Llama 3.1 8B) on Groq + local Ollama

Setup once:  pip install -r prop_dynamics/requirements.txt -r prop_dynamics/attack/requirements.txt -r prop_dynamics/threat/requirements.txt -r prop_dynamics/defense/requirements.txt "openai>=1.0"
Groq key  -> paste into env_groq.txt.   Ollama: `ollama pull llama3.1:8b`.
Check:    python check_api.py --env env_groq.txt     and     python check_api.py --env env_local.txt
Smoke:    python run_stage.py p1 --tiny --env env_local.txt      (30 seconds)

TERMINAL A (Groq)   : python run_stage.py attack  --env env_groq.txt   (~1,512 calls, 3 scenarios, ~504 each, saved per scenario)
TERMINAL B (Ollama) : python run_stage.py p1      --env env_local.txt  (~675 calls)
                      python run_stage.py threat  --env env_local.txt  (~380 calls)
                      python run_stage.py defense --env env_local.txt  (~1,200 calls est., resumable)
Results: outputs/  (v1_topology, threat, attack/mega_exp/summary.json + mega_table.csv, defense/exp)
