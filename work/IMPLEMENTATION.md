# Running From-Spark-to-Fire on a free API (what was changed + how to run)

## What was wrong
All code calls the OLD `openai.ChatCompletion.create(...)` (openai 0.28 style) and 3 of the 4 parts hardcoded
`import openai` + an OpenAI key. The "frameworks" (CrewAI, AutoGen, LangGraph, MetaGPT, CAMEL, LangChain) are
*simulated with plain chat calls*, so you only need one OpenAI-compatible endpoint, no framework installs.

## What was patched
- `setup_openai()` in runner.py, attack/config.py, threat/config.py now routes through `openai_compat.configure_openai`
  (works with openai>=1.x AND any base_url: Groq / Gemini / Ollama / OpenRouter).
- Model names come from env `OPENAI_MODEL` (was hardcoded gpt-4o-mini in ~12 files).
- runner.py no longer silently logs empty answers when the API fails (`FAIL_FAST=1`).
- Added: `.env.example`, `check_api.py`, `run_small.sh`.

## Steps
pip install -r prop_dynamics/requirements.txt -r prop_dynamics/defense/requirements.txt "openai>=1.0"
cp .env.example .env   # fill provider block
set -a; source .env; set +a
python check_api.py
./run_small.sh
