# From Spark to Fire

Research code for studying how factual and faithfulness errors spread through
LLM multi-agent workflows, and how message-layer screening can limit that
spread. The project is organized into propagation dynamics, vulnerability
experiments, attacks, and defenses. The research reference is [From Spark to
Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent
Collaboration](https://arxiv.org/abs/2603.04474).

This repository is an experimental reproduction, not a production multi-agent
platform. Model-based decisions are fallible and results depend on model
weights, provider, prompts, task data, decoding settings, dependency versions,
and hardware. Follow the paper's exact task protocol and model setup for
paper-aligned results.

## Setup

Use Python 3.10 or newer. To install dependencies for all experiment stages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r prop_dynamics/requirements.txt -r prop_dynamics/threat/requirements.txt -r prop_dynamics/attack/requirements.txt -r prop_dynamics/defense/requirements.txt
```

Set provider configuration using environment variables (see `.env.example`),
then verify the configured OpenAI-compatible endpoint:

```powershell
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_BASE_URL = "<optional-provider-base-url>"
$env:OPENAI_MODEL = "<provider-model-name>"
python check_api.py
```

Keep credentials out of source control. The experiment stages make external
model calls and may incur provider charges.

## Run research stages

The root `run_stage.py` is the cross-platform runner. `--tiny` runs smaller
smoke experiments; standard runs make many more model calls.

```powershell
python run_stage.py p1 --tiny
python run_stage.py threat --tiny
python run_stage.py attack --tiny
python run_stage.py defense --tiny
```

Stage-specific instructions and the legacy scripts are in
[Propagation Dynamics](prop_dynamics/README.md), [Threat
Experiments](prop_dynamics/threat/README.md), [Attacks](prop_dynamics/attack/README.md),
and [Defenses](prop_dynamics/defense/README.md). Results are written under
`outputs/`.

## GUARD API

GUARD exposes the existing defense middleware as a local authenticated API. It
stores operator-designated facts in SQLite and offers fact management, claim
explanations, and message screening. Setup, bearer-key configuration, endpoint
examples, limits, and known caveats are in [GUARD_README.md](GUARD_README.md).

```powershell
python -m pip install -r requirements-guard.txt
$env:GUARD_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
python -m uvicorn prop_dynamics.guard.main:app --host 127.0.0.1 --port 8000
```

## Tests

From the repository root:

```powershell
python -m pytest tests -q
```

The GUARD tests use temporary databases and injected local classifiers; they do
not require credentials or live model API calls. The running service needs its
pretrained local model weights, which are downloaded on startup if not cached.
