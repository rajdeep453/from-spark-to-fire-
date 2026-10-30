# GUARD message screening API

GUARD is the project's HTTP interface to its existing defense middleware. It
stores operator-designated trusted facts in SQLite, mirrors them into the
defense `GenealogyGraph`, classifies individual claims against verified graph
facts, and can screen an inter-agent message through the existing four-stage
defense pipeline.

GUARD is an experimental research component. Its NLI and language-model
decisions can be wrong; a `supported` result is a model judgment against the
configured evidence, not a guarantee of factual truth or a complete security
boundary. The API must be deployed behind TLS and appropriate network access
controls if it is exposed beyond the local machine.

## Architecture

- `fact_store.py` stores operator-entered facts and provenance labels in
  SQLite. On startup it rebuilds the graph's confirmed facts from the database.
  Claims seen during message screening are held in the in-memory graph only;
  they are not persisted as trusted facts. Green claims are recorded as
  verified; yellow claims are recorded as unverified and excluded from trusted
  fact retrieval.
- `explain.py` delegates the claim decision to the project's NLI classifier
  and formats the classifier's actual decision, confidence, reason, and
  returned evidence. It does not invent supporting facts.
- `app.py` exposes fact management, claim explanation, health, and message
  screening endpoints. Requests that can read or change facts or invoke model
  inference require a bearer token.
- `main.py` is the Uvicorn entry point. Importing it constructs the FastAPI
  application but does not create the database or load models; those resources
  initialize during application startup.
- `defense/` supplies the graph, NLI classifier, decomposer, verifier, and
  message middleware reused by GUARD.

SQLite facts have `id`, `content`, `source`, and `created_at` fields. Existing
databases created by the initial GUARD version are migrated to add `source`
with a default value of `manual`. Fact content is whitespace-normalized,
unique, and limited to 8,000 characters. Source is an operator-supplied label;
it is not independently authenticated by GUARD.

The graph's normal operation uses the configured sentence embedding and NLI
models. The graph class has a stable local feature-hashing fallback for
repeatable offline unit tests, but the live API requires sentence-transformers
and its NLI weights. That lexical fallback is not semantically equivalent to
the sentence model and should not be used to claim research-equivalent results.

## Install and start

Use Python 3.10 or newer. Install the GUARD API, its message-defense model
runtime, and test dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-guard.txt
```

The first service startup downloads and loads the sentence embedding and NLI models.
The host needs access to the model registry unless the models are already
cached. An OpenAI-compatible API key is only needed when the selected message
screening path invokes an LLM; local NLI screening and fact management do not
require network inference. The shipped API does not configure an external
web-search provider; message verification uses the trusted graph and can
optionally ask the configured OpenAI-compatible model to adjudicate that graph
evidence.

Create a strong API key and configure it in the current PowerShell session:

```powershell
$env:GUARD_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:GUARD_DB = "guard_facts.db"
```

Keep the generated key in a secret manager for persistent deployments; do not
commit it. To enable OpenAI-compatible calls, also set `OPENAI_API_KEY` and, if
needed, `OPENAI_BASE_URL` and `OPENAI_MODEL`. `GUARD_ENV_FILE` may point to a
local `KEY=value` settings file; values already present in the environment take
precedence. Environment files containing credentials must remain untracked.

Start the local service (Uvicorn binds to loopback by default):

```powershell
python -m uvicorn prop_dynamics.guard.main:app --host 127.0.0.1 --port 8000
```

Configure message screening with `GUARD_STRATEGY=speed|balanced|strict`
(`balanced` by default) and optional comma-separated `GUARD_HUB_AGENTS`. The
project middleware uses these values to route uncertain claims for verification.
Only expose the service outside loopback behind TLS, authentication, firewall
rules, and operational monitoring.

## HTTP API

All endpoints except `/health` require:

```http
Authorization: Bearer <GUARD_API_KEY>
```

Bodies are limited to 4 MiB. Pydantic validation rejects extra fields, empty
strings, out-of-range pagination, and oversized inputs with HTTP 422. Missing
or invalid credentials return 401; protected requests with no configured
`GUARD_API_KEY` return 503.
Fact batches contain at most 100 facts (8,000 characters each); source labels
are limited to 500 characters. Claims are limited to 2,000 characters with
`top_k` from 1 to 10. Messages are limited to 8,000 characters and sender and
receiver names to 200 characters each.

### Health

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Returns service status only. It is a liveness response; it does not expose
trusted-fact counts or certify that external model services are healthy.

### Add and list trusted facts

```powershell
$headers = @{ Authorization = "Bearer $env:GUARD_API_KEY" }
$body = @{ facts = @("The release date is 2026-10-30."); source = "project brief" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/facts -Headers $headers -ContentType "application/json" -Body $body
Invoke-RestMethod "http://127.0.0.1:8000/facts?limit=100&offset=0" -Headers $headers
```

Duplicate normalized content is not inserted a second time. The response
separates newly added facts from duplicate entries and reports the total count.
Listing supports `limit` (1–1000) and non-negative `offset`.

Delete a fact by its returned integer ID:

```powershell
Invoke-RestMethod -Method Delete -Uri http://127.0.0.1:8000/facts/1 -Headers $headers
```

Deletion updates both SQLite and the in-memory graph. Missing IDs return 404.

### Explain a claim

```powershell
$body = @{ claim = "The release date is 2026-10-30."; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/explain -Headers $headers -ContentType "application/json" -Body $body
```

The response includes `claim`, `decision` (`supported`, `unsupported`, or
`unknown`), model `confidence`, classifier `reason`, the evidence strings
returned by the NLI classifier, and a human-readable `explanation`. `unknown`
means the available evidence did not resolve the claim. Confidence is the
classifier's score, not a calibrated probability.

### Screen an inter-agent message

```powershell
$body = @{ sender = "Analyst"; receiver = "Reviewer"; message = "The release date is 2026-10-30." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/check -Headers $headers -ContentType "application/json" -Body $body
```

The response contains `can_send`, `final_message`, rollback `feedback`, and
per-claim screening records. `final_message` is `null` while a red claim is
blocked for correction. This calls the repository's `DefenseMiddleware`;
it is not an independent implementation of the paper's complete protocol.
The experimental middleware has known limitations, including approximate
claim decomposition and model-dependent classification. Review the returned
tags and decisions in safety-sensitive workflows.

## Database reset and tests

Stop the service before resetting the database. To reset, delete the configured
SQLite file and any `-wal`/`-shm` sidecars, then restart the service. Database
files are ignored by Git.

Run tests from the repository root:

```powershell
python -m pytest tests -q
```

Tests use temporary SQLite databases and injected local classifiers/guards;
they do not make model API calls. The default live service still requires the
model dependencies and cached or downloadable model weights.

## Reproducibility and limits

The project uses a research pipeline with model-driven components. Results can
vary with model weights, dependency versions, provider, prompts, decoding
settings, and hardware. Save the database seed, environment settings (without
secrets), model identifiers, task data, and run outputs when reporting results.
For paper-aligned comparisons, use the exact datasets, model versions,
framework protocols, random seeds, and metrics specified by the paper; the
standalone GUARD API by itself does not reproduce those experiments.
