"""Entry point:  uvicorn prop_dynamics.guard.main:app --port 8000

Reads settings from GUARD_ENV_FILE (default env_groq.txt) if the file exists.
Variables already set in the environment win over the file.
"""

import os
from pathlib import Path

from .app import create_app


def _load_env_file(path: str) -> None:
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_env_file = os.environ.get("GUARD_ENV_FILE", "env_groq.txt")
if Path(_env_file).exists():
    _load_env_file(_env_file)

app = create_app(db_path=os.environ.get("GUARD_DB", "guard_facts.db"))
