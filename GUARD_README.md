# GUARD

GUARD is a proposed local component for tracking facts used during the
project's multi-agent propagation experiments and explaining a core
fact-checking decision with its supplied evidence. The dedicated GUARD core
modules are not present in this checkout, so core persistence, claim checking,
and the application adapter remain pending teammate implementation. The
contract below is the interface used by `explain.py` and its tests; it is not
a claim that the pending core behavior already exists.

## Proposed module responsibilities

- `prop_dynamics.guard.fact_store`: SQLite persistence for fact records. A
  proposed `facts` table has `id TEXT PRIMARY KEY`, `content TEXT NOT NULL`,
  `source TEXT NOT NULL`, and `created_at TEXT NOT NULL`. The module should
  expose `FactStore(db_path="guard_facts.db")` with `add_fact(content,
  source) -> dict`, `get_fact(fact_id) -> Optional[dict]`, `list_facts() ->
  List[dict]`, and `delete_fact(fact_id) -> bool`. Returned fact dictionaries
  should contain `id`, `content`, `source`, and `created_at`. The database
  should be opened on explicit store use, not at import time.
- `prop_dynamics.guard.main`: the core claim check. For the minimal first
  implementation, a normalized exact fact-text match may produce `supported`;
  no match should produce `unknown`, since absence is not evidence of
  falsity. Implement `check_claim(claim, store) -> dict`, returning a result
  mapping with `decision`, `reason`, and `facts`. Facts should be the records
  actually used as evidence. A claim without usable evidence must produce
  `unknown`; `unsupported` should remain unused until a concrete contradiction
  rule is established.
- `prop_dynamics.guard.explain`: formats that result deterministically. It
  does not re-check claims or create evidence.
- `prop_dynamics.guard.app`: optional thin HTTP adapter that calls the core
  API. No web framework or endpoint is established by this repository yet.
- `prop_dynamics.guard.__init__`: exports selected public symbols without
  creating a database or making network calls during import.

The exact-match behavior is a minimal proposal for the core owner to
implement or revise. It does not imply semantic verification, source
authentication, or a security guarantee. The core must reject empty or
non-string claims with `ValueError` or `TypeError` and must not perform
database work at import time.

## Explanation contract

`explain_result(result)` accepts a mapping shaped like:

```python
{
    "decision": "supported",  # supported, unsupported, or unknown
    "reason": "An exact fact was found.",
    "facts": [
        {"id": "fact-1", "content": "The sky appears blue.", "source": "observation"}
    ],
}
```

It returns a stable text explanation that repeats the decision and reason and
lists only facts supplied in the result. With an empty facts list, it states
that GUARD supplied no matching facts. Missing or malformed required fields
raise `TypeError` or `ValueError` instead of producing an ambiguous account.

## Integration status

`prop_dynamics/guard/explain.py` and `tests/test_guard.py` implement and test
the result explanation contract. The SQLite store, core decision function,
package exports, and HTTP adapter are absent from this checkout. Teammates
should implement the proposed names and signatures above. `check_claim` must
return `decision` (`supported`, `unsupported`, or `unknown`), a non-empty
`reason`, and `facts` as a sequence of mappings with non-empty `content`; each
fact may also provide string `id` and `source` fields. Because no contradiction
rule has been proposed, it should currently return `unsupported` only if the
core is extended with an explicit, evidence-backed rule; an empty store should
yield `unknown`.

No environment variables, API credentials, network access, or third-party
dependencies are needed by the explanation module. Database path defaults,
HTTP endpoints, and reset commands depend on the pending core implementation
and are intentionally undocumented here.

## Tests

Run the focused standard-library tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_guard.py"
```

These tests cover explanation output, empty evidence, decision normalization,
and malformed inputs. They do not verify SQLite persistence or the pending
core integration.
