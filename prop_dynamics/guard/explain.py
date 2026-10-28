"""Deterministic explanations for structured GUARD results.

The core GUARD result contract is a mapping with ``decision`` and ``reason``
strings, and an optional ``facts`` sequence. Each fact is a mapping containing
``content`` and optional ``id`` and ``source`` strings. This module deliberately
does not decide whether a claim is supported; it reports the core's decision
and the evidence supplied with it.
"""

from typing import Any, Mapping, Sequence


_DECISIONS = {"supported", "unsupported", "unknown"}


def explain_result(result: Mapping[str, Any]) -> str:
    """Render a GUARD result as deterministic, evidence-grounded text.

    ``result`` must contain a recognized ``decision`` and a string ``reason``.
    Its optional ``facts`` value must be a sequence of fact mappings with a
    string ``content`` field. Malformed results raise ``TypeError`` or
    ``ValueError`` rather than producing an explanation that could misstate
    the evidence.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    decision = result.get("decision")
    if not isinstance(decision, str):
        raise TypeError("result.decision must be a string")
    decision = decision.strip().lower()
    if decision not in _DECISIONS:
        raise ValueError("result.decision must be supported, unsupported, or unknown")

    reason = result.get("reason")
    if not isinstance(reason, str):
        raise TypeError("result.reason must be a string")
    reason = reason.strip()
    if not reason:
        raise ValueError("result.reason must not be empty")

    facts = result.get("facts", ())
    if isinstance(facts, (str, bytes)) or not isinstance(facts, Sequence):
        raise TypeError("result.facts must be a sequence of fact mappings")

    rendered_facts = []
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, Mapping):
            raise TypeError(f"result.facts[{index - 1}] must be a mapping")
        content = fact.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"result.facts[{index - 1}].content must be a non-empty string")

        details = []
        for key in ("id", "source"):
            value = fact.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"result.facts[{index - 1}].{key} must be a string")
                if value.strip():
                    details.append(f"{key}: {value.strip()}")
        suffix = f" ({'; '.join(details)})" if details else ""
        rendered_facts.append(f"{index}. {content.strip()}{suffix}")

    explanation = f"GUARD decision: {decision}. Reason: {reason}"
    if rendered_facts:
        explanation += " Evidence supplied by GUARD:\n" + "\n".join(rendered_facts)
    else:
        explanation += " GUARD supplied no matching facts."
    return explanation


__all__ = ["explain_result"]
