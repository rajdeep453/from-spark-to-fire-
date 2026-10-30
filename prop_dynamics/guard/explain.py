"""Deterministic explanations for structured GUARD results.

The core GUARD result contract is a mapping with ``decision`` and ``reason``
strings, and an optional ``facts`` sequence. Each fact is a mapping containing
``content`` and optional ``id`` and ``source`` strings. This module deliberately
does not decide whether a claim is supported; it reports the core's decision
and the evidence supplied with it.
"""

import math
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


def explain_claim(claim: str, graph: Any, classifier: Any, top_k: int = 3) -> dict:
    """Classify ``claim`` against the graph and explain the classifier output.

    ``classifier`` is the project's NLI classifier (or another object with the
    same ``classify(claim, graph, top_k=...)`` interface). The decision and
    evidence come only from its returned ``ClassificationOutput``. A claim
    with no returned evidence is reported as ``unknown`` even if a custom
    classifier returns a stronger label.
    """
    if not isinstance(claim, str):
        raise TypeError("claim must be a string")
    claim = claim.strip()
    if not claim:
        raise ValueError("claim must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not callable(getattr(classifier, "classify", None)):
        raise TypeError("classifier must provide classify(claim, graph, top_k=...)")

    classified = classifier.classify(claim, graph, top_k=top_k)
    label = getattr(getattr(classified, "result", None), "value", None)
    decisions = {
        "entailment": "supported",
        "green": "supported",
        "contradiction": "unsupported",
        "red": "unsupported",
        "neutral": "unknown",
        "yellow": "unknown",
    }
    if not isinstance(label, str) or label.lower() not in decisions:
        raise ValueError("classifier returned an unsupported decision")

    evidence = getattr(classified, "evidence", None)
    if evidence is None:
        evidence = []
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("classifier evidence must be a sequence of strings")
    facts = []
    for index, item in enumerate(evidence):
        if not isinstance(item, str):
            raise TypeError(f"classifier evidence[{index}] must be a string")
        content = item.strip()
        if content:
            facts.append({"content": content})

    evidence_ids = getattr(classified, "evidence_ids", None) or []
    if isinstance(evidence_ids, (str, bytes)) or not isinstance(evidence_ids, Sequence):
        raise TypeError("classifier evidence_ids must be a sequence of strings")
    if evidence_ids and len(evidence_ids) != len(facts):
        raise ValueError("classifier evidence_ids must align with its evidence strings")
    for index, fact_id in enumerate(evidence_ids):
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise ValueError(f"classifier evidence_ids[{index}] must be a non-empty string")
        facts[index]["id"] = fact_id.strip()

    decision = decisions[label.lower()] if facts else "unknown"
    reason = getattr(classified, "reason", "")
    if not isinstance(reason, str):
        raise TypeError("classifier reason must be a string")
    reason = reason.strip()
    if not reason:
        reason = {
            "supported": "The classifier found evidence that entails the claim.",
            "unsupported": "The classifier found evidence that contradicts the claim.",
            "unknown": "The available evidence did not resolve the claim.",
        }[decision]
    if not facts and decision == "unknown":
        reason = "No relevant trusted facts were supplied by the classifier."

    confidence = getattr(classified, "confidence", None)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("classifier confidence must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("classifier confidence must be finite and between 0 and 1")

    result = {"decision": decision, "reason": reason, "facts": facts}
    return {
        "claim": claim,
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "evidence": facts,
        "explanation": explain_result(result),
    }


__all__ = ["explain_claim", "explain_result"]
