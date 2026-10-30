import os
"""Classifier."""

import json
import logging
from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from .genealogy_graph import GenealogyGraph

logger = logging.getLogger(__name__)


class ClassifyResult(Enum):
    GREEN = "entailment"
    YELLOW = "neutral"
    RED = "contradiction"


@dataclass
class ClassificationOutput:
    result: ClassifyResult
    confidence: float
    evidence: List[str]
    reason: str
    evidence_ids: List[str] = field(default_factory=list)


CLASSIFY_PROMPT = """You are a fact checker. Determine the relationship between the [Claim] and [Trusted Facts].

[Claim]: {atom}

[Trusted Facts]:
{trusted_facts}

Criteria:
- ENTAILMENT: Claim can be derived from trusted facts (paraphrase, summary, logical implication)
- NEUTRAL: Claim is new information, cannot be derived but does not contradict
- CONTRADICTION: Claim explicitly contradicts trusted facts

Output only one word: ENTAILMENT or NEUTRAL or CONTRADICTION"""


BATCH_CLASSIFY_PROMPT = """Classify each claim using only the trusted facts listed for that claim.

[Claim-specific evidence]:
{claims}

For each claim: ENTAILMENT (matches), NEUTRAL (new), or CONTRADICTION (conflicts).

Output ONLY a JSON array with one label per claim, nothing else: ["ENTAILMENT", "NEUTRAL", ...]"""


def call_llm(openai_module, model: str, messages: list, 
             temperature: float = 0.1, max_tokens: int = 50):
    try:
        resp = openai_module.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        tokens = resp.get("usage", {}).get("total_tokens", 0)
        return resp["choices"][0]["message"]["content"].strip(), tokens
    except Exception:
        print("[Classifier] Request failed.")
        raise


class Classifier:
    def __init__(self, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"), openai_module=None):
        self.model = model
        self._openai = openai_module
        self.tokens_used = 0
    
    def set_openai(self, openai_module):
        self._openai = openai_module
    
    def reset_tokens(self):
        self.tokens_used = 0
    
    def classify(self, atom: str, graph: GenealogyGraph, top_k: int = 5) -> ClassificationOutput:
        similar_facts = graph.search(atom, top_k=top_k)
        if not similar_facts:
            return ClassificationOutput(ClassifyResult.YELLOW, 0.5, [], "No trusted facts")
        
        trusted_facts_text = "\n".join([f"- {f.content}" for f, _ in similar_facts])
        evidence = [f.content for f, _ in similar_facts]
        evidence_ids = [f.id for f, _ in similar_facts]
        
        if self._openai is None:
            best_sim = similar_facts[0][1]
            if best_sim > 0.9:
                return ClassificationOutput(ClassifyResult.GREEN, best_sim, evidence[:1], "High similarity", evidence_ids[:1])
            return ClassificationOutput(ClassifyResult.YELLOW, 0.5, evidence[:1], "No LLM", evidence_ids[:1])
        
        messages = [
            {"role": "system", "content": "You are a fact checker."},
            {"role": "user", "content": CLASSIFY_PROMPT.format(atom=atom, trusted_facts=trusted_facts_text)}
        ]
        result_text, tokens = call_llm(self._openai, self.model, messages, 0.1, 20)
        self.tokens_used += tokens
        label = result_text.strip().upper()

        if label == "ENTAILMENT":
            return ClassificationOutput(ClassifyResult.GREEN, 0.9, evidence, "Entailed", evidence_ids)
        elif label == "CONTRADICTION":
            return ClassificationOutput(ClassifyResult.RED, 0.9, evidence, "Contradicts", evidence_ids)
        return ClassificationOutput(ClassifyResult.YELLOW, 0.7, evidence, "Neutral", evidence_ids)
    
    def classify_batch(self, atoms: List[str], graph: GenealogyGraph, top_k: int = 3) -> List[ClassificationOutput]:
        if not atoms:
            return []
        
        claim_facts = []
        for atom in atoms:
            # Keep the candidate evidence tied to this claim. A shared union of
            # facts can make one claim inherit another claim's lineage.
            seen = set()
            facts = []
            for fact, _ in graph.search(atom, top_k=top_k):
                if fact.id not in seen:
                    facts.append(fact)
                    seen.add(fact.id)
            claim_facts.append(facts)

        outputs = [
            ClassificationOutput(ClassifyResult.YELLOW, 0.5, [], "No trusted facts")
            if not facts else None
            for facts in claim_facts
        ]
        valid_indices = [i for i, facts in enumerate(claim_facts) if facts]
        if not valid_indices:
            return outputs
        if self._openai is None:
            for i in valid_indices:
                facts = claim_facts[i]
                outputs[i] = ClassificationOutput(
                    ClassifyResult.YELLOW, 0.5,
                    [fact.content for fact in facts], "No LLM",
                    [fact.id for fact in facts],
                )
            return outputs

        claim_blocks = []
        for i in valid_indices:
            fact_lines = "\n".join(
                f"- [{fact.id}] {fact.content}" for fact in claim_facts[i]
            )
            claim_blocks.append(
                f"Claim {i + 1}: {atoms[i]}\n"
                f"Trusted facts for Claim {i + 1} (consider only these facts):\n{fact_lines}"
            )
        claims_and_facts = "\n\n".join(claim_blocks)
        
        messages = [
            {"role": "system", "content": "Output valid JSON array only."},
            {"role": "user", "content": BATCH_CLASSIFY_PROMPT.format(claims=claims_and_facts)}
        ]
        result_text, tokens = call_llm(
            self._openai, self.model, messages, 0.1, len(valid_indices) * 15 + 20
        )
        self.tokens_used += tokens

        try:
            labels = json.loads(result_text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Batch classifier returned invalid JSON; treating claims as neutral")
            labels = None

        if not isinstance(labels, list) or len(labels) != len(valid_indices):
            logger.warning("Batch classifier returned an invalid label list; treating claims as neutral")
            labels = [None] * len(valid_indices)

        label_results = {
            "ENTAILMENT": (ClassifyResult.GREEN, "Entailed"),
            "CONTRADICTION": (ClassifyResult.RED, "Contradicts"),
            "NEUTRAL": (ClassifyResult.YELLOW, "Neutral"),
        }
        for i, raw_label in zip(valid_indices, labels):
            label = raw_label.strip().upper() if isinstance(raw_label, str) else ""
            result, reason = label_results.get(
                label, (ClassifyResult.YELLOW, "Invalid classifier label")
            )
            facts = claim_facts[i]
            outputs[i] = ClassificationOutput(
                result, 0.9 if result != ClassifyResult.YELLOW else 0.7,
                [fact.content for fact in facts], reason,
                [fact.id for fact in facts],
            )
        return outputs
