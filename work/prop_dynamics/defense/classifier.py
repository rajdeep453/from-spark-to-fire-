import os
"""Classifier."""

import json
from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass

from .genealogy_graph import GenealogyGraph


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


CLASSIFY_PROMPT = """You are a fact checker. Determine the relationship between the [Claim] and [Trusted Facts].

[Claim]: {atom}

[Trusted Facts]:
{trusted_facts}

Criteria:
- ENTAILMENT: Claim can be derived from trusted facts (paraphrase, summary, logical implication)
- NEUTRAL: Claim is new information, cannot be derived but does not contradict
- CONTRADICTION: Claim explicitly contradicts trusted facts

Output only one word: ENTAILMENT or NEUTRAL or CONTRADICTION"""


BATCH_CLASSIFY_PROMPT = """Classify each claim against trusted facts.

[Claims]:
{claims}

[Trusted Facts]:
{trusted_facts}

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
        
        if self._openai is None:
            best_sim = similar_facts[0][1]
            if best_sim > 0.9:
                return ClassificationOutput(ClassifyResult.GREEN, best_sim, evidence, "High similarity")
            return ClassificationOutput(ClassifyResult.YELLOW, 0.5, evidence, "No LLM")
        
        messages = [
            {"role": "system", "content": "You are a fact checker."},
            {"role": "user", "content": CLASSIFY_PROMPT.format(atom=atom, trusted_facts=trusted_facts_text)}
        ]
        result_text, tokens = call_llm(self._openai, self.model, messages, 0.1, 20)
        self.tokens_used += tokens
        result_text = result_text.upper()
        
        if "ENTAILMENT" in result_text:
            return ClassificationOutput(ClassifyResult.GREEN, 0.9, evidence, "Entailed")
        elif "CONTRADICTION" in result_text:
            return ClassificationOutput(ClassifyResult.RED, 0.9, evidence, "Contradicts")
        return ClassificationOutput(ClassifyResult.YELLOW, 0.7, evidence, "Neutral")
    
    def classify_batch(self, atoms: List[str], graph: GenealogyGraph, top_k: int = 3) -> List[ClassificationOutput]:
        if not atoms:
            return []
        
        all_facts = set()
        for atom in atoms:
            for fact, _ in graph.search(atom, top_k=top_k):
                all_facts.add(fact.content)
        
        if not all_facts:
            return [ClassificationOutput(ClassifyResult.YELLOW, 0.5, [], "No facts") for _ in atoms]
        
        trusted_text = "\n".join([f"- {f}" for f in list(all_facts)[:10]])
        claims_text = "\n".join([f"{i+1}. {a}" for i, a in enumerate(atoms)])
        evidence = list(all_facts)[:3]
        
        if self._openai is None:
            return [ClassificationOutput(ClassifyResult.YELLOW, 0.5, evidence, "No LLM") for _ in atoms]
        
        messages = [
            {"role": "system", "content": "Output valid JSON array only."},
            {"role": "user", "content": BATCH_CLASSIFY_PROMPT.format(claims=claims_text, trusted_facts=trusted_text)}
        ]
        result_text, tokens = call_llm(self._openai, self.model, messages, 0.1, len(atoms) * 15 + 20)
        self.tokens_used += tokens
        
        with open("run.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== BATCH DEBUG ===\nClaims ({len(atoms)}):\n{claims_text}\n\nTrusted Facts:\n{trusted_text}\n\nLLM Response:\n{result_text}\n===================\n")
        
        try:
            labels = json.loads(result_text)
        except json.JSONDecodeError as e:
            with open("run.log", "a", encoding="utf-8") as f:
                f.write(f"\n[ERROR] JSON Decode Failed: {e}\nRaw Response: {result_text}\n")
            return [ClassificationOutput(ClassifyResult.YELLOW, 0.5, evidence, "JSON Parse Error") for _ in atoms]
            
        outputs = []
        for i, atom in enumerate(atoms):
            label = labels[i].upper() if i < len(labels) else "NEUTRAL"
            if "ENTAILMENT" in label:
                outputs.append(ClassificationOutput(ClassifyResult.GREEN, 0.9, evidence, "Entailed"))
            elif "CONTRADICTION" in label:
                outputs.append(ClassificationOutput(ClassifyResult.RED, 0.9, evidence, "Contradicts"))
            else:
                outputs.append(ClassificationOutput(ClassifyResult.YELLOW, 0.7, evidence, "Neutral"))
        return outputs
