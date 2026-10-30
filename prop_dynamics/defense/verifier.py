import os
import logging
"""Verifier."""

from enum import Enum
from typing import List, Optional, Tuple, Callable, TYPE_CHECKING
from dataclasses import dataclass

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .genealogy_graph import GenealogyGraph


class VerifyResult(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFO = "not_enough_info"
    NO_EVIDENCE = "no_evidence"


@dataclass
class VerificationOutput:
    result: VerifyResult
    evidence: List[str]
    source_urls: List[str]
    confidence: float
    reason: str


VERIFY_PROMPT = """You are a fact checker.

[Claim]: {claim}

[Evidence]:
{evidence}

Based on the evidence, determine if the claim is supported:
- SUPPORTED: Evidence clearly supports or is consistent with the claim
- CONTRADICTED: Evidence contradicts the claim (e.g., resource mismatch)
- NOT_ENOUGH_INFO: Evidence is insufficient to verify the claim

Output ONLY the label (SUPPORTED, CONTRADICTED, or NOT_ENOUGH_INFO), nothing else."""


def call_llm(openai_module, model: str, messages: list, 
             temperature: float = 0.1, max_tokens: int = 50):
    import time
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries + 1):
        try:
            resp = openai_module.ChatCompletion.create(
                model=model, messages=messages, 
                temperature=temperature, max_tokens=max_tokens
            )
            tokens = resp.get("usage", {}).get("total_tokens", 0)
            return resp["choices"][0]["message"]["content"].strip(), tokens
        except Exception:
            if attempt == max_retries:
                raise RuntimeError("LLM request failed.")
            
            wait_time = base_delay * (2 ** attempt)
            print(f"[Verifier] Temporary issue. Retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)


class ExternalVerifier:
    def __init__(self, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"), search_top_k: int = 5, openai_module=None, enable_cache: bool = True):
        self.model = model
        self.search_top_k = search_top_k
        self._openai = openai_module
        self._knowledge_base: Optional['GenealogyGraph'] = None
        self._external_search_func: Optional[Callable] = None
        self.tokens_used = 0
        
        self.enable_cache = enable_cache
        self._cache = {}
        self._cache_size = 1000

    def set_openai(self, openai_module):
        self._openai = openai_module
    
    def reset_tokens(self):
        self.tokens_used = 0
        
    def _get_from_cache(self, claim: str) -> Optional[VerificationOutput]:
        if not self.enable_cache:
            return None
        return self._cache.get(self._cache_key(claim))
    
    def _add_to_cache(self, claim: str, result: VerificationOutput):
        if not self.enable_cache:
            return
        
        if len(self._cache) >= self._cache_size:
            first_key = next(iter(self._cache))
            self._cache.pop(first_key)
            
        self._cache[self._cache_key(claim)] = result
    
    def set_knowledge_base(self, graph: 'GenealogyGraph'):
        self._knowledge_base = graph
        self._cache.clear()
    
    def set_external_search_func(self, func: Callable):
        self._external_search_func = func
        self._cache.clear()

    def _cache_key(self, claim: str):
        graph = self._knowledge_base
        return (claim, id(graph), getattr(graph, "version", None))
    
    def _search(self, query: str) -> List[Tuple[str, str]]:
        results = []
        
        if self._external_search_func is not None:
            try:
                external_results = self._external_search_func(query)
                for content, url in external_results[:self.search_top_k]:
                    results.append((content, "TRUSTED_KB", url))
            except Exception:
                pass
        
        if self._knowledge_base is not None:
            remaining = self.search_top_k - len(results)
            if remaining > 0:
                graph_results = self._knowledge_base.search(query, top_k=remaining)
                for fact, score in graph_results:
                    results.append((fact.content, "LINEAGE_HISTORY", f"lineage:{fact.id}"))
        
        return results[:self.search_top_k]
    
    def verify(self, claim: str) -> VerificationOutput:
        cached_res = self._get_from_cache(claim)
        if cached_res:
            return cached_res

        search_results = self._search(claim)
        
        if not search_results:
            return VerificationOutput(
                result=VerifyResult.NO_EVIDENCE,
                evidence=[], source_urls=[],
                confidence=0.3, reason="No relevant evidence in knowledge base"
            )
        
        trusted_evidence = []
        lineage_evidence = []
        all_evidence_list = []
        sources = []
        
        for content, src_type, src_id in search_results:
            all_evidence_list.append(content)
            sources.append(src_id)
            if src_type == "TRUSTED_KB":
                trusted_evidence.append(content)
            else:
                lineage_evidence.append(content)
        
        if self._openai is None:
            return VerificationOutput(
                result=VerifyResult.NOT_ENOUGH_INFO,
                evidence=all_evidence_list, source_urls=sources,
                confidence=0.3, reason="No LLM for judgment"
            )
        
        evidence_parts = []
        if trusted_evidence:
            evidence_parts.append("= OFFICIAL SOURCES (High Authority) =")
            for i, e in enumerate(trusted_evidence):
                evidence_parts.append(f"[A-{i+1}] {e}")
        
        if lineage_evidence:
            evidence_parts.append("\n= CONVERSATION HISTORY (Low Authority) =")
            for i, e in enumerate(lineage_evidence):
                evidence_parts.append(f"[B-{i+1}] {e}")
                
        evidence_text = "\n".join(evidence_parts)
        
        try:
            messages = [
                {"role": "system", "content": "You are a rigorous fact checker. Output only the label."},
                {"role": "user", "content": VERIFY_PROMPT.format(claim=claim, evidence=evidence_text)}
            ]
            result_text, tokens = call_llm(self._openai, self.model, messages, 0.1, 50)
            self.tokens_used += tokens
            result_label = result_text.strip().upper()
            
            if result_label == "SUPPORTED":
                res = VerificationOutput(VerifyResult.SUPPORTED, all_evidence_list, sources, 0.85, "Evidence supports claim")
            elif result_label == "CONTRADICTED":
                res = VerificationOutput(VerifyResult.CONTRADICTED, all_evidence_list, sources, 0.85, "Evidence contradicts claim")
            else:
                res = VerificationOutput(VerifyResult.NOT_ENOUGH_INFO, all_evidence_list, sources, 0.5, "Insufficient evidence")

            # Never persist raw claims, evidence, or model output in an
            # implicit debug file. These values may contain sensitive data.
            logger.debug(
                "Verifier completed: result=%s evidence_count=%d",
                res.result.value, len(all_evidence_list),
            )
            
            self._add_to_cache(claim, res)
            return res
        except Exception:
            raise RuntimeError("Verification failed.")
