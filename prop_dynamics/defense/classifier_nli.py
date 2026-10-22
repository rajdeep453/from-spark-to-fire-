"""Classifier."""

from typing import List, Dict, Optional
from dataclasses import dataclass
from threading import Lock
from .classifier import ClassifyResult, ClassificationOutput
from .genealogy_graph import GenealogyGraph

try:
    from sentence_transformers import CrossEncoder
    HAS_NLI = True
except ImportError:
    HAS_NLI = False

_model_cache: Dict[str, "CrossEncoder"] = {}
_model_lock = Lock()


def _get_shared_model(model_name: str) -> "CrossEncoder":
    global _model_cache
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                print(f"Loading CrossEncoder model '{model_name}'... ")
                _model_cache[model_name] = CrossEncoder(model_name)
                print(f"Model loaded successfully! System ready for inference.")
    return _model_cache[model_name]


class NLIClassifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        if not HAS_NLI:
            raise ImportError("Dependency unavailable.")
        self.model = _get_shared_model(model_name)
        self.labels = ['contradiction', 'entailment', 'neutral']
        self.tokens_used = 0
    
    def reset_tokens(self):
        self.tokens_used = 0
    
    def set_openai(self, openai_module):
        pass
    
    def classify(self, atom: str, graph: GenealogyGraph, top_k: int = 5) -> ClassificationOutput:
        similar_facts = graph.search(atom, top_k=top_k)
        
        if not similar_facts:
            return ClassificationOutput(
                result=ClassifyResult.YELLOW, confidence=0.5,
                evidence=[], reason="No trusted facts to compare"
            )
        
        evidence_list = [fact.content for fact, _ in similar_facts]
        best_evidence = similar_facts[0][0].content
        
        scores = self.model.predict([(best_evidence, atom)])[0]
        label_idx = scores.argmax()
        label = self.labels[label_idx]
        confidence = float(scores[label_idx])
        
        if label == 'entailment':
            return ClassificationOutput(
                result=ClassifyResult.GREEN, confidence=confidence,
                evidence=evidence_list[:2], reason="NLI: entailment"
            )
        elif label == 'contradiction':
            return ClassificationOutput(
                result=ClassifyResult.RED, confidence=confidence,
                evidence=evidence_list[:2], reason="NLI: contradiction"
            )
        return ClassificationOutput(
            result=ClassifyResult.YELLOW, confidence=confidence,
            evidence=evidence_list[:2], reason="NLI: neutral"
        )
    
    def classify_batch(self, atoms: List[str], graph: GenealogyGraph, 
                       top_k: int = 3) -> List[ClassificationOutput]:
        if not atoms:
            return []
        
        pairs = []
        evidence_map = {}
        for i, atom in enumerate(atoms):
            similar_facts = graph.search(atom, top_k=top_k)
            if similar_facts:
                pairs.append((similar_facts[0][0].content, atom))
                evidence_map[i] = [f.content for f, _ in similar_facts[:2]]
            else:
                pairs.append((None, atom))
                evidence_map[i] = []
        
        valid_indices = [i for i, (ev, _) in enumerate(pairs) if ev is not None]
        valid_pairs = [(pairs[i][0], pairs[i][1]) for i in valid_indices]
        
        results = [None] * len(atoms)
        if valid_pairs:
            all_scores = self.model.predict(valid_pairs)
            for idx, scores in zip(valid_indices, all_scores):
                label_idx = scores.argmax()
                label = self.labels[label_idx]
                confidence = float(scores[label_idx])
                evidence = evidence_map[idx]
                
                if label == 'entailment':
                    results[idx] = ClassificationOutput(ClassifyResult.GREEN, confidence, evidence, "NLI: entailment")
                elif label == 'contradiction':
                    results[idx] = ClassificationOutput(ClassifyResult.RED, confidence, evidence, "NLI: contradiction")
                else:
                    results[idx] = ClassificationOutput(ClassifyResult.YELLOW, confidence, evidence, "NLI: neutral")
        
        for i in range(len(atoms)):
            if results[i] is None:
                results[i] = ClassificationOutput(ClassifyResult.YELLOW, 0.5, [], "No trusted facts")
        
        return results


class SimilarityClassifier:
    def __init__(self, entail_threshold: float = 0.9, contradict_threshold: float = 0.3):
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self.tokens_used = 0
    
    def reset_tokens(self):
        self.tokens_used = 0
    
    def set_openai(self, openai_module):
        pass
    
    def classify(self, atom: str, graph: GenealogyGraph, top_k: int = 5) -> ClassificationOutput:
        similar_facts = graph.search(atom, top_k=top_k)
        
        if not similar_facts:
            return ClassificationOutput(
                result=ClassifyResult.YELLOW, confidence=0.5,
                evidence=[], reason="No trusted facts"
            )
        
        best_sim = similar_facts[0][1]
        evidence_list = [fact.content for fact, _ in similar_facts]
        
        if best_sim >= self.entail_threshold:
            return ClassificationOutput(
                result=ClassifyResult.GREEN, confidence=best_sim,
                evidence=evidence_list[:2], reason=f"High similarity: {best_sim:.2f}"
            )
        elif best_sim <= self.contradict_threshold:
            return ClassificationOutput(
                result=ClassifyResult.RED, confidence=1-best_sim,
                evidence=evidence_list[:2], reason=f"Low similarity: {best_sim:.2f}"
            )
        return ClassificationOutput(
            result=ClassifyResult.YELLOW, confidence=best_sim,
            evidence=evidence_list[:2], reason=f"Medium similarity: {best_sim:.2f}"
        )
