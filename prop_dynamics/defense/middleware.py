"""Middleware."""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time
import numpy as np

from .config import DefenseConfig
from .genealogy_graph import GenealogyGraph, VerificationStatus
from .atomic_decomposer import AtomicDecomposer
from .classifier import Classifier, ClassifyResult
from .classifier_nli import NLIClassifier
from .verifier import ExternalVerifier, VerifyResult


class AtomColor(Enum):
    GREEN = "green"
    RED = "red"
    YELLOW = "yellow"


@dataclass
class AtomInfo:
    content: str
    color: AtomColor
    tag: Optional[str] = None
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class ProcessResult:
    can_send: bool
    send_buffer: List[AtomInfo]
    rollback_queue: List[AtomInfo]
    rollback_feedback: Optional[str]
    final_message: str
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "green": sum(1 for a in self.send_buffer if a.color == AtomColor.GREEN),
            "yellow": sum(1 for a in self.send_buffer if a.color == AtomColor.YELLOW),
            "red_blocked": sum(1 for a in self.rollback_queue if a.color == AtomColor.RED),
        }


class DefenseMiddleware:
    def __init__(self, config: DefenseConfig, graph: Optional[GenealogyGraph] = None,
                 openai_module=None):
        self.config = config
        self.strategy = config.strategy
        self.classifier_type = config.classifier_type
        self.max_retries = config.max_retries
        self._openai = openai_module
        self.max_workers = 20
        
        self.graph = graph if graph is not None else GenealogyGraph(
            similarity_threshold=config.similarity_threshold
        )
        if config.use_local_embedding:
            self.graph.set_local_model(config.embedding_model)
        
        self.decomposer = AtomicDecomposer(model=config.decomposer_model, openai_module=openai_module, 
                                           enable_cache=config.enable_decomposer_cache)
        
        if config.classifier_type == "nli":
            self.classifier = NLIClassifier(model_name=config.nli_model)
        else:
            self.classifier = Classifier(model=config.classifier_model, openai_module=openai_module)
        
        self.verifier = ExternalVerifier(model=config.verifier_model, 
                                         search_top_k=config.search_top_k, openai_module=openai_module)
        
        if config.enable_graph:
            self.verifier.set_knowledge_base(self.graph)
        else:
            self.verifier.set_knowledge_base(None)
        
        if openai_module:
            self.set_openai(openai_module)
        
        self._hub_agents: set = set()
        self._stats_lock = Lock()
        self.stats = {
            "total_atoms": 0, "green": 0, "yellow": 0, "red": 0,
            "verified_true": 0, "verified_false": 0, "unverifiable": 0,
            "rollbacks": 0, "forced_pass": 0, "blocked_atoms": 0
        }
        self.latency_stats = {
            "decompose": 0.0, "classify": 0.0, "verify": 0.0, "assemble": 0.0
        }
    
    def set_openai(self, openai_module):
        self._openai = openai_module
        self.decomposer.set_openai(openai_module)
        self.classifier.set_openai(openai_module)
        self.verifier.set_openai(openai_module)
        
        if openai_module is not None:
            def get_emb(text: str):
                if hasattr(openai_module, 'embeddings') and hasattr(openai_module.embeddings, 'create'):
                    resp = openai_module.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return np.array(resp.data[0].embedding, dtype=np.float32)
                elif hasattr(openai_module, 'Embedding') and hasattr(openai_module.Embedding, 'create'):
                    resp = openai_module.Embedding.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    return np.array(resp["data"][0]["embedding"], dtype=np.float32)
                else:
                    raise RuntimeError("Embedding service unavailable")
            self.graph.set_embedding_func(get_emb)
    
    def set_hub_agents(self, agents: List[str]):
        self._hub_agents = set(agents)
    
    def is_hub(self, agent: str) -> bool:
        return agent in self._hub_agents
    
    def init_graph_from_task(self, task) -> int:
        if not self.config.init_facts:
            self.graph.clear()
            return 0

        self.graph.clear()
        seed_facts = []
        if hasattr(task, 'resource_url') and task.resource_url:
            seed_facts.append(f"The trusted resource URL is {task.resource_url}")
        if hasattr(task, 'facts') and task.facts:
            for fact in task.facts:
                atoms = self.decomposer.decompose(fact)
                seed_facts.extend(atoms)
        for fact in seed_facts:
            self.graph.add(content=fact, agent_id="SYSTEM", 
                          status=VerificationStatus.VERIFIED, hop_count=0)
        return len(seed_facts)
    
    def _classify_single(self, atom: str) -> AtomInfo:
        classification = self.classifier.classify(atom, self.graph)
        
        if classification.result == ClassifyResult.GREEN:
            color = AtomColor.GREEN
            reason = "Matches consensus in lineage graph"
        elif classification.result == ClassifyResult.RED:
            color = AtomColor.RED
            reason = "Conflicts with lineage graph"
        else:
            color = AtomColor.YELLOW
            reason = "New information, not in lineage graph"
        
        return AtomInfo(
            content=atom, color=color, reason=reason,
            evidence=classification.evidence[:2] if classification.evidence else [],
            sources=list(classification.evidence_ids or []),
        )
    
    def _stage1_screen(self, atoms: List[str]) -> List[AtomInfo]:
        if not atoms:
            return []
        
        if hasattr(self.classifier, "classify_batch"):
            classifications = self.classifier.classify_batch(atoms, self.graph)
            results = []
            for atom, cls in zip(atoms, classifications):
                if cls.result == ClassifyResult.GREEN:
                    color, reason = AtomColor.GREEN, "Matches consensus"
                elif cls.result == ClassifyResult.RED:
                    color, reason = AtomColor.RED, "Conflicts"
                else:
                    color, reason = AtomColor.YELLOW, "New info"
                with self._stats_lock:
                    self.stats[color.value] += 1
                results.append(
                    AtomInfo(
                        content=atom, color=color, reason=reason,
                        evidence=cls.evidence[:2], sources=list(cls.evidence_ids or []),
                    )
                )
            return results
        
        results = [None] * len(atoms)
        def classify_with_idx(idx: int, atom: str):
            result = self._classify_single(atom)
            with self._stats_lock:
                self.stats[result.color.value] += 1
            return idx, result
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(classify_with_idx, i, atom) for i, atom in enumerate(atoms)]
            for future in futures:
                idx, result = future.result()
                results[idx] = result
        
        return results
    
    def _stage2_route(self, atoms: List[AtomInfo], sender: str, receiver: str) -> List[AtomInfo]:
        is_hub_related = self.is_hub(sender) or self.is_hub(receiver)
        
        for atom in atoms:
            if atom.color != AtomColor.YELLOW:
                continue
            
            if self.strategy == "speed":
                atom.tag = "Unverified"
            elif self.strategy == "balanced":
                if is_hub_related:
                    atom.tag = "NEEDS_VERIFICATION"
                else:
                    atom.tag = "Unverified"
            else:
                atom.tag = "NEEDS_VERIFICATION"
        
        return atoms
    
    def _verify_single(self, atom: AtomInfo) -> AtomInfo:
        verification = self.verifier.verify(atom.content)
        if verification.evidence:
            atom.evidence = verification.evidence[:2]
        atom.sources = [
            source[len("lineage:"):]
            for source in verification.source_urls
            if isinstance(source, str) and source.startswith("lineage:")
        ]
        
        if verification.result == VerifyResult.SUPPORTED:
            atom.color = AtomColor.GREEN
            atom.tag = None
            atom.reason = "Verified by knowledge base"
            with self._stats_lock:
                self.stats["verified_true"] += 1
        elif verification.result == VerifyResult.CONTRADICTED:
            atom.color = AtomColor.RED
            atom.tag = None
            atom.reason = "Contradicted by knowledge base"
            with self._stats_lock:
                self.stats["verified_false"] += 1
        elif verification.result == VerifyResult.NO_EVIDENCE:
            atom.tag = "Unverified:KnowledgeGap"
            atom.reason = "No relevant evidence found in knowledge base"
            with self._stats_lock:
                self.stats["unverifiable"] += 1
        else:
            atom.tag = "Unverified:InsufficientLogic"
            atom.reason = "Evidence exists but is logically insufficient"
            with self._stats_lock:
                self.stats["unverifiable"] += 1
        
        return atom
    
    def _stage3_verify(self, atoms: List[AtomInfo]) -> List[AtomInfo]:
        to_verify_indices = [i for i, a in enumerate(atoms) if a.tag == "NEEDS_VERIFICATION"]
        
        if not to_verify_indices:
            return atoms
        
        def verify_with_idx(idx: int):
            atoms[idx] = self._verify_single(atoms[idx])
            return idx
        
        with ThreadPoolExecutor(max_workers=min(20, len(to_verify_indices))) as executor:
            futures = [executor.submit(verify_with_idx, idx) for idx in to_verify_indices]
            for future in futures:
                future.result()
        
        return atoms
    
    def _stage4_assemble(self, atoms: List[AtomInfo], sender: str) -> ProcessResult:
        send_buffer = []
        rollback_queue = []
        
        skip_blocking = self.config.suppress_tags

        for atom in atoms:
            if atom.color == AtomColor.GREEN:
                self.graph.add(content=atom.content, agent_id=sender,
                              status=VerificationStatus.VERIFIED, sources=atom.sources, hop_count=1)
                send_buffer.append(atom)
            elif atom.color == AtomColor.RED:
                if skip_blocking:
                    send_buffer.append(atom)
                else:
                    rollback_queue.append(atom)
            else:
                # Keep uncertain claims in the lineage for auditing, but
                # never let them enter verified-fact retrieval.
                self.graph.add(
                    content=atom.content, agent_id=sender,
                    status=VerificationStatus.UNVERIFIED,
                    sources=atom.sources, hop_count=1,
                )
                send_buffer.append(atom)
        
        parts = []
        for atom in send_buffer:
            if atom.tag and not self.config.suppress_tags:
                parts.append(f"[{atom.tag}] {atom.content}")
            else:
                parts.append(atom.content)
        final_message = " ".join(parts)
        
        feedback = None
        if rollback_queue:
            feedback = self._generate_rollback_feedback(rollback_queue)
        
        return ProcessResult(
            can_send=len(rollback_queue) == 0,
            send_buffer=send_buffer,
            rollback_queue=rollback_queue,
            rollback_feedback=feedback,
            final_message=final_message
        )
    
    def _generate_rollback_feedback(self, atoms: List[AtomInfo]) -> str:
        lines = ["Your message contains errors that need correction:"]
        for atom in atoms:
            evidence_str = f" (Evidence: {atom.evidence[0]})" if atom.evidence else ""
            lines.append(f"- [{atom.color.value.upper()}] \"{atom.content}\" - {atom.reason}{evidence_str}")
        lines.append("\nPlease rewrite these parts and resubmit.")
        return "\n".join(lines)
    
    def process_message(self, sender: str, receiver: str, message: str) -> ProcessResult:
        t0 = time.time()
        if not self.config.enable_decomposer:
             raw_atoms = [message]
        else:
            raw_atoms = self.decomposer.decompose(message)
        t1 = time.time()
        with self._stats_lock:
            self.stats["total_atoms"] += len(raw_atoms)
            self.latency_stats["decompose"] += t1 - t0
        
        if not raw_atoms:
            return ProcessResult(True, [], [], None, message)
        
        t2 = time.time()
        if self.config.enable_classifier:
            atoms = self._stage1_screen(raw_atoms)
        else:
            atoms = [AtomInfo(content=a, color=AtomColor.YELLOW, reason="Classifier disabled") 
                     for a in raw_atoms]
            with self._stats_lock:
                self.stats["yellow"] += len(atoms)
        t3 = time.time()
        with self._stats_lock:
            self.latency_stats["classify"] += t3 - t2
        
        atoms = self._stage2_route(atoms, sender, receiver)
        
        t4 = time.time()
        if self.config.enable_verifier:
            atoms = self._stage3_verify(atoms)
        else:
            for atom in atoms:
                if atom.tag == "NEEDS_VERIFICATION":
                    atom.tag = "Unverified:VerifierDisabled"
        t5 = time.time()
        with self._stats_lock:
            self.latency_stats["verify"] += t5 - t4
        
        t6 = time.time()
        result = self._stage4_assemble(atoms, sender)
        t7 = time.time()
        with self._stats_lock:
            self.latency_stats["assemble"] += t7 - t6
        
        return result
    
    def handle_rewrite(self, atoms: List[AtomInfo], rewritten: Dict[str, str], 
                       sender: str, receiver: str) -> ProcessResult:
        new_atoms = []
        for atom in atoms:
            if atom.content in rewritten:
                new_content = rewritten[atom.content]
                new_raw = self.decomposer.decompose(new_content)
                for raw in new_raw:
                    new_atom = self._classify_single(raw)
                    new_atom.retry_count = atom.retry_count + 1
                    self.stats[new_atom.color.value] += 1
                    new_atoms.append(new_atom)
            else:
                new_atoms.append(atom)
        
        new_atoms = self._stage2_route(new_atoms, sender, receiver)
        new_atoms = self._stage3_verify(new_atoms)
        return self._stage4_assemble(new_atoms, sender)
    
    def force_finalize(self, result: ProcessResult, sender: str) -> ProcessResult:
        with self._stats_lock:
            self.stats["rollbacks"] += 1
        
        for atom in result.rollback_queue:
            if atom.color == AtomColor.RED:
                with self._stats_lock:
                    self.stats["blocked_atoms"] += 1
            else:
                atom.tag = "HighRisk:MaxRetries"
                with self._stats_lock:
                    self.stats["forced_pass"] += 1
                result.send_buffer.append(atom)
        
        parts = []
        for atom in result.send_buffer:
            if atom.tag and not self.config.suppress_tags:
                parts.append(f"[{atom.tag}] {atom.content}")
            else:
                parts.append(atom.content)
        
        return ProcessResult(
            can_send=True,
            send_buffer=result.send_buffer,
            rollback_queue=[],
            rollback_feedback=None,
            final_message=" ".join(parts)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "graph_size": len(self.graph),
            "verified_facts": len(self.graph.get_verified_facts()),
            "api_tokens": self.decomposer.tokens_used + self.classifier.tokens_used + self.verifier.tokens_used,
            "tokens_decomposer": self.decomposer.tokens_used,
            "tokens_classifier": self.classifier.tokens_used,
            "tokens_verifier": self.verifier.tokens_used,
            "latency_decompose": round(self.latency_stats["decompose"], 3),
            "latency_classify": round(self.latency_stats["classify"], 3),
            "latency_verify": round(self.latency_stats["verify"], 3),
            "latency_assemble": round(self.latency_stats["assemble"], 3),
        }
    
    def get_graph_snapshot(self) -> List[dict]:
        return [fact.to_dict() for fact in self.graph.nodes.values()]
    
    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
        for key in self.latency_stats:
            self.latency_stats[key] = 0.0
        self.decomposer.reset_tokens()
        self.classifier.reset_tokens()
        self.verifier.reset_tokens()
