"""Genealogy Graph."""

import uuid
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class VerificationStatus(Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


@dataclass
class AtomicFact:
    id: str
    content: str
    agent_id: str
    timestamp: float
    embedding: Optional[np.ndarray] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    sources: List[str] = field(default_factory=list)
    hop_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id, "content": self.content, "agent_id": self.agent_id,
            "timestamp": self.timestamp, "status": self.status.value,
            "sources": self.sources, "hop_count": self.hop_count,
        }


_SHARED_MODELS: Dict[str, 'SentenceTransformer'] = {}
_MODEL_LOCK = __import__('threading').Lock()


def get_shared_model(model_name: str = "all-MiniLM-L6-v2"):
    with _MODEL_LOCK:
        if model_name not in _SHARED_MODELS and HAS_ST:
            _SHARED_MODELS[model_name] = SentenceTransformer(model_name)
        return _SHARED_MODELS.get(model_name)


class GenealogyGraph:
    def __init__(self, embedding_dim: int = 384, similarity_threshold: float = 0.8):
        self.nodes: Dict[str, AtomicFact] = {}
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self._embeddings: List[np.ndarray] = []
        self._id_to_idx: Dict[str, int] = {}
        self._embedding_func: Optional[Callable] = None
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._local_model = None
        self._norm_matrix_cache: Optional[np.ndarray] = None
        self._cache_valid = False
    
    def set_local_model(self, model_name: str = "all-MiniLM-L6-v2"):
        self._local_model = get_shared_model(model_name)
        if self._local_model:
            self.embedding_dim = self._local_model.get_sentence_embedding_dimension()
    
    def set_embedding_func(self, func: Callable):
        self._embedding_func = func
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if text in self._emb_cache:
            return self._emb_cache[text]
        if self._local_model is not None:
            emb = self._local_model.encode(text, convert_to_numpy=True).astype(np.float32)
            self._emb_cache[text] = emb
            return emb
        if self._embedding_func is not None:
            emb = self._embedding_func(text)
            self._emb_cache[text] = emb
            return emb
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def add(self, content: str, agent_id: str, 
            status: VerificationStatus = VerificationStatus.VERIFIED,
            sources: List[str] = None, hop_count: int = 0) -> AtomicFact:
        fact_id = str(uuid.uuid4())[:8]
        embedding = self._get_embedding(content)
        
        fact = AtomicFact(
            id=fact_id, content=content, agent_id=agent_id,
            timestamp=time.time(), embedding=embedding,
            status=status, sources=sources or [], hop_count=hop_count,
        )
        
        self.nodes[fact_id] = fact
        self._embeddings.append(embedding)
        self._id_to_idx[fact_id] = len(self._embeddings) - 1
        self._cache_valid = False
        return fact
    
    def _get_norm_matrix(self) -> np.ndarray:
        if not self._cache_valid or self._norm_matrix_cache is None:
            if not self._embeddings:
                return np.array([])
            emb_matrix = np.array(self._embeddings)
            self._norm_matrix_cache = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
            self._cache_valid = True
        return self._norm_matrix_cache
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[AtomicFact, float]]:
        if not self._embeddings:
            return []
        
        query_emb = self._get_embedding(query)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        emb_norms = self._get_norm_matrix()
        if emb_norms.size == 0:
            return []
        similarities = np.dot(emb_norms, query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        idx_to_id = {v: k for k, v in self._id_to_idx.items()}
        
        results = []
        for idx in top_indices:
            fact_id = idx_to_id[idx]
            fact = self.nodes[fact_id]
            sim = float(similarities[idx])
            if sim >= self.similarity_threshold:
                results.append((fact, sim))
        return results
    
    def get_verified_facts(self) -> List[AtomicFact]:
        return [f for f in self.nodes.values() if f.status == VerificationStatus.VERIFIED]
    
    def mark_rejected(self, fact_id: str):
        if fact_id in self.nodes:
            self.nodes[fact_id].status = VerificationStatus.REJECTED
    
    def get_descendants(self, fact_id: str) -> List[str]:
        descendants = []
        for fid, fact in self.nodes.items():
            if fact_id in fact.sources:
                descendants.append(fid)
                descendants.extend(self.get_descendants(fid))
        return list(set(descendants))
    
    def prune_branch(self, fact_id: str):
        self.mark_rejected(fact_id)
        for desc_id in self.get_descendants(fact_id):
            self.mark_rejected(desc_id)
    
    def clear(self):
        self.nodes.clear()
        self._embeddings.clear()
        self._id_to_idx.clear()
        self._emb_cache.clear()
        self._norm_matrix_cache = None
        self._cache_valid = False
    
    def __len__(self) -> int:
        return len(self.nodes)
