"""Genealogy Graph."""

import hashlib
import re
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
        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        if not np.isfinite(similarity_threshold) or not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
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
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic revision for decisions cached against this graph."""
        return self._version
    
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
            emb = np.asarray(self._embedding_func(text), dtype=np.float32)
            self._emb_cache[text] = emb
            return emb

        # Stable signed feature hashing is a deterministic offline fallback.
        # It is lexical, not a substitute for the configured sentence model.
        tokens = re.findall(r"\w+", text.casefold())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] & 1 else -1.0
            embedding[bucket] += sign
        return embedding
    
    def add(self, content: str, agent_id: str, 
            status: VerificationStatus = VerificationStatus.VERIFIED,
            sources: List[str] = None, hop_count: int = 0) -> AtomicFact:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("fact content must be a non-empty string")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(status, VerificationStatus):
            raise TypeError("status must be a VerificationStatus")
        if sources is None:
            sources = []
        if not isinstance(sources, list) or any(
            not isinstance(source, str) or not source.strip() for source in sources
        ):
            raise ValueError("sources must be a list of non-empty fact IDs")
        if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count < 0:
            raise ValueError("hop_count must be a non-negative integer")
        fact_id = uuid.uuid4().hex
        embedding = np.asarray(self._get_embedding(content), dtype=np.float32)
        if embedding.ndim != 1 or embedding.size != self.embedding_dim:
            raise ValueError("embedding must be one-dimensional and match embedding_dim")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("embedding must contain only finite values")
        
        fact = AtomicFact(
            id=fact_id, content=content, agent_id=agent_id,
            timestamp=time.time(), embedding=embedding,
            status=status, sources=list(sources), hop_count=hop_count,
        )
        
        self.nodes[fact_id] = fact
        self._embeddings.append(embedding)
        self._id_to_idx[fact_id] = len(self._embeddings) - 1
        self._cache_valid = False
        self._version += 1
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
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._embeddings:
            return []
        
        query_emb = np.asarray(self._get_embedding(query), dtype=np.float32)
        if query_emb.ndim != 1 or query_emb.size != self.embedding_dim:
            raise ValueError("query embedding must be one-dimensional and match embedding_dim")
        if not np.all(np.isfinite(query_emb)):
            raise ValueError("query embedding must contain only finite values")
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        emb_norms = self._get_norm_matrix()
        if emb_norms.size == 0:
            return []
        similarities = np.dot(emb_norms, query_norm)
        
        verified_indices = [
            idx for fact_id, idx in self._id_to_idx.items()
            if self.nodes[fact_id].status == VerificationStatus.VERIFIED
        ]
        top_indices = sorted(
            verified_indices,
            key=lambda idx: (-float(similarities[idx]), idx),
        )[:top_k]
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
            if self.nodes[fact_id].status != VerificationStatus.REJECTED:
                self.nodes[fact_id].status = VerificationStatus.REJECTED
                self._version += 1
    
    def get_descendants(self, fact_id: str) -> List[str]:
        descendants = set()
        pending = [fact_id]
        while pending:
            parent_id = pending.pop()
            for child_id, fact in self.nodes.items():
                if parent_id in fact.sources and child_id != fact_id and child_id not in descendants:
                    descendants.add(child_id)
                    pending.append(child_id)
        return sorted(descendants)
    
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
        self._version += 1
    
    def __len__(self) -> int:
        return len(self.nodes)
