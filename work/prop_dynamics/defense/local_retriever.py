"""Local retrieval."""

import json
import os
from typing import List, Tuple, Optional

import numpy as np
import faiss


class LocalRetriever:
    def __init__(
        self,
        index_dir: str,
        openai_module=None,
        embed_model: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadata.jsonl")
        self._openai = openai_module
        self.embed_model = embed_model
        self.top_k = top_k
        self._index = faiss.read_index(self.index_path)
        self._meta = self._load_meta(self.meta_path)

    def _load_meta(self, path: str) -> List[dict]:
        meta = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meta.append(json.loads(line))
        return meta

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._openai is None:
            raise RuntimeError("Embedding service unavailable")
        if hasattr(self._openai, "embeddings") and hasattr(self._openai.embeddings, "create"):
            resp = self._openai.embeddings.create(model=self.embed_model, input=texts)
            vecs = [d.embedding for d in resp.data]
        elif hasattr(self._openai, "Embedding") and hasattr(self._openai.Embedding, "create"):
            resp = self._openai.Embedding.create(model=self.embed_model, input=texts)
            vecs = [d["embedding"] for d in resp["data"]]
        else:
            raise RuntimeError("Embedding service unavailable")
        arr = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(arr)
        return arr

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, str]]:
        k = top_k or self.top_k
        q = self._embed([query])
        scores, idxs = self._index.search(q, k)
        results = []
        for idx in idxs[0]:
            if idx < 0 or idx >= len(self._meta):
                continue
            item = self._meta[idx]
            text = item.get("text", "")
            source = item.get("source", "")
            chunk_id = item.get("chunk_id", idx)
            results.append((text, f"local:{source}#chunk_{chunk_id}"))
        return results
