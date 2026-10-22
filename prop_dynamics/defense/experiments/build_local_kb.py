"""Build local FAISS index from outputs."""

import json
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import faiss

from prop_dynamics.defense.config import DefenseConfig, setup_openai
from prop_dynamics.defense.atomic_decomposer import AtomicDecomposer


def collect_texts(source_file: str) -> List[Tuple[str, str]]:
    items = []
    if not os.path.exists(source_file):
        return items
    with open(source_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append((line, source_file))
    return items


def embed_texts(openai_module, model: str, texts: List[str], batch_size: int = 512) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if hasattr(openai_module, "embeddings") and hasattr(openai_module.embeddings, "create"):
            resp = openai_module.embeddings.create(model=model, input=batch)
            vecs = [d.embedding for d in resp.data]
        elif hasattr(openai_module, "Embedding") and hasattr(openai_module.Embedding, "create"):
            resp = openai_module.Embedding.create(model=model, input=batch)
            vecs = [d["embedding"] for d in resp["data"]]
        else:
            raise RuntimeError("Embedding service unavailable")
        all_vecs.extend(vecs)
    arr = np.array(all_vecs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def build_index(source_file: str, out_dir: str, embed_model: str = "text-embedding-3-small"):
    print(f"=Loading knowledge base from: {source_file}")
    if not os.path.exists(source_file):
        print("=Source unavailable.")
        return

    os.makedirs(out_dir, exist_ok=True)
    config = DefenseConfig()
    try:
        openai_module = setup_openai(config)
    except Exception:
        print("Setup failed.")
        return

    decomposer = AtomicDecomposer(openai_module=None)

    raw_items = collect_texts(source_file)
    print(f"Found {len(raw_items)} raw items in source file.")
    
    if not raw_items:
        print("[!] No usable items.")
        return

    chunks = []
    print("Decomposing texts...")
    for text, source in raw_items:
        atoms = decomposer.decompose(text)
        for atom in atoms:
            atom = atom.strip()
            if atom:
                chunks.append((atom, source))

    seen = set()
    uniq_chunks = []
    for text, source in chunks:
        key = (text, source)
        if key in seen:
            continue
        seen.add(key)
        uniq_chunks.append((text, source))

    print(f"Generated {len(uniq_chunks)} unique chunks from {len(raw_items)} source items.")

    texts = [t for t, _ in uniq_chunks]
    print(f"embedding {len(texts)} chunks with model '{embed_model}'...")
    try:
        embeddings = embed_texts(openai_module, embed_model, texts)
    except Exception:
        print("Embedding failed.")
        return

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    index_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"saved FAISS index to: {index_path}")

    meta_path = os.path.join(out_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, (text, source) in enumerate(uniq_chunks):
            obj = {"chunk_id": i, "text": text, "source": source}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved metadata to: {meta_path}")
    print("Knowledge base build complete.")


if __name__ == "__main__":
    source_file = os.path.join("outputs", "defense", "kb_source.txt")
    out_dir = os.path.join("outputs", "defense", "kb")
    build_index(source_file, out_dir)
