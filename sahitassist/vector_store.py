from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np

META_FILENAME = "meta.jsonl"
VECTORS_FILENAME = "vectors.npy"
STATS_FILENAME = "stats.json"

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.meta: List[Dict[str, Any]] = []
        self._id_to_index: Dict[str, int] = {}

    def add(self, vector: List[float], meta: Dict[str, Any]):
        idx = len(self.meta)
        self.meta.append(meta)
        if self.vectors.shape[0] == 0:
            self.vectors = np.zeros((1, self.dim), dtype="float32")
            self.vectors[0] = np.array(vector, dtype="float32")
        else:
            self.vectors = np.vstack([self.vectors, np.array(vector, dtype="float32")])
        _id = meta.get("uuid") or meta.get("id")
        if _id:
            self._id_to_index[_id] = idx

    def size(self) -> int:
        return len(self.meta)

    def search(self, vector: List[float], top_k: int = 5, exclude_ids: List[str] | None = None) -> List[Tuple[float, Dict[str, Any]]]:
        if self.size() == 0:
            return []
        q = np.array(vector, dtype="float32")
        qn = q / (np.linalg.norm(q) + 1e-9)
        vn = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
        sims = vn @ qn
        exclude_set = set(exclude_ids or [])
        indices = np.argsort(-sims)[: top_k + (len(exclude_ids) if exclude_ids else 0)]
        results = []
        for idx in indices:
            m = self.meta[idx]
            uid = m.get("uuid") or m.get("id")
            if uid and uid in exclude_set:
                continue
            results.append((float(sims[idx]), m))
            if len(results) >= top_k:
                break
        return results

    def persist(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        vectors_path = os.path.join(directory, VECTORS_FILENAME)
        meta_path = os.path.join(directory, META_FILENAME)
        stats_path = os.path.join(directory, STATS_FILENAME)
        np.save(vectors_path, self.vectors)
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({"size": self.size(), "dim": self.dim}, f)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        vectors_path = os.path.join(directory, VECTORS_FILENAME)
        meta_path = os.path.join(directory, META_FILENAME)
        if not (os.path.exists(vectors_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Vector store files not found.")
        vectors = np.load(vectors_path)
        dim = vectors.shape[1]
        store = cls(dim=dim)
        store.vectors = vectors
        with open(meta_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                meta = json.loads(line)
                store.meta.append(meta)
                uid = meta.get("uuid") or meta.get("id")
                if uid:
                    store._id_to_index[uid] = idx
        return store