from __future__ import annotations
import httpx
from typing import List
import numpy as np
from .config import EmbeddingConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class EmbeddingError(Exception):
    pass

def chunkify(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, EmbeddingError)),
)
async def embed_texts(cfg: EmbeddingConfig, client: httpx.AsyncClient, texts: List[str]) -> List[List[float]]:
    url = f"{cfg.endpoint}/openai/deployments/{cfg.deployment}/embeddings?api-version={cfg.api_version}"
    vectors: List[List[float]] = []
    for batch in chunkify(texts, cfg.batch_size):
        payload = {"input": batch}
        try:
            resp = await client.post(
                url,
                headers={"api-key": cfg.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingError(str(e)) from e
        data = resp.json()
        try:
            ordered = sorted(data["data"], key=lambda d: d["index"])
            for item in ordered:
                vectors.append(item["embedding"])
        except Exception as e:
            raise EmbeddingError(f"Malformed embedding response: {e}") from e
    return vectors

def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    qn = query / (np.linalg.norm(query) + 1e-9)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return mn @ qn