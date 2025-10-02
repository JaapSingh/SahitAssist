from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AzureConfig:
    endpoint: str
    api_key: str
    deployment: str
    temperature: float = 0.2
    api_version: str = "2024-08-01-preview"

@dataclass
class EmbeddingConfig:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str = "2024-08-01-preview"
    dim: int = 1536
    batch_size: int = 64

@dataclass
class RetrievalConfig:
    top_k: int = 5
    reuse_threshold: float = 0.95
    enable: bool = False
    max_context_chars: int = 1200

@dataclass
class PipelineConfig:
    batch_size: int = 8
    max_retries: int = 5
    concurrency: int = 4
    timeout_seconds: int = 60
    dry_run: bool = False
    force: bool = False
    retrieval: RetrievalConfig = RetrievalConfig()

def load_azure_config() -> AzureConfig:
    return AzureConfig(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt5-translation"),
        temperature=float(os.getenv("TRANSLATION_MODEL_TEMPERATURE", "0.2")),
    )

def load_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "embeddings-gpt5"),
        api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-08-01-preview"),
        dim=int(os.getenv("EMBEDDING_MODEL_DIM", "1536")),
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
    )

def load_retrieval_config(enable: bool = False) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        reuse_threshold=float(os.getenv("RETRIEVAL_REUSE_THRESHOLD", "0.95")),
        enable=enable,
        max_context_chars=int(os.getenv("RETRIEVAL_MAX_CONTEXT_CHARS", "1200")),
    )

def validate_config(azure: AzureConfig):
    missing = []
    if not azure.endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not azure.api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment vars: {', '.join(missing)}")