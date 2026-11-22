"""
config/settings.py
==================
Centralised, type-safe application settings.

All tuneable parameters live here so that every other module imports
from one place.  Values can be overridden via environment variables
(loaded from .env by python-dotenv / pydantic-settings).

Usage
-----
    from config.settings import settings

    top_k = settings.retrieval.top_k
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Root project path (the directory that contains config/, src/, data/, …)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Sub-setting groups (kept as plain dataclasses-style Pydantic models so they
# can be nested inside the top-level Settings without inheriting BaseSettings
# themselves — avoiding multiple .env loads).
# ---------------------------------------------------------------------------

from pydantic import BaseModel  # noqa: E402  (placed after stdlib imports)


class LLMSettings(BaseModel):
    """Settings for the language model provider."""

    provider: Literal["openai"] = "openai"
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    request_timeout: int = Field(default=60, description="Seconds before timeout")
    max_retries: int = Field(default=3, description="Retry attempts on API errors")


class EmbeddingSettings(BaseModel):
    """Settings for the sentence-transformer embedding model."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace SentenceTransformer model name or local path",
    )
    batch_size: int = Field(default=64, gt=0)
    normalize_embeddings: bool = True


class ChunkingSettings(BaseModel):
    """Settings that control how documents are split into chunks."""

    strategy: Literal["fixed", "sentence", "recursive"] = "recursive"
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Target chunk size in tokens (approximate)",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Token overlap between consecutive chunks",
    )
    min_chunk_length: int = Field(
        default=50,
        gt=0,
        description="Chunks shorter than this are discarded",
    )


class VectorDBSettings(BaseModel):
    """ChromaDB connection and collection settings."""

    mode: Literal["local", "http"] = Field(
        default="local",
        description="'local' = in-process persistent client; 'http' = remote server",
    )
    persist_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "vectorstore",
        description="Storage directory for local ChromaDB",
    )
    host: str = "localhost"
    port: int = 8001
    collection_name: str = "rag_documents"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"


class RetrievalSettings(BaseModel):
    """Semantic retrieval / search settings."""

    top_k: int = Field(default=5, gt=0, description="Number of chunks to retrieve")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score; 0 disables filtering",
    )


class EvaluationSettings(BaseModel):
    """Settings for the evaluation and benchmarking framework."""

    benchmark_file: Path = Field(
        default=PROJECT_ROOT / "data" / "benchmarks" / "golden_qa.jsonl"
    )
    output_dir: Path = Field(default=PROJECT_ROOT / "reports")
    precision_k_values: list[int] = Field(
        default=[1, 3, 5],
        description="K values for Precision@K and Recall@K",
    )
    ragas_metrics: list[str] = Field(
        default=["faithfulness", "answer_correctness", "context_precision"],
        description="Ragas metric names to run during evaluation",
    )


class APISettings(BaseModel):
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Enterprise RAG Quality Engineering Platform"
    version: str = "0.1.0"


class LoggingSettings(BaseModel):
    """Application logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    log_to_file: bool = False
    log_file: Path = PROJECT_ROOT / "logs" / "app.log"
    rotation: str = "10 MB"


# ---------------------------------------------------------------------------
# Top-level settings — values are pulled from environment / .env automatically
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Master settings object for the entire application.

    Environment variables are read (with priority) in this order:
        1. Shell environment variables
        2. .env file (if present)
        3. Defaults defined in the sub-models above

    Sub-group env-var prefix mapping:
        llm.*         →  LLM_*          (e.g. LLM_MODEL, LLM_TEMPERATURE)
        embedding.*   →  EMBEDDING_*    (e.g. EMBEDDING_MODEL)
        chunking.*    →  CHUNK_*        (e.g. CHUNK_SIZE, CHUNK_OVERLAP)
        vectordb.*    →  CHROMA_*       (e.g. CHROMA_MODE, CHROMA_HOST)
        retrieval.*   →  RETRIEVAL_*    (e.g. RETRIEVAL_TOP_K)
        api.*         →  API_*          (e.g. API_HOST, API_PORT)
        logging.*     →  LOG_*          (e.g. LOG_LEVEL)
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",   # allows CHUNKING__CHUNK_SIZE=512 nesting
        case_sensitive=False,
        extra="ignore",
    )

    # Flat env-var overrides for the most commonly tuned values
    openai_api_key: str = Field(default="", repr=False)
    openai_org_id: str = Field(default="", repr=False)

    # Nested settings groups
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Convenience paths
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_docs_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_docs_dir(self) -> Path:
        return self.data_dir / "processed"


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
