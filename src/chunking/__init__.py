"""
src/chunking/__init__.py
========================
Public API for the chunking module.

    from src.chunking import chunk_document, chunk_documents, ChunkingConfig, Chunk
"""

from src.chunking.config import (
    ALL_PRESETS,
    PRESET_LARGE,
    PRESET_MEDIUM,
    PRESET_SMALL,
    ChunkingConfig,
)
from src.chunking.chunker import chunk_document, chunk_documents, get_chunker
from src.chunking.models import Chunk

__all__ = [
    "ALL_PRESETS",
    "Chunk",
    "ChunkingConfig",
    "PRESET_LARGE",
    "PRESET_MEDIUM",
    "PRESET_SMALL",
    "chunk_document",
    "chunk_documents",
    "get_chunker",
]
