"""
src/chunking/base.py
====================
Abstract base class for all chunking strategies.

Adding a new strategy means:
  1. Subclass ``BaseChunker``.
  2. Implement ``strategy_name`` and ``chunk_document``.
  3. Register the class in ``chunker.py``.

Nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.chunking.config import ChunkingConfig
from src.chunking.models import Chunk
from src.ingestion.models import ParsedDocument


class BaseChunker(ABC):
    """
    Abstract base for chunking strategy implementations.

    Parameters
    ----------
    config:
        ``ChunkingConfig`` instance controlling this chunker's behaviour.
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Short name identifying this strategy, e.g. ``"fixed"``."""

    @abstractmethod
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Split a parsed document into an ordered list of chunks.

        Parameters
        ----------
        document:
            Output from Stage 2 ingestion.

        Returns
        -------
        List[Chunk]
            Ordered list of chunks.  May be empty for empty documents.
        """

    # ------------------------------------------------------------------
    # Shared utility: attach standard metadata to a chunk
    # ------------------------------------------------------------------

    def _base_metadata(self) -> dict:
        """Return metadata fields common to all strategies."""
        return {
            "chunking_strategy": self.config.strategy,
            "chunk_size_config": self.config.chunk_size,
            "overlap_config": self.config.overlap,
        }
