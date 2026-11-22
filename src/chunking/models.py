"""
src/chunking/models.py
======================
Chunk data model for the RAG chunking pipeline.

A Chunk is the atomic unit that gets embedded and stored in the vector
database.  Every field needed for retrieval, evaluation, and before/after
comparison experiments is preserved here so downstream modules never need
to reach back to the original ParsedDocument.

Chunk IDs are deterministic: the same document, section, strategy, and
position always produce the same ID regardless of when chunking is run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.models import FileType


def _make_chunk_id(
    document_id: str,
    section_id: Optional[str],
    strategy: str,
    chunk_size: int,
    overlap: int,
    chunk_index: int,
) -> str:
    """
    Produce a deterministic chunk identifier.

    The ID is a 16-character hex digest of the concatenated inputs.
    Given identical parameters the ID is always the same, making it safe
    to re-run ingestion and chunking without orphaning vectors in the
    database.

    Parameters
    ----------
    document_id:
        The ``ParsedDocument.document_id`` of the source document.
    section_id:
        The ``Section.section_id`` (may be ``None``).
    strategy:
        Chunking strategy name, e.g. ``"fixed"``.
    chunk_size:
        Configured chunk size in characters.
    overlap:
        Configured overlap in characters.
    chunk_index:
        Zero-based position of this chunk within its source document.

    Returns
    -------
    str
        16-character lowercase hex string.
    """
    raw = "|".join(
        [
            document_id,
            section_id or "",
            strategy,
            str(chunk_size),
            str(overlap),
            str(chunk_index),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and vector storage.

    Attributes
    ----------
    chunk_id:
        Deterministic 16-char hex identifier (see :func:`_make_chunk_id`).
    document_id:
        Source ``ParsedDocument.document_id``.
    filename:
        Original source filename.
    file_type:
        ``FileType`` enum value from the source document.
    section_id:
        Original ``Section.section_id`` — ``None`` when the source had none.
    section_heading:
        Original section heading text — empty string when unavailable.
    chunk_index:
        Zero-based position of this chunk across the whole document.
    content:
        The actual text of this chunk.
    start_char:
        Starting character offset within the source text (section or document).
    end_char:
        Ending character offset within the source text (section or document).
    page_numbers:
        All PDF page numbers covered by this chunk.  Empty list for non-PDFs.
    metadata:
        Arbitrary key/value pairs including strategy, config, and provenance.
    """

    chunk_id: str
    document_id: str
    filename: str
    file_type: FileType
    section_id: Optional[str]
    section_heading: str
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    page_numbers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def char_length(self) -> int:
        """Number of characters in the chunk content."""
        return len(self.content)

    @property
    def has_section_id(self) -> bool:
        """True when the chunk carries an explicit section ID."""
        return self.section_id is not None

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.chunk_id!r}, "
            f"doc={self.filename!r}, "
            f"idx={self.chunk_index}, "
            f"chars={self.char_length})"
        )
