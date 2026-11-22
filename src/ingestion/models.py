"""
src/ingestion/models.py
=======================
Document data models for the ingestion pipeline.

These dataclasses define the canonical structure that every parser produces.
Downstream modules (chunking, embedding, evaluation) consume these models
without knowing which parser created them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileType(str, Enum):
    """Supported document types."""

    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    UNKNOWN = "unknown"


@dataclass
class Section:
    """
    A logical section within a parsed document.

    Attributes
    ----------
    heading:
        The section heading text. Empty string when no heading is detected.
    content:
        All text that belongs to this section (excluding the heading line itself).
    section_id:
        The explicit anchor/ID found in the source (e.g. ``{#my-section}`` in
        Markdown).  ``None`` when the source contains no explicit ID — IDs are
        never invented.
    page_number:
        1-based PDF page number where the section begins.  ``None`` for
        non-PDF formats or when page information is unavailable.
    metadata:
        Arbitrary per-section key/value pairs (e.g. heading level, bbox).
    """

    heading: str
    content: str
    section_id: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """
    A fully parsed document ready for the chunking pipeline.

    Attributes
    ----------
    document_id:
        UUID-like unique identifier generated at parse time.
    filename:
        Original filename (e.g. ``policy.pdf``).
    file_path:
        Absolute path of the source file.
    file_type:
        ``FileType`` enum value.
    title:
        Document title extracted from content (first H1 for Markdown, first
        line heuristic for plain text, metadata for PDF).  ``None`` when
        no title is determinable.
    content:
        Full extracted text of the document, sections joined with newlines.
    sections:
        Ordered list of :class:`Section` objects.
    page_count:
        Number of pages for PDF documents; ``None`` for other formats.
    metadata:
        Document-level metadata (author, creation date, source path, etc.).
    """

    document_id: str
    filename: str
    file_path: str
    file_type: FileType
    title: Optional[str]
    content: str
    sections: List[Section]
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        """Return True when the document has no meaningful text content."""
        return not self.content.strip()

    @property
    def section_count(self) -> int:
        """Number of sections extracted."""
        return len(self.sections)

    def __repr__(self) -> str:
        return (
            f"ParsedDocument(id={self.document_id!r}, "
            f"file={self.filename!r}, "
            f"type={self.file_type.value}, "
            f"sections={self.section_count}, "
            f"chars={len(self.content)})"
        )
