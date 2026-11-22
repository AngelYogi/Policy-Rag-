"""
src/chunking/fixed_size.py
==========================
Strategy A — Fixed-size character chunking.

Algorithm
---------
1. Build the full text to split from the document's content.
2. Walk through the text in steps of ``(chunk_size - overlap)`` characters.
3. At each step, extend the window to the nearest word boundary (space) so
   we do not cut mid-word.  If no space is found within a reasonable look-
   ahead the hard boundary is used instead.
4. Discard chunks shorter than ``config.min_chunk_length``.
5. Assign deterministic chunk IDs.

Metadata preserved per chunk
----------------------------
- document_id, filename, file_type
- section_id / section_heading (best-effort: the section whose character
  range contains the chunk's start_char)
- page_numbers (PDF: pages whose text overlaps the chunk's character range)
- chunking_strategy, chunk_size_config, overlap_config
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.chunking.base import BaseChunker
from src.chunking.config import ChunkingConfig
from src.chunking.models import Chunk, _make_chunk_id
from src.ingestion.models import FileType, ParsedDocument, Section

# Maximum characters to look ahead/behind for a word boundary
_WORD_BOUNDARY_LOOKAHEAD = 50


def _find_word_boundary(text: str, pos: int, direction: int = 1) -> int:
    """
    Scan from ``pos`` in ``direction`` (+1 forward, -1 backward) until a
    space character is found, or the scan exhausts ``_WORD_BOUNDARY_LOOKAHEAD``
    characters.  Returns the adjusted position, or the original ``pos`` if no
    boundary is found within the window.
    """
    limit = min(len(text), pos + _WORD_BOUNDARY_LOOKAHEAD * direction)
    candidate = pos
    step = direction
    while 0 <= candidate < len(text):
        if text[candidate] == " ":
            return candidate
        candidate += step
        if abs(candidate - pos) > _WORD_BOUNDARY_LOOKAHEAD:
            break
    return pos


def _build_section_index(
    document: ParsedDocument,
) -> List[Tuple[int, int, Section]]:
    """
    Build a list of (start_char, end_char, section) tuples by walking
    ``document.content`` and finding where each section's text begins.

    Used to attribute a chunk's character range to the right section.
    """
    index: List[Tuple[int, int, Section]] = []
    search_start = 0
    for section in document.sections:
        text_to_find = section.content.strip()
        if not text_to_find:
            continue
        pos = document.content.find(text_to_find, search_start)
        if pos == -1:
            # Fallback: section content not literally present (shouldn't happen)
            continue
        end = pos + len(text_to_find)
        index.append((pos, end, section))
        search_start = pos  # allow overlapping ranges
    return index


def _resolve_section(
    section_index: List[Tuple[int, int, Section]],
    start_char: int,
) -> Tuple[Optional[str], str]:
    """
    Return ``(section_id, section_heading)`` for the section that best
    covers ``start_char``.  Picks the last section whose start is ≤
    ``start_char`` (i.e. the most recently opened section).
    """
    best_id: Optional[str] = None
    best_heading: str = ""
    for sec_start, _sec_end, section in section_index:
        if sec_start <= start_char:
            best_id = section.section_id
            best_heading = section.heading
        else:
            break
    return best_id, best_heading


def _resolve_page_numbers(
    document: ParsedDocument,
    section_index: List[Tuple[int, int, Section]],
    start_char: int,
    end_char: int,
) -> List[int]:
    """
    Collect PDF page numbers whose section range overlaps [start_char, end_char].
    Returns an empty list for non-PDF documents.
    """
    if document.file_type != FileType.PDF:
        return []
    pages: List[int] = []
    for sec_start, sec_end, section in section_index:
        if section.page_number is None:
            continue
        # Overlap check
        if sec_start < end_char and sec_end > start_char:
            if section.page_number not in pages:
                pages.append(section.page_number)
    return sorted(pages)


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size character chunker with configurable overlap and
    word-boundary snapping.
    """

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Split ``document.content`` into fixed-size character chunks.

        Parameters
        ----------
        document:
            A ``ParsedDocument`` produced by Stage 2.

        Returns
        -------
        List[Chunk]
            Ordered list of chunks; empty list for empty documents.
        """
        text = document.content
        if not text.strip():
            logger.debug(
                "FixedSizeChunker: document '{}' is empty, returning no chunks.",
                document.filename,
            )
            return []

        section_index = _build_section_index(document)
        chunks: List[Chunk] = []
        step = self.config.chunk_size - self.config.overlap
        pos = 0
        global_index = 0

        while pos < len(text):
            end = min(pos + self.config.chunk_size, len(text))

            # Snap end to a word boundary (avoid cutting mid-word)
            if end < len(text):
                snapped = _find_word_boundary(text, end, direction=-1)
                if snapped > pos:
                    end = snapped

            chunk_text = text[pos:end]

            if len(chunk_text.strip()) >= self.config.min_chunk_length:
                section_id, section_heading = _resolve_section(
                    section_index, pos
                )
                page_numbers = _resolve_page_numbers(
                    document, section_index, pos, end
                )

                chunk_id = _make_chunk_id(
                    document_id=document.document_id,
                    section_id=section_id,
                    strategy=self.strategy_name,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.overlap,
                    chunk_index=global_index,
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        filename=document.filename,
                        file_type=document.file_type,
                        section_id=section_id,
                        section_heading=section_heading,
                        chunk_index=global_index,
                        content=chunk_text,
                        start_char=pos,
                        end_char=end,
                        page_numbers=page_numbers,
                        metadata={
                            **self._base_metadata(),
                            "word_boundary_snapping": True,
                        },
                    )
                )
                global_index += 1

            # Move forward by step (chunk_size - overlap)
            pos += step
            if step <= 0:
                break  # guard against zero-step infinite loop

        logger.info(
            "FixedSizeChunker: '{}' → {} chunk(s) "
            "(size={}, overlap={})",
            document.filename,
            len(chunks),
            self.config.chunk_size,
            self.config.overlap,
        )
        return chunks
