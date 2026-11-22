"""
src/chunking/recursive.py
==========================
Strategy B — Recursive character chunking.

Algorithm
---------
Text is split hierarchically through a prioritised list of separators:

  1. Double newline  (paragraph boundary)
  2. Single newline  (line boundary)
  3. Period + space  (sentence boundary)
  4. Space           (word boundary)
  5. Empty string    (character, last resort)

At each level the algorithm tries to merge adjacent small pieces into chunks
≤ chunk_size.  If a piece is itself larger than chunk_size it is split
further using the next separator in the list.

This preserves semantic cohesion as long as possible — paragraphs are kept
intact whenever they fit, then lines, then sentences, before resorting to
word or character splits.

Overlap is applied between consecutive chunks by prepending the tail of the
previous chunk to the start of the next.

Metadata preserved
------------------
Same as FixedSizeChunker (section attribution via character offset index).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from src.chunking.base import BaseChunker
from src.chunking.config import ChunkingConfig
from src.chunking.models import Chunk, _make_chunk_id
from src.ingestion.models import FileType, ParsedDocument, Section

# Ordered list of separators — tried from coarsest to finest
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text_recursive(
    text: str,
    chunk_size: int,
    separators: List[str],
) -> List[str]:
    """
    Recursively split ``text`` into pieces no longer than ``chunk_size``
    using the first applicable separator in ``separators``.

    Returns a flat list of text pieces (no overlap applied yet).
    """
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        # Last resort: hard character split
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        # Character-level split
        return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)

    # Merge small parts together and recurse on oversized parts
    merged: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + separator + part).lstrip(separator) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            if len(part) > chunk_size:
                # This single part is too big — go deeper
                merged.extend(_split_text_recursive(part, chunk_size, remaining))
                current = ""
            else:
                current = part

    if current:
        merged.append(current)

    return merged


def _apply_overlap(pieces: List[str], overlap: int) -> List[Tuple[str, int, int]]:
    """
    Given a flat list of text pieces (no overlap), produce
    ``(content, start_char, end_char)`` tuples with overlap applied.

    ``start_char`` / ``end_char`` are offsets into the reconstructed
    original text (pieces joined without separator — conservative estimate).

    The overlap is taken from the **tail** of the previous chunk and
    prepended to the current chunk content.
    """
    if not pieces:
        return []

    result: List[Tuple[str, int, int]] = []
    # Build cumulative character positions in the original text
    # (treating pieces as contiguous)
    offsets: List[int] = []
    pos = 0
    for p in pieces:
        offsets.append(pos)
        pos += len(p)

    for i, piece in enumerate(pieces):
        if i == 0 or overlap == 0:
            content = piece
            start = offsets[i]
        else:
            # Prepend tail of previous chunk
            prev = pieces[i - 1]
            tail = prev[-overlap:] if len(prev) >= overlap else prev
            content = tail + piece
            start = max(0, offsets[i] - len(tail))

        end = offsets[i] + len(piece)
        result.append((content, start, end))

    return result


def _build_section_index(
    document: ParsedDocument,
) -> List[Tuple[int, int, Section]]:
    """Identical to the one in fixed_size.py — duplicated to keep modules independent."""
    index: List[Tuple[int, int, Section]] = []
    search_start = 0
    for section in document.sections:
        text_to_find = section.content.strip()
        if not text_to_find:
            continue
        pos = document.content.find(text_to_find, search_start)
        if pos == -1:
            continue
        end = pos + len(text_to_find)
        index.append((pos, end, section))
        search_start = pos
    return index


def _resolve_section(
    section_index: List[Tuple[int, int, Section]],
    start_char: int,
) -> Tuple[Optional[str], str]:
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
    if document.file_type != FileType.PDF:
        return []
    pages: List[int] = []
    for sec_start, sec_end, section in section_index:
        if section.page_number is None:
            continue
        if sec_start < end_char and sec_end > start_char:
            if section.page_number not in pages:
                pages.append(section.page_number)
    return sorted(pages)


class RecursiveChunker(BaseChunker):
    """
    Recursive character chunker that respects semantic boundaries.

    Tries paragraph → line → sentence → word → character splits in
    priority order.
    """

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Split a document using recursive separator-based chunking.

        Parameters
        ----------
        document:
            ``ParsedDocument`` from Stage 2.

        Returns
        -------
        List[Chunk]
        """
        text = document.content
        if not text.strip():
            logger.debug(
                "RecursiveChunker: document '{}' is empty, returning no chunks.",
                document.filename,
            )
            return []

        section_index = _build_section_index(document)
        raw_pieces = _split_text_recursive(
            text, self.config.chunk_size, _SEPARATORS
        )
        # Discard near-empty pieces before applying overlap
        raw_pieces = [p for p in raw_pieces if p.strip()]

        overlapped = _apply_overlap(raw_pieces, self.config.overlap)
        chunks: List[Chunk] = []

        for idx, (content, start, end) in enumerate(overlapped):
            if len(content.strip()) < self.config.min_chunk_length:
                continue

            section_id, section_heading = _resolve_section(section_index, start)
            page_numbers = _resolve_page_numbers(
                document, section_index, start, end
            )

            chunk_id = _make_chunk_id(
                document_id=document.document_id,
                section_id=section_id,
                strategy=self.strategy_name,
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
                chunk_index=idx,
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    section_id=section_id,
                    section_heading=section_heading,
                    chunk_index=idx,
                    content=content,
                    start_char=start,
                    end_char=end,
                    page_numbers=page_numbers,
                    metadata={
                        **self._base_metadata(),
                        "separators": _SEPARATORS,
                    },
                )
            )

        logger.info(
            "RecursiveChunker: '{}' → {} chunk(s) "
            "(size={}, overlap={})",
            document.filename,
            len(chunks),
            self.config.chunk_size,
            self.config.overlap,
        )
        return chunks
