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

At each level, if a piece is longer than chunk_size, it is recursively
subdivided using the next separator in the hierarchy.  The resulting atomic
pieces are then merged together using the appropriate separator up to
chunk_size, with configurable overlap maintained between consecutive chunks.

This preserves semantic boundaries as long as possible:
- Paragraphs are kept intact whenever they fit.
- Lines, sentences, words, and characters are split only when necessary.
- Natural whitespace and formatting are preserved without smashing words together.

Metadata preserved
------------------
- document_id, filename, file_type
- section_id, section_heading
- page_numbers (for PDFs)
- chunking_strategy, chunk_size_config, overlap_config
- exact start_char and end_char in document.content
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


def _split_text_hierarchical(
    text: str,
    separators: List[str],
    chunk_size: int,
) -> Tuple[str, List[str]]:
    """
    Split text using the coarsest separator that occurs in `text`,
    recursing on any piece that exceeds `chunk_size`.

    Returns
    -------
    (primary_separator, list_of_pieces)
    """
    if not text:
        return "", []

    sep = separators[-1]
    new_seps = []
    for i, s in enumerate(separators):
        if s == "":
            sep = s
            break
        if s in text:
            sep = s
            new_seps = separators[i + 1 :]
            break

    if sep == "":
        splits = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        splits = text.split(sep)

    good_splits: List[str] = []
    for s in splits:
        if len(s) <= chunk_size:
            good_splits.append(s)
        else:
            if new_seps:
                _child_sep, child_splits = _split_text_hierarchical(s, new_seps, chunk_size)
                good_splits.extend(child_splits)
            else:
                good_splits.extend([s[i : i + chunk_size] for i in range(0, len(s), chunk_size)])

    return sep, good_splits


def _merge_splits_with_overlap(
    splits: List[str],
    separator: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Merge atomic splits into chunks up to `chunk_size`, maintaining `overlap`.
    """
    if not splits:
        return []

    chunks: List[str] = []
    curr: List[str] = []
    curr_len = 0

    for s in splits:
        s_len = len(s)
        join_len = len(separator) if curr else 0

        if curr_len + join_len + s_len <= chunk_size:
            curr.append(s)
            curr_len += join_len + s_len
        else:
            if curr:
                chunk_str = separator.join(curr)
                if chunk_str.strip():
                    chunks.append(chunk_str)
                # Overlap: keep trailing pieces from curr that fit within overlap
                while curr and (
                    curr_len > overlap
                    or (curr_len + join_len + s_len > chunk_size and len(curr) > 1)
                ):
                    popped = curr.pop(0)
                    curr_len -= len(popped) + (len(separator) if curr else 0)

            curr.append(s)
            curr_len += s_len + (len(separator) if len(curr) > 1 else 0)

    if curr:
        chunk_str = separator.join(curr)
        if chunk_str.strip():
            chunks.append(chunk_str)

    return chunks


def _build_section_index(
    document: ParsedDocument,
) -> List[Tuple[int, int, Section]]:
    """Build a list of (start_char, end_char, section) tuples."""
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

    Tries paragraph -> line -> sentence -> word -> character splits in
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
        if not text or not text.strip():
            logger.debug(
                "RecursiveChunker: document '{}' is empty, returning no chunks.",
                document.filename,
            )
            return []

        section_index = _build_section_index(document)
        sep, splits = _split_text_hierarchical(text, _SEPARATORS, self.config.chunk_size)
        raw_chunks = _merge_splits_with_overlap(
            splits, sep, self.config.chunk_size, self.config.overlap
        )

        chunks: List[Chunk] = []
        search_pos = 0

        for idx, content in enumerate(raw_chunks):
            if not content.strip():
                continue

            # Find exact character offsets in original content
            found_pos = text.find(content, search_pos)
            if found_pos != -1:
                start = found_pos
                end = found_pos + len(content)
                search_pos = max(0, start + 1)
            else:
                # Fallback in case of subtle substring alignment
                start = search_pos
                end = min(start + len(content), len(text))
                search_pos = end

            section_id, section_heading = _resolve_section(section_index, start)
            page_numbers = _resolve_page_numbers(document, section_index, start, end)

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
                        "primary_separator": sep,
                    },
                )
            )

        logger.info(
            "RecursiveChunker: '{}' -> {} chunk(s) (size={}, overlap={})",
            document.filename,
            len(chunks),
            self.config.chunk_size,
            self.config.overlap,
        )
        return chunks
