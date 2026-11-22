"""
src/chunking/section_based.py
==============================
Strategy C — Section-aware chunking.

Algorithm
---------
Each ``Section`` from Stage 2 is treated as an independent unit:

1. If the section content is ≤ ``chunk_size``, it becomes a single chunk.
2. If the section content exceeds ``chunk_size``, it is split further using
   the recursive strategy (same separator hierarchy as RecursiveChunker) so
   that semantic boundaries are still respected within oversized sections.

Content from one section is NEVER merged with content from another section.
The section heading and section_id are preserved on every chunk produced
from that section.

Page numbers
------------
Every chunk inherits the ``page_number`` of its parent section.  When
overflow splitting creates multiple chunks from one section they all share
that same page number (the page where the section begins).  This is an
approximation: PyMuPDF sections are per-page so a section never spans
multiple pages with this parser.

Metadata preserved
------------------
- section_id (from Section, never invented)
- section_heading
- page_numbers
- chunking_strategy, chunk_size_config, overlap_config
- section_index (position of the source section within the document)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from src.chunking.base import BaseChunker
from src.chunking.config import ChunkingConfig
from src.chunking.models import Chunk, _make_chunk_id
from src.chunking.recursive import _apply_overlap, _split_text_recursive, _SEPARATORS
from src.ingestion.models import ParsedDocument, Section


class SectionBasedChunker(BaseChunker):
    """
    Section-aware chunker — each section is chunked independently.

    Sections that exceed ``chunk_size`` are split with the recursive
    strategy so word and sentence boundaries are still respected.
    """

    @property
    def strategy_name(self) -> str:
        return "section"

    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Produce chunks from ``document``, one independent batch per section.

        Parameters
        ----------
        document:
            ``ParsedDocument`` from Stage 2.

        Returns
        -------
        List[Chunk]
            Chunks in document order.  Empty for empty documents.
        """
        if not document.content.strip():
            logger.debug(
                "SectionBasedChunker: document '{}' is empty, returning no chunks.",
                document.filename,
            )
            return []

        if not document.sections:
            # No sections from the parser — treat the full content as one section
            return self._chunk_fallback(document)

        chunks: List[Chunk] = []
        global_index = 0

        for sec_idx, section in enumerate(document.sections):
            section_chunks = self._chunk_section(
                document=document,
                section=section,
                sec_idx=sec_idx,
                global_index_start=global_index,
            )
            global_index += len(section_chunks)
            chunks.extend(section_chunks)

        logger.info(
            "SectionBasedChunker: '{}' → {} chunk(s) from {} section(s) "
            "(size={}, overlap={})",
            document.filename,
            len(chunks),
            len(document.sections),
            self.config.chunk_size,
            self.config.overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_section(
        self,
        document: ParsedDocument,
        section: Section,
        sec_idx: int,
        global_index_start: int,
    ) -> List[Chunk]:
        """
        Chunk a single section and return its chunks.

        Empty sections produce no chunks.
        """
        text = section.content.strip()
        if not text:
            return []

        page_numbers: List[int] = (
            [section.page_number] if section.page_number is not None else []
        )

        if len(text) <= self.config.chunk_size:
            # Section fits in a single chunk
            pieces_with_offsets: List[Tuple[str, int, int]] = [(text, 0, len(text))]
        else:
            # Split using recursive strategy
            raw_pieces = _split_text_recursive(
                text, self.config.chunk_size, _SEPARATORS
            )
            raw_pieces = [p for p in raw_pieces if p.strip()]
            pieces_with_offsets = _apply_overlap(raw_pieces, self.config.overlap)

        result: List[Chunk] = []
        for local_idx, (content, start, end) in enumerate(pieces_with_offsets):
            if len(content.strip()) < self.config.min_chunk_length:
                continue

            global_idx = global_index_start + local_idx
            chunk_id = _make_chunk_id(
                document_id=document.document_id,
                section_id=section.section_id,
                strategy=self.strategy_name,
                chunk_size=self.config.chunk_size,
                overlap=self.config.overlap,
                chunk_index=global_idx,
            )

            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    section_id=section.section_id,
                    section_heading=section.heading,
                    chunk_index=global_idx,
                    content=content,
                    start_char=start,
                    end_char=end,
                    page_numbers=page_numbers,
                    metadata={
                        **self._base_metadata(),
                        "section_index": sec_idx,
                        "section_local_chunk_index": local_idx,
                    },
                )
            )

        return result

    def _chunk_fallback(self, document: ParsedDocument) -> List[Chunk]:
        """
        When a document has no sections, treat the full content as one
        anonymous section and apply recursive splitting.
        """
        text = document.content.strip()
        raw_pieces = _split_text_recursive(text, self.config.chunk_size, _SEPARATORS)
        raw_pieces = [p for p in raw_pieces if p.strip()]
        overlapped = _apply_overlap(raw_pieces, self.config.overlap)

        chunks: List[Chunk] = []
        for idx, (content, start, end) in enumerate(overlapped):
            if len(content.strip()) < self.config.min_chunk_length:
                continue
            chunk_id = _make_chunk_id(
                document_id=document.document_id,
                section_id=None,
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
                    section_id=None,
                    section_heading="",
                    chunk_index=idx,
                    content=content,
                    start_char=start,
                    end_char=end,
                    page_numbers=[],
                    metadata={
                        **self._base_metadata(),
                        "section_index": None,
                        "note": "no_sections_fallback",
                    },
                )
            )

        logger.info(
            "SectionBasedChunker (fallback): '{}' → {} chunk(s)",
            document.filename,
            len(chunks),
        )
        return chunks
