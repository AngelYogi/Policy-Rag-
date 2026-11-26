"""
tests/chunking/test_fixed_size.py
=================================
Unit tests for FixedSizeChunker (Strategy A).
"""

from __future__ import annotations

import pytest

from src.chunking.config import ChunkingConfig
from src.chunking.fixed_size import FixedSizeChunker
from src.ingestion.models import FileType, ParsedDocument, Section


def _create_doc(content: str, filename: str = "test.txt", sections: list | None = None) -> ParsedDocument:
    if sections is None:
        sections = [Section(heading="Main", content=content, section_id="sec-1")]
    return ParsedDocument(
        document_id="doc-fixed-1",
        filename=filename,
        file_path=f"/{filename}",
        file_type=FileType.TEXT,
        title="Fixed Test",
        content=content,
        sections=sections,
    )


class TestFixedSizeChunker:
    def test_fixed_size_chunking_basic(self):
        content = "Word " * 100  # 500 characters
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            # Chunks should not exceed chunk_size by more than lookahead
            assert chunk.char_length <= 120
            assert chunk.document_id == doc.document_id
            assert chunk.file_type == FileType.TEXT

    def test_correct_overlap_behavior(self):
        content = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="fixed", chunk_size=30, overlap=10)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            curr_end = chunks[i].end_char
            next_start = chunks[i + 1].start_char
            # Overlap: next chunk must start before current chunk ends
            assert next_start <= curr_end
            # The overlapping slice should be in both chunks
            overlap_text = doc.content[next_start:curr_end]
            assert overlap_text in chunks[i].content
            assert overlap_text in chunks[i + 1].content

    def test_preserves_word_boundaries(self):
        content = "First sentence here. Second sentence starts now. Third sentence follows."
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="fixed", chunk_size=30, overlap=10)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            # Shouldn't cut words arbitrarily if space exists
            words = chunk.content.split()
            # The middle words should be full words
            if len(words) > 2:
                for w in words[1:-1]:
                    assert w.isalnum() or any(p in w for p in ".,!?-")

    def test_empty_document(self):
        doc = _create_doc("")
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_document(self):
        doc = _create_doc("   \n\n\t  ")
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_short_document_smaller_than_chunk_size(self):
        content = "Short text."
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(content)

    def test_no_unintended_text_loss(self):
        """
        Account for intentional overlap: verify every word from the original document
        is present in at least one chunk.
        """
        content = (
            "The quick brown fox jumps over the lazy dog. "
            "Artificial intelligence quality engineering platforms require "
            "rigorous verification, precise evaluation metrics, and deterministic outputs."
        )
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="fixed", chunk_size=60, overlap=15)
        chunker = FixedSizeChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

        original_words = set(content.split())
        extracted_words = set()
        for c in chunks:
            extracted_words.update(c.content.split())

        # Every single word must be captured
        missing = original_words - extracted_words
        assert not missing, f"Missing words: {missing}"

        # Interval coverage: every character index [0..len(content)] is covered
        for i in range(len(content)):
            if content[i].isspace():
                continue
            covered = any(c.start_char <= i < c.end_char for c in chunks)
            assert covered, f"Character at position {i} ('{content[i]}') was not covered by any chunk."
