"""
tests/chunking/test_chunker.py
==============================
Unit tests for configuration validation, presets, and the central chunker dispatcher.
"""

from __future__ import annotations

import pytest

from src.chunking.chunker import chunk_document, chunk_documents, get_chunker
from src.chunking.config import (
    ALL_PRESETS,
    PRESET_LARGE,
    PRESET_MEDIUM,
    PRESET_SMALL,
    ChunkingConfig,
)
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.recursive import RecursiveChunker
from src.chunking.section_based import SectionBasedChunker
from src.ingestion.models import FileType, ParsedDocument, Section


def _create_sample_doc(doc_id: str, filename: str) -> ParsedDocument:
    return ParsedDocument(
        document_id=doc_id,
        filename=filename,
        file_path=f"/{filename}",
        file_type=FileType.MARKDOWN,
        title="Title",
        content="First paragraph for testing.\n\nSecond paragraph for testing.",
        sections=[
            Section(heading="S1", content="First paragraph for testing.", section_id="s1"),
            Section(heading="S2", content="Second paragraph for testing.", section_id="s2"),
        ],
    )


class TestChunkerConfig:
    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=0, overlap=0)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            ChunkingConfig(chunk_size=-10, overlap=0)

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            ChunkingConfig(chunk_size=100, overlap=-5)

    def test_invalid_overlap_greater_or_equal_chunk_size(self):
        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=100)
        with pytest.raises(ValueError, match="overlap .* must be less than chunk_size"):
            ChunkingConfig(chunk_size=100, overlap=150)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ChunkingConfig(strategy="unsupported")

    def test_presets_exist_and_are_valid(self):
        assert PRESET_SMALL.chunk_size == 350
        assert PRESET_SMALL.overlap == 50

        assert PRESET_MEDIUM.chunk_size == 700
        assert PRESET_MEDIUM.overlap == 100

        assert PRESET_LARGE.chunk_size == 1000
        assert PRESET_LARGE.overlap == 150

        assert "small" in ALL_PRESETS
        assert "medium" in ALL_PRESETS
        assert "large" in ALL_PRESETS


class TestChunkerDispatcher:
    def test_get_chunker_instantiation(self):
        c_fixed = get_chunker(ChunkingConfig(strategy="fixed", chunk_size=200, overlap=20))
        assert isinstance(c_fixed, FixedSizeChunker)

        c_rec = get_chunker(ChunkingConfig(strategy="recursive", chunk_size=200, overlap=20))
        assert isinstance(c_rec, RecursiveChunker)

        c_sec = get_chunker(ChunkingConfig(strategy="section", chunk_size=200, overlap=20))
        assert isinstance(c_sec, SectionBasedChunker)

    def test_chunk_document_api(self):
        doc = _create_sample_doc("doc-1", "doc1.md")
        chunks = chunk_document(doc, PRESET_MEDIUM)
        assert len(chunks) >= 1
        assert chunks[0].document_id == "doc-1"

    def test_chunk_documents_batch_api(self):
        docs = [
            _create_sample_doc("doc-1", "doc1.md"),
            _create_sample_doc("doc-2", "doc2.md"),
        ]
        chunks = chunk_documents(docs, PRESET_MEDIUM)
        assert len(chunks) >= 2
        doc_ids = {c.document_id for c in chunks}
        assert doc_ids == {"doc-1", "doc-2"}
