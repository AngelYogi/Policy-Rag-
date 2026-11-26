"""
tests/chunking/test_chunk_metadata.py
=====================================
Unit tests for Chunk metadata preservation and deterministic ID generation.
"""

from __future__ import annotations

import pytest

from src.chunking.config import ChunkingConfig
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.models import _make_chunk_id
from src.chunking.recursive import RecursiveChunker
from src.chunking.section_based import SectionBasedChunker
from src.ingestion.models import FileType, ParsedDocument, Section


class TestChunkMetadata:
    def test_deterministic_chunk_ids(self):
        # Running ID generation multiple times with identical parameters must yield identical IDs
        id1 = _make_chunk_id("doc-123", "sec-456", "recursive", 500, 50, 0)
        id2 = _make_chunk_id("doc-123", "sec-456", "recursive", 500, 50, 0)
        assert id1 == id2
        assert len(id1) == 16

        # Different index must produce different ID
        id3 = _make_chunk_id("doc-123", "sec-456", "recursive", 500, 50, 1)
        assert id1 != id3

        # Different config must produce different ID
        id4 = _make_chunk_id("doc-123", "sec-456", "recursive", 700, 100, 0)
        assert id1 != id4

    def test_metadata_preservation_across_strategies(self):
        doc = ParsedDocument(
            document_id="doc-meta-test",
            filename="compliance.pdf",
            file_path="/docs/compliance.pdf",
            file_type=FileType.PDF,
            title="Compliance Policy",
            content="Page 1 content.\n\nPage 2 content.",
            sections=[
                Section(heading="P1", content="Page 1 content.", page_number=1, section_id="p1-id"),
                Section(heading="P2", content="Page 2 content.", page_number=2, section_id=None),
            ],
            page_count=2,
        )

        for strategy in ["fixed", "recursive", "section"]:
            config = ChunkingConfig(strategy=strategy, chunk_size=150, overlap=20)
            if strategy == "fixed":
                chunker = FixedSizeChunker(config)
            elif strategy == "recursive":
                chunker = RecursiveChunker(config)
            else:
                chunker = SectionBasedChunker(config)

            chunks = chunker.chunk_document(doc)
            assert len(chunks) >= 1
            for c in chunks:
                assert c.document_id == "doc-meta-test"
                assert c.filename == "compliance.pdf"
                assert c.file_type == FileType.PDF
                assert c.metadata["chunking_strategy"] == strategy
                assert c.metadata["chunk_size_config"] == 150
                assert c.metadata["overlap_config"] == 20

    def test_pdf_page_reference_preservation(self):
        doc = ParsedDocument(
            document_id="doc-pdf-pages",
            filename="handbook.pdf",
            file_path="/handbook.pdf",
            file_type=FileType.PDF,
            title="Handbook",
            content="Page 1 details.\n\nPage 2 details.\n\nPage 3 details.",
            sections=[
                Section(heading="Intro", content="Page 1 details.", page_number=1),
                Section(heading="Middle", content="Page 2 details.", page_number=2),
                Section(heading="End", content="Page 3 details.", page_number=3),
            ],
            page_count=3,
        )

        chunker = SectionBasedChunker(ChunkingConfig(strategy="section", chunk_size=200, overlap=20))
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 3
        assert chunks[0].page_numbers == [1]
        assert chunks[0].page_number == 1
        assert chunks[1].page_numbers == [2]
        assert chunks[1].page_number == 2
        assert chunks[2].page_numbers == [3]
        assert chunks[2].page_number == 3

    def test_documents_without_section_ids(self):
        doc = ParsedDocument(
            document_id="doc-no-id",
            filename="notes.txt",
            file_path="/notes.txt",
            file_type=FileType.TEXT,
            title="Notes",
            content="Plain unformatted notes with no explicit section IDs.",
            sections=[Section(heading="Notes", content="Plain unformatted notes with no explicit section IDs.", section_id=None)],
        )

        chunker = FixedSizeChunker(ChunkingConfig(strategy="fixed", chunk_size=200, overlap=20))
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].section_id is None
        assert not chunks[0].has_section_id
