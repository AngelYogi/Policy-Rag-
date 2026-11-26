"""
tests/chunking/test_section_based.py
====================================
Unit tests for SectionBasedChunker (Strategy C).
"""

from __future__ import annotations

import pytest

from src.chunking.config import ChunkingConfig
from src.chunking.section_based import SectionBasedChunker
from src.ingestion.models import FileType, ParsedDocument, Section


def _create_multi_section_doc(sections: list[Section]) -> ParsedDocument:
    full_content = "\n\n".join(s.content for s in sections if s.content)
    return ParsedDocument(
        document_id="doc-sec-1",
        filename="policy.md",
        file_path="/policy.md",
        file_type=FileType.MARKDOWN,
        title="Multi-section Policy",
        content=full_content,
        sections=sections,
    )


class TestSectionBasedChunker:
    def test_section_based_independent_chunking(self):
        s1 = Section(heading="Scope", content="Content of scope section.", section_id="scope-id")
        s2 = Section(heading="Policy", content="Content of policy section.", section_id="policy-id")
        doc = _create_multi_section_doc([s1, s2])

        config = ChunkingConfig(strategy="section", chunk_size=200, overlap=30)
        chunker = SectionBasedChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 2

        assert chunks[0].section_id == "scope-id"
        assert chunks[0].section_heading == "Scope"
        assert "Content of scope" in chunks[0].content

        assert chunks[1].section_id == "policy-id"
        assert chunks[1].section_heading == "Policy"
        assert "Content of policy" in chunks[1].content

    def test_never_combines_unrelated_sections(self):
        s1 = Section(heading="Confidential A", content="Secret A information.", section_id="sec-a")
        s2 = Section(heading="Public B", content="Public B information.", section_id="sec-b")
        doc = _create_multi_section_doc([s1, s2])

        # Large chunk size that could easily hold both sections if combined
        config = ChunkingConfig(strategy="section", chunk_size=500, overlap=50)
        chunker = SectionBasedChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 2
        assert "Public B" not in chunks[0].content
        assert "Secret A" not in chunks[1].content

    def test_oversized_section_handling(self):
        large_text = (
            "Paragraph one is long and detailed. " * 5
            + "\n\n"
            + "Paragraph two is equally detailed and long. " * 5
        )
        s1 = Section(heading="Detailed Section", content=large_text, section_id="detailed-id")
        doc = _create_multi_section_doc([s1])

        config = ChunkingConfig(strategy="section", chunk_size=150, overlap=30)
        chunker = SectionBasedChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        # All sub-chunks must preserve the section heading and ID
        for c in chunks:
            assert c.section_id == "detailed-id"
            assert c.section_heading == "Detailed Section"

    def test_empty_sections_handled_gracefully(self):
        s1 = Section(heading="Empty Section", content="", section_id="empty-id")
        s2 = Section(heading="Valid Section", content="Valid content here.", section_id="valid-id")
        doc = _create_multi_section_doc([s1, s2])

        config = ChunkingConfig(strategy="section", chunk_size=200, overlap=30)
        chunker = SectionBasedChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].section_id == "valid-id"

    def test_fallback_when_document_has_no_sections(self):
        doc = ParsedDocument(
            document_id="doc-no-sec",
            filename="raw.txt",
            file_path="/raw.txt",
            file_type=FileType.TEXT,
            title="Raw Doc",
            content="Line one of raw doc.\nLine two of raw doc.",
            sections=[],
        )
        config = ChunkingConfig(strategy="section", chunk_size=100, overlap=20)
        chunker = SectionBasedChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].section_id is None
