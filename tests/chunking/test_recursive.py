"""
tests/chunking/test_recursive.py
================================
Unit tests for RecursiveChunker (Strategy B).
"""

from __future__ import annotations

import pytest

from src.chunking.config import ChunkingConfig
from src.chunking.recursive import RecursiveChunker
from src.ingestion.models import FileType, ParsedDocument, Section


def _create_doc(content: str, filename: str = "test.md") -> ParsedDocument:
    return ParsedDocument(
        document_id="doc-rec-1",
        filename=filename,
        file_path=f"/{filename}",
        file_type=FileType.MARKDOWN,
        title="Recursive Test",
        content=content,
        sections=[Section(heading="Section 1", content=content, section_id="sec-1")],
    )


class TestRecursiveChunker:
    def test_recursive_paragraph_splitting(self):
        p1 = "Paragraph 1: " + "A" * 80 + "."
        p2 = "Paragraph 2: " + "B" * 80 + "."
        p3 = "Paragraph 3: " + "C" * 80 + "."
        content = f"{p1}\n\n{p2}\n\n{p3}"
        doc = _create_doc(content)

        # chunk_size allows one paragraph (~100 chars) but not two (~200 chars)
        config = ChunkingConfig(strategy="recursive", chunk_size=120, overlap=20)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 3
        assert "Paragraph 1" in chunks[0].content
        assert "Paragraph 2" in chunks[1].content
        assert "Paragraph 3" in chunks[2].content

    def test_recursive_sentence_splitting(self):
        s1 = "This is the first sentence about policies."
        s2 = "This is the second sentence about security."
        s3 = "This is the third sentence about compliance."
        content = f"{s1} {s2} {s3}"
        doc = _create_doc(content)

        # chunk_size allows roughly one sentence
        config = ChunkingConfig(strategy="recursive", chunk_size=55, overlap=10)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 3
        assert "first sentence" in chunks[0].content
        assert "second sentence" in chunks[1].content

    def test_recursive_word_and_char_fallback(self):
        # A single word that exceeds chunk_size
        long_word = "Supercalifragilisticexpialidocious" * 4  # ~136 chars
        doc = _create_doc(long_word)
        config = ChunkingConfig(strategy="recursive", chunk_size=40, overlap=10)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 3
        # Ensure pieces were created without crashing
        total_len = sum(len(c.content) for c in chunks)
        assert total_len >= len(long_word)

    def test_empty_and_whitespace_documents(self):
        chunker = RecursiveChunker(ChunkingConfig(strategy="recursive", chunk_size=100, overlap=20))
        assert chunker.chunk_document(_create_doc("")) == []
        assert chunker.chunk_document(_create_doc("   \n\n\n\t  ")) == []

    def test_no_unintended_text_loss(self):
        content = (
            "# Section 1\n\n"
            "Data retention is vital for compliance. Financial records must be kept for seven years.\n\n"
            "Security audits occur quarterly. Access keys must be rotated every ninety days.\n\n"
            "Unanswerable questions must be detected and rejected gracefully."
        )
        doc = _create_doc(content)
        config = ChunkingConfig(strategy="recursive", chunk_size=100, overlap=25)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

        # Check all key keywords appear in chunks
        keywords = [
            "retention",
            "compliance",
            "Financial",
            "Security",
            "audits",
            "quarterly",
            "keys",
            "rotated",
            "Unanswerable",
            "gracefully",
        ]
        for kw in keywords:
            found = any(kw in c.content for c in chunks)
            assert found, f"Keyword '{kw}' lost during recursive chunking."
