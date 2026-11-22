"""
tests/ingestion/test_text_parser.py
=====================================
Unit tests for TextParser.

All tests use synthetic in-memory fixtures.
"""

from __future__ import annotations

import pytest

from src.ingestion.text_parser import TextParser
from src.ingestion.models import FileType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_TXT = """\
INTRODUCTION

This is the introduction paragraph.
It spans multiple lines.

METHODOLOGY

Here is the methodology section.
More details about methodology.

CONCLUSION

Final remarks go here.
"""

PLAIN_TXT = """\
This file has no headings at all.
Just regular paragraphs of text.
Nothing special here.
"""

SETEXT_TXT = """\
Main Title
==========

Some content under the main title.

Sub Section
-----------

Content under the sub section.
"""

EMPTY_TXT = ""

SINGLE_LINE_TXT = "Just one line of content."

LATIN1_BYTES = "Caf\xe9 au lait policy document.\nSection content here.\n"


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

class TestTextParserBasic:
    def test_file_type_is_text(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.file_type == FileType.TEXT

    def test_filename_recorded(self, tmp_path):
        f = tmp_path / "policy.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.filename == "policy.txt"

    def test_content_preserved(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.content == SIMPLE_TXT

    def test_page_count_is_none(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.page_count is None

    def test_document_id_unique(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc1 = TextParser().parse(f)
        doc2 = TextParser().parse(f)
        assert doc1.document_id != doc2.document_id


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

class TestTextParserHeadings:
    def test_all_caps_headings_detected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        headings = [s.heading for s in doc.sections if s.heading]
        assert "INTRODUCTION" in headings
        assert "METHODOLOGY" in headings
        assert "CONCLUSION" in headings

    def test_setext_headings_detected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SETEXT_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        headings = [s.heading for s in doc.sections if s.heading]
        assert "Main Title" in headings or "Sub Section" in headings

    def test_no_headings_when_plain_text(self, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_text(PLAIN_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        # All caps heuristic should not fire on mixed-case prose
        for section in doc.sections:
            assert section.section_id is None

    def test_section_ids_never_invented(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        for section in doc.sections:
            assert section.section_id is None


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

class TestTextParserTitle:
    def test_title_from_first_heading(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(SIMPLE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        # First ALL CAPS heading should be the title
        assert doc.title == "INTRODUCTION"

    def test_title_fallback_first_line_when_no_headings(self, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_text(PLAIN_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.title == "This file has no headings at all."

    def test_single_line_file_title_is_that_line(self, tmp_path):
        f = tmp_path / "single.txt"
        f.write_text(SINGLE_LINE_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.title == SINGLE_LINE_TXT

    def test_empty_file_title_is_none(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text(EMPTY_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.title is None


# ---------------------------------------------------------------------------
# Encoding handling
# ---------------------------------------------------------------------------

class TestTextParserEncoding:
    def test_utf8_read_correctly(self, tmp_path):
        content = "Policy for naïve implementations — §3\n"
        f = tmp_path / "utf8.txt"
        f.write_text(content, encoding="utf-8")
        doc = TextParser().parse(f)
        assert "naïve" in doc.content

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "latin1.txt"
        f.write_bytes(LATIN1_BYTES.encode("latin-1"))
        # Should not raise; falls back to latin-1
        doc = TextParser().parse(f)
        assert doc.metadata["encoding"] == "latin-1"
        assert len(doc.content) > 0


# ---------------------------------------------------------------------------
# Empty and edge cases
# ---------------------------------------------------------------------------

class TestTextParserEdgeCases:
    def test_empty_file_is_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text(EMPTY_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.is_empty

    def test_empty_file_no_sections(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text(EMPTY_TXT, encoding="utf-8")
        doc = TextParser().parse(f)
        assert doc.sections == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestTextParserErrors:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TextParser().parse(tmp_path / "nonexistent.txt")

    def test_supported_extensions(self):
        parser = TextParser()
        assert "txt" in parser.supported_extensions

    def test_can_handle_txt(self, tmp_path):
        assert TextParser().can_handle(tmp_path / "file.txt")

    def test_cannot_handle_pdf(self, tmp_path):
        assert not TextParser().can_handle(tmp_path / "file.pdf")
