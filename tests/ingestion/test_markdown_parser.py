"""
tests/ingestion/test_markdown_parser.py
=======================================
Unit tests for MarkdownParser.

All tests use synthetic in-memory fixtures written to a tmp_path.
No external files or APIs are required.
"""

from __future__ import annotations

import pytest

from src.ingestion.markdown_parser import MarkdownParser
from src.ingestion.models import FileType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_MD = """\
# My Document Title

This is the preamble paragraph.

## Section One

Content for section one.
More content here.

## Section Two {#sec-two}

Content for section two.

### Subsection 2.1

Nested content.
"""

EMPTY_MD = ""

NO_HEADINGS_MD = """\
Just some plain text.
No headings at all.
Multiple lines.
"""

MULTI_H1_MD = """\
# First Title

Intro text.

# Second Title

More text.
"""

SECTION_ID_ONLY_MD = """\
# Introduction {#intro}

Some content here.

## Details {#details-section}

Detail content.
"""

PREAMBLE_ONLY_MD = """\
Text before any heading.
More preamble text.
"""


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

class TestMarkdownTitleExtraction:
    def test_extracts_first_h1_as_title(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.title == "My Document Title"

    def test_no_headings_title_is_none(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(NO_HEADINGS_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.title is None

    def test_multiple_h1_uses_first(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(MULTI_H1_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.title == "First Title"

    def test_empty_file_title_is_none(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text(EMPTY_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.title is None


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

class TestMarkdownSectionExtraction:
    def test_correct_number_of_sections(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        # Sections: H1 (My Document Title), H2 (Section One), H2 (Section Two), H3 (Subsection)
        headings = [s.heading for s in doc.sections if s.heading]
        assert "My Document Title" in headings
        assert "Section One" in headings
        assert "Section Two" in headings
        assert "Subsection 2.1" in headings

    def test_section_content_associated_correctly(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        section_one = next(s for s in doc.sections if s.heading == "Section One")
        assert "Content for section one." in section_one.content
        assert "More content here." in section_one.content

    def test_preamble_captured_before_first_heading(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        # The H1 section should contain the preamble text
        h1_section = next(s for s in doc.sections if s.heading == "My Document Title")
        assert "preamble paragraph" in h1_section.content

    def test_no_headings_produces_single_section_with_empty_heading(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(NO_HEADINGS_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert len(doc.sections) == 1
        assert doc.sections[0].heading == ""
        assert "plain text" in doc.sections[0].content

    def test_preamble_only_has_no_heading(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(PREAMBLE_ONLY_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert all(s.heading == "" for s in doc.sections)

    def test_empty_file_produces_no_sections(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text(EMPTY_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.sections == []


# ---------------------------------------------------------------------------
# Section ID preservation
# ---------------------------------------------------------------------------

class TestMarkdownSectionIDs:
    def test_explicit_section_id_extracted(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        section_two = next(s for s in doc.sections if s.heading == "Section Two")
        assert section_two.section_id == "sec-two"

    def test_heading_without_id_has_none_section_id(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        section_one = next(s for s in doc.sections if s.heading == "Section One")
        assert section_one.section_id is None

    def test_multiple_explicit_ids_all_captured(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SECTION_ID_ONLY_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        intro = next(s for s in doc.sections if s.heading == "Introduction")
        details = next(s for s in doc.sections if s.heading == "Details")
        assert intro.section_id == "intro"
        assert details.section_id == "details-section"

    def test_section_ids_not_invented(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(NO_HEADINGS_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        for section in doc.sections:
            assert section.section_id is None


# ---------------------------------------------------------------------------
# Document model and metadata
# ---------------------------------------------------------------------------

class TestMarkdownDocumentModel:
    def test_file_type_is_markdown(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.file_type == FileType.MARKDOWN

    def test_filename_matches_file(self, tmp_path):
        f = tmp_path / "my_policy.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.filename == "my_policy.md"

    def test_content_preserves_original_text(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.content == SIMPLE_MD

    def test_page_count_is_none_for_markdown(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.page_count is None

    def test_document_id_is_unique(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc1 = MarkdownParser().parse(f)
        doc2 = MarkdownParser().parse(f)
        assert doc1.document_id != doc2.document_id

    def test_is_empty_false_for_non_empty(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text(SIMPLE_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert not doc.is_empty

    def test_is_empty_true_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text(EMPTY_MD, encoding="utf-8")
        doc = MarkdownParser().parse(f)
        assert doc.is_empty


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestMarkdownParserErrors:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MarkdownParser().parse(tmp_path / "nonexistent.md")

    def test_supported_extensions(self):
        parser = MarkdownParser()
        assert "md" in parser.supported_extensions
        assert "markdown" in parser.supported_extensions

    def test_can_handle_md_extension(self, tmp_path):
        parser = MarkdownParser()
        f = tmp_path / "file.md"
        assert parser.can_handle(f)

    def test_cannot_handle_txt_extension(self, tmp_path):
        parser = MarkdownParser()
        f = tmp_path / "file.txt"
        assert not parser.can_handle(f)
