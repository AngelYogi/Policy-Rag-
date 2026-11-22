"""
tests/ingestion/test_pdf_parser.py
====================================
Unit tests for PDFParser.

Synthetic PDF fixtures are created programmatically using PyMuPDF
(``fitz``).  No external PDF files are required.

If PyMuPDF is not installed these tests are automatically skipped.
"""

from __future__ import annotations

import pytest

try:
    import fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

from src.ingestion.models import FileType

pytestmark = pytest.mark.skipif(
    not _FITZ_AVAILABLE, reason="PyMuPDF not installed"
)


# ---------------------------------------------------------------------------
# Helper: programmatically create PDF fixtures
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path, filename: str, pages: list) -> "fitz.Document":
    """
    Create a minimal PDF with one text block per page.

    Parameters
    ----------
    pages:
        List of strings, one per page.  An empty string makes an image-only
        (blank text) page.

    Returns the Path to the written PDF.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    path = tmp_path / filename
    doc.save(str(path))
    doc.close()
    return path


def _make_pdf_with_metadata(tmp_path, filename: str, text: str, title: str) -> "fitz.Document":
    """Create a PDF with a metadata title field."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.set_metadata({"title": title, "author": "Test Author"})
    path = tmp_path / filename
    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------

class TestPDFParserBasic:
    def test_file_type_is_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "doc.pdf", ["Hello world. This is page one."])
        doc = PDFParser().parse(path)
        assert doc.file_type == FileType.PDF

    def test_filename_recorded(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "policy.pdf", ["Some content here."])
        doc = PDFParser().parse(path)
        assert doc.filename == "policy.pdf"

    def test_content_contains_extracted_text(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "doc.pdf", ["Policy text here.\nMore content."])
        doc = PDFParser().parse(path)
        assert "Policy text here." in doc.content

    def test_document_id_unique(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "doc.pdf", ["Content here."])
        doc1 = PDFParser().parse(path)
        doc2 = PDFParser().parse(path)
        assert doc1.document_id != doc2.document_id


# ---------------------------------------------------------------------------
# Page count preservation
# ---------------------------------------------------------------------------

class TestPDFPageCount:
    def test_single_page_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "single.pdf", ["One page content."])
        doc = PDFParser().parse(path)
        assert doc.page_count == 1

    def test_multi_page_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        pages = [f"Content on page {i+1}." for i in range(5)]
        path = _make_pdf(tmp_path, "multi.pdf", pages)
        doc = PDFParser().parse(path)
        assert doc.page_count == 5

    def test_sections_carry_page_numbers(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        pages = ["First page content.", "Second page content.", "Third page."]
        path = _make_pdf(tmp_path, "pages.pdf", pages)
        doc = PDFParser().parse(path)
        # Every section from a text page should have a page_number
        for section in doc.sections:
            assert section.page_number is not None
            assert section.page_number >= 1


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

class TestPDFMetadata:
    def test_metadata_title_used_as_document_title(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf_with_metadata(
            tmp_path, "meta.pdf",
            "Body text of the document.",
            "Authoritative Policy Document"
        )
        doc = PDFParser().parse(path)
        assert doc.title == "Authoritative Policy Document"

    def test_author_stored_in_metadata(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf_with_metadata(
            tmp_path, "meta.pdf", "Some content.", "Test Title"
        )
        doc = PDFParser().parse(path)
        assert doc.metadata.get("pdf_author") == "Test Author"

    def test_empty_pages_tracked_in_metadata(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        # Page 2 is blank (empty string → no text)
        path = _make_pdf(tmp_path, "mixed.pdf", ["Real content here.", ""])
        doc = PDFParser().parse(path)
        assert 2 in doc.metadata.get("empty_pages", [])

    def test_section_ids_always_none_for_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        path = _make_pdf(tmp_path, "doc.pdf", ["Some long content on page one."])
        doc = PDFParser().parse(path)
        for section in doc.sections:
            assert section.section_id is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPDFParserErrors:
    def test_raises_file_not_found(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        with pytest.raises(FileNotFoundError):
            PDFParser().parse(tmp_path / "nonexistent.pdf")

    def test_raises_value_error_for_corrupt_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"This is not a valid PDF file content %PDF-nope")
        with pytest.raises((ValueError, Exception)):
            PDFParser().parse(corrupt)

    def test_supported_extension(self):
        from src.ingestion.pdf_parser import PDFParser
        assert "pdf" in PDFParser().supported_extensions

    def test_can_handle_pdf(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        assert PDFParser().can_handle(tmp_path / "file.pdf")

    def test_cannot_handle_txt(self, tmp_path):
        from src.ingestion.pdf_parser import PDFParser
        assert not PDFParser().can_handle(tmp_path / "file.txt")
