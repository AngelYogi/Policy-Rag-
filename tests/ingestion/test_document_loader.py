"""
tests/ingestion/test_document_loader.py
========================================
Unit tests for DocumentLoader and IngestionResult.

All tests use temporary directories and synthetic file content.
No external files, PDFs, or APIs are needed.
"""

from __future__ import annotations

import pytest

from src.ingestion.document_loader import DocumentLoader, IngestionResult
from src.ingestion.models import FileType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
# Test Document

## Section A

Content for section A.

## Section B

Content for section B.
"""

SAMPLE_TXT = """\
OVERVIEW

This is a plain text document.
It has some content.
"""

SAMPLE_TXT_2 = """\
APPENDIX

Additional reference material.
"""


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------

class TestDocumentLoaderSingleFile:
    def test_load_markdown_file(self, tmp_path):
        f = tmp_path / "policy.md"
        f.write_text(SAMPLE_MD, encoding="utf-8")
        loader = DocumentLoader()
        doc = loader.load_file(f)
        assert doc.file_type == FileType.MARKDOWN
        assert doc.filename == "policy.md"

    def test_load_text_file(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text(SAMPLE_TXT, encoding="utf-8")
        loader = DocumentLoader()
        doc = loader.load_file(f)
        assert doc.file_type == FileType.TEXT

    def test_load_unsupported_raises_value_error(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n", encoding="utf-8")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file(f)

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file(tmp_path / "ghost.md")


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------

class TestDocumentLoaderDirectory:
    def test_loads_all_supported_files(self, tmp_path):
        (tmp_path / "a.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "b.txt").write_text(SAMPLE_TXT, encoding="utf-8")
        (tmp_path / "c.txt").write_text(SAMPLE_TXT_2, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.success_count == 3

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (tmp_path / "script.py").write_text("print('hello')\n", encoding="utf-8")
        (tmp_path / "valid.md").write_text(SAMPLE_MD, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.success_count == 1
        assert len(result.skipped_files) == 2

    def test_valid_files_load_despite_one_invalid(self, tmp_path):
        (tmp_path / "good.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "good2.txt").write_text(SAMPLE_TXT, encoding="utf-8")
        # Create a file with an unsupported extension (not an invalid parse,
        # but exercise the skip path alongside valid ones)
        (tmp_path / "ignore.xyz").write_text("garbage\n", encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.success_count == 2
        # The xyz file should be skipped, not failed
        assert "ignore.xyz" in " ".join(result.skipped_files)

    def test_recursive_discovery(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.md").write_text(SAMPLE_MD, encoding="utf-8")
        (subdir / "sub.txt").write_text(SAMPLE_TXT, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path, recursive=True)
        assert result.success_count == 2

    def test_non_recursive_does_not_descend(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.md").write_text(SAMPLE_MD, encoding="utf-8")
        (subdir / "nested.txt").write_text(SAMPLE_TXT, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path, recursive=False)
        assert result.success_count == 1

    def test_empty_directory_returns_zero_documents(self, tmp_path):
        result = DocumentLoader().load_directory(tmp_path)
        assert result.success_count == 0
        assert result.failed_count == 0

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory(tmp_path / "does_not_exist")

    def test_path_that_is_a_file_raises_not_a_directory(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text(SAMPLE_MD, encoding="utf-8")
        loader = DocumentLoader()
        with pytest.raises(NotADirectoryError):
            loader.load_directory(f)

    def test_hidden_files_are_ignored(self, tmp_path):
        (tmp_path / ".hidden.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "visible.md").write_text(SAMPLE_MD, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.success_count == 1


# ---------------------------------------------------------------------------
# IngestionResult statistics
# ---------------------------------------------------------------------------

class TestIngestionResult:
    def test_total_sections_aggregated(self, tmp_path):
        (tmp_path / "a.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "b.txt").write_text(SAMPLE_TXT, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.total_sections >= 2  # at least one section per doc

    def test_empty_document_count(self, tmp_path):
        (tmp_path / "empty.md").write_text("", encoding="utf-8")
        (tmp_path / "real.md").write_text(SAMPLE_MD, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.empty_document_count == 1

    def test_duration_is_non_negative(self, tmp_path):
        (tmp_path / "doc.md").write_text(SAMPLE_MD, encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.duration_seconds >= 0.0

    def test_total_discovered_matches_sum(self, tmp_path):
        (tmp_path / "a.md").write_text(SAMPLE_MD, encoding="utf-8")
        (tmp_path / "b.csv").write_text("a,b\n", encoding="utf-8")
        result = DocumentLoader().load_directory(tmp_path)
        assert result.total_discovered == result.success_count + result.failed_count + len(result.skipped_files)


# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

class TestDocumentLoaderExtensions:
    def test_default_extensions_registered(self):
        loader = DocumentLoader()
        exts = loader.supported_extensions
        assert "md" in exts
        assert "txt" in exts
        assert "pdf" in exts

    def test_markdown_extension_reported(self):
        loader = DocumentLoader()
        assert "markdown" in loader.supported_extensions
