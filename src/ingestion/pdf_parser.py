"""
src/ingestion/pdf_parser.py
============================
Parser for PDF files using PyMuPDF (``fitz``).

Parsing strategy
----------------
* Open the PDF with ``fitz.open``.
* Iterate page by page, extracting text with ``page.get_text("text")``.
* Attempt lightweight heading detection: lines that are significantly shorter
  than the page average and appear to be title-cased or ALL CAPS are
  tagged as headings.
* Pages where no text is extractable (scanned / image-only) are logged as
  warnings.  No OCR is performed; content for those pages is an empty string.
* Preserve page numbers (1-based) in every :class:`Section`.
* Extract document metadata (title, author, creation date) from PDF info dict.

Dependencies
------------
  PyMuPDF >= 1.25  (imported as ``fitz``)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from src.ingestion.base import BaseParser
from src.ingestion.models import FileType, ParsedDocument, Section

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FITZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Heading detection thresholds
# ---------------------------------------------------------------------------

_MIN_HEADING_LINE_LEN = 3
_MAX_HEADING_LINE_LEN = 150
_HEADING_LENGTH_RATIO = 0.6   # line is "short" if len < ratio * avg_line_len


def _looks_like_heading(line: str, avg_len: float) -> bool:
    """
    Heuristic: a line might be a heading when it is short relative to the
    page average, is not purely punctuation, and is title-cased or ALL CAPS.
    """
    stripped = line.strip()
    if not (_MIN_HEADING_LINE_LEN <= len(stripped) <= _MAX_HEADING_LINE_LEN):
        return False
    if avg_len > 0 and len(stripped) > avg_len * _HEADING_LENGTH_RATIO:
        return False
    is_title = stripped.istitle() or stripped.isupper()
    has_alpha = any(ch.isalpha() for ch in stripped)
    return is_title and has_alpha


def _extract_page_sections(
    page_text: str,
    page_num: int,
) -> Tuple[List[Section], bool]:
    """
    Split a single PDF page's text into sections.

    Returns
    -------
    (sections, has_text)
        ``has_text`` is False when the page yielded no extractable text.
    """
    if not page_text.strip():
        return [], False

    lines = page_text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    avg_len = (
        sum(len(l.strip()) for l in non_empty) / len(non_empty) if non_empty else 0
    )

    sections: List[Section] = []
    current_heading = ""
    current_lines: List[str] = []

    def _flush(heading: str, body: List[str]) -> None:
        content = "\n".join(body).strip()
        if heading or content:
            sections.append(
                Section(
                    heading=heading,
                    content=content,
                    section_id=None,  # PDFs never carry explicit anchor IDs
                    page_number=page_num,
                )
            )

    for line in lines:
        if _looks_like_heading(line, avg_len):
            _flush(current_heading, current_lines)
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    _flush(current_heading, current_lines)
    return sections, True


class PDFParser(BaseParser):
    """Parses PDF files into a structured :class:`ParsedDocument` using PyMuPDF."""

    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF file page by page.

        Scanned / image-only pages are handled gracefully: a warning is
        logged and the page contributes an empty string to the content.
        No OCR is performed.

        Parameters
        ----------
        file_path:
            Path to the ``.pdf`` file.

        Returns
        -------
        ParsedDocument

        Raises
        ------
        ImportError
            When PyMuPDF is not installed.
        FileNotFoundError
            When the file does not exist.
        PermissionError
            When the file cannot be read.
        ValueError
            When the file is not a valid PDF or is encrypted beyond repair.
        """
        if not _FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. "
                "Install it with: pip install PyMuPDF"
            )

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.debug("Parsing PDF file: {}", file_path)

        try:
            doc = fitz.open(str(file_path))
        except PermissionError as exc:
            raise PermissionError(f"Cannot read {file_path}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{file_path}': {exc}") from exc

        try:
            return self._process_document(doc, file_path)
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_document(
        self, doc: "fitz.Document", file_path: Path
    ) -> ParsedDocument:
        """Extract text and structure from an open fitz Document."""
        page_count = doc.page_count
        pdf_meta = doc.metadata or {}

        all_sections: List[Section] = []
        content_parts: List[str] = []
        empty_pages: List[int] = []

        for page_index in range(page_count):
            page_num = page_index + 1
            page = doc[page_index]

            try:
                page_text = page.get_text("text")
            except Exception as exc:
                logger.warning(
                    "Could not extract text from page {} of '{}': {}",
                    page_num, file_path.name, exc,
                )
                page_text = ""

            page_sections, has_text = _extract_page_sections(page_text, page_num)

            if not has_text:
                logger.warning(
                    "Page {} of '{}' has no extractable text (may be scanned/image).",
                    page_num, file_path.name,
                )
                empty_pages.append(page_num)
            else:
                all_sections.extend(page_sections)
                content_parts.append(page_text)

        full_content = "\n\n".join(content_parts)
        title = self._extract_title(pdf_meta, all_sections)

        metadata: Dict = {
            "pdf_author": pdf_meta.get("author", ""),
            "pdf_creator": pdf_meta.get("creator", ""),
            "pdf_creation_date": pdf_meta.get("creationDate", ""),
            "pdf_subject": pdf_meta.get("subject", ""),
            "empty_pages": empty_pages,
        }

        doc_obj = ParsedDocument(
            document_id=str(uuid.uuid4()),
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=FileType.PDF,
            title=title,
            content=full_content,
            sections=all_sections,
            page_count=page_count,
            metadata=metadata,
        )

        logger.info(
            "Parsed PDF '{}': {} page(s), {} section(s), {} empty page(s), title={!r}",
            file_path.name, page_count, len(all_sections), len(empty_pages), title,
        )
        return doc_obj

    def _extract_title(
        self, pdf_meta: Dict, sections: List[Section]
    ) -> Optional[str]:
        """
        Prefer the PDF metadata title field.  Fall back to the first detected
        heading, then return None when nothing is determinable.
        """
        meta_title = (pdf_meta.get("title") or "").strip()
        if meta_title:
            return meta_title
        for section in sections:
            if section.heading:
                return section.heading
        return None
