"""
src/ingestion/markdown_parser.py
=================================
Parser for Markdown (``.md``) files.

Parsing strategy
----------------
* Read the file with UTF-8 encoding.
* Walk through lines sequentially building a list of :class:`Section` objects.
* Any line beginning with ``#`` marks a new section heading.
* Explicit section IDs written as ``{#my-id}`` (Pandoc / MyST syntax) are
  extracted from heading lines and stored in ``Section.section_id``.
  IDs are never invented.
* The first H1 (``# Title``) becomes the document title.
* Everything before the first heading is captured as an implicit preamble
  section with an empty heading.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.ingestion.base import BaseParser
from src.ingestion.models import FileType, ParsedDocument, Section

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches Markdown ATX headings: "## Heading Text {#optional-id}"
_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)(?:\s+\{#(?P<id>[^}]+)\})?\s*$"
)

# Fallback: matches bare "{#id}" on its own line or inline
_INLINE_ID_RE = re.compile(r"\{#(?P<id>[^}]+)\}")


def _parse_heading_line(line: str) -> Optional[Tuple[int, str, Optional[str]]]:
    """
    Attempt to parse a Markdown ATX heading line.

    Returns
    -------
    (level, text, section_id) tuple, or ``None`` if the line is not a heading.
    """
    m = _HEADING_RE.match(line)
    if not m:
        return None
    level = len(m.group("hashes"))
    text = m.group("text").strip()
    section_id = m.group("id")  # None when absent — never invented
    return level, text, section_id


class MarkdownParser(BaseParser):
    """Parses Markdown files into a structured :class:`ParsedDocument`."""

    @property
    def supported_extensions(self) -> List[str]:
        return ["md", "markdown"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a Markdown file.

        Parameters
        ----------
        file_path:
            Path to the ``.md`` file.

        Returns
        -------
        ParsedDocument
            Structured representation with sections and optional title.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        PermissionError
            When the file cannot be read.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        logger.debug("Parsing Markdown file: {}", file_path)

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise PermissionError(f"Cannot read {file_path}: {exc}") from exc

        sections = self._extract_sections(raw_text)
        title = self._extract_title(sections)
        full_content = raw_text  # preserve original text

        doc = ParsedDocument(
            document_id=str(uuid.uuid4()),
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=FileType.MARKDOWN,
            title=title,
            content=full_content,
            sections=sections,
            page_count=None,
            metadata={
                "encoding": "utf-8",
                "line_count": raw_text.count("\n") + 1,
                "char_count": len(raw_text),
            },
        )

        logger.info(
            "Parsed Markdown '{}': {} section(s), title={!r}",
            file_path.name,
            len(sections),
            title,
        )
        return doc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> List[Section]:
        """
        Walk through Markdown text line by line and produce ordered sections.

        Content before the first heading becomes a preamble section with an
        empty heading.  Empty sections (no content, no heading) are discarded.
        """
        sections: List[Section] = []
        current_heading: str = ""
        current_id: Optional[str] = None
        current_lines: List[str] = []

        def _flush(heading: str, sid: Optional[str], lines: List[str]) -> None:
            content = "\n".join(lines).strip()
            if heading or content:  # skip truly empty sections
                sections.append(
                    Section(
                        heading=heading,
                        content=content,
                        section_id=sid,
                    )
                )

        for line in text.splitlines():
            parsed = _parse_heading_line(line)
            if parsed is not None:
                _flush(current_heading, current_id, current_lines)
                _level, current_heading, current_id = parsed
                current_lines = []
            else:
                current_lines.append(line)

        _flush(current_heading, current_id, current_lines)
        return sections

    def _extract_title(self, sections: List[Section]) -> Optional[str]:
        """Return the text of the first H1 heading, or None."""
        for section in sections:
            if section.heading:
                return section.heading
        return None
