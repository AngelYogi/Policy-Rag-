"""
src/ingestion/text_parser.py
=============================
Parser for plain-text (``.txt``) files.

Parsing strategy
----------------
* Read the file with UTF-8 encoding; fall back to Latin-1 on decode errors.
* Use simple heuristics to detect candidate headings:
    - Lines that are ALL CAPS and 3–80 characters long.
    - Lines followed immediately by a line of ``===`` or ``---`` (Setext style).
  Heuristics are conservative: when a line does not clearly look like a
  heading it is treated as body text.
* Section IDs are **never** invented for plain-text files; ``section_id``
  is always ``None``.
* The first detected heading (or the first non-blank line when no heading is
  found) is used as the document title.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from src.ingestion.base import BaseParser
from src.ingestion.models import FileType, ParsedDocument, Section

# ---------------------------------------------------------------------------
# Heading-detection helpers
# ---------------------------------------------------------------------------

_SETEXT_UNDERLINE_CHARS = {"=", "-"}
_MIN_HEADING_LEN = 3
_MAX_HEADING_LEN = 120


def _is_setext_underline(line: str) -> bool:
    """Return True for lines composed entirely of ``=`` or ``-`` chars."""
    stripped = line.strip()
    return (
        len(stripped) >= 3
        and all(ch in _SETEXT_UNDERLINE_CHARS for ch in stripped)
        and len(set(stripped)) == 1  # all same character
    )


def _is_all_caps_heading(line: str) -> bool:
    """
    Heuristic: a line is a heading candidate when it is ALL CAPS,
    contains at least one alphabetic character, and is within the
    expected length range.
    """
    stripped = line.strip()
    return (
        _MIN_HEADING_LEN <= len(stripped) <= _MAX_HEADING_LEN
        and stripped == stripped.upper()
        and any(ch.isalpha() for ch in stripped)
    )


def _detect_headings(
    lines: List[str],
) -> Tuple[List[Tuple[int, bool]], List[bool]]:
    """
    Analyse each line and return:
      - A list of ``(line_index, is_heading)`` tuples (one per line).
      - A parallel ``is_underline`` bool list (True for Setext underline rows).

    The underline row in a Setext-style heading is marked so it can be
    skipped when building section content.
    """
    is_heading: List[bool] = [False] * len(lines)
    is_underline: List[bool] = [False] * len(lines)

    for i, line in enumerate(lines):
        # Setext: previous non-blank line + this underline
        if _is_setext_underline(line) and i > 0:
            prev = lines[i - 1].strip()
            if prev and not is_underline[i - 1]:
                is_heading[i - 1] = True
                is_underline[i] = True
                continue
        # ALL-CAPS heuristic
        if _is_all_caps_heading(line):
            is_heading[i] = True

    return [(i, h) for i, h in enumerate(is_heading)], is_underline


class TextParser(BaseParser):
    """Parses plain-text files into a structured :class:`ParsedDocument`."""

    @property
    def supported_extensions(self) -> List[str]:
        return ["txt", "text"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a plain-text file.

        Attempts UTF-8 first; falls back to Latin-1 on decode errors to
        avoid crashing on Windows-encoded files.

        Parameters
        ----------
        file_path:
            Path to the ``.txt`` file.

        Returns
        -------
        ParsedDocument

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        PermissionError
            When the file cannot be read.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        logger.debug("Parsing text file: {}", file_path)

        encoding_used = "utf-8"
        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except PermissionError as exc:
            raise PermissionError(f"Cannot read {file_path}: {exc}") from exc
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for '{}'; falling back to Latin-1.",
                file_path.name,
            )
            raw_text = file_path.read_text(encoding="latin-1")
            encoding_used = "latin-1"

        sections = self._extract_sections(raw_text)
        title = self._extract_title(sections, raw_text)

        doc = ParsedDocument(
            document_id=str(uuid.uuid4()),
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=FileType.TEXT,
            title=title,
            content=raw_text,
            sections=sections,
            page_count=None,
            metadata={
                "encoding": encoding_used,
                "line_count": raw_text.count("\n") + 1,
                "char_count": len(raw_text),
            },
        )

        logger.info(
            "Parsed text '{}': {} section(s), title={!r}",
            file_path.name,
            len(sections),
            title,
        )
        return doc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_sections(self, text: str) -> List[Section]:
        """Split plain text into sections based on heading heuristics."""
        lines = text.splitlines()
        if not lines:
            return []

        (heading_flags, underline_flags) = _detect_headings(lines)

        sections: List[Section] = []
        current_heading = ""
        current_lines: List[str] = []

        def _flush(heading: str, body_lines: List[str]) -> None:
            content = "\n".join(body_lines).strip()
            if heading or content:
                sections.append(
                    Section(
                        heading=heading,
                        content=content,
                        section_id=None,  # never invented for plain text
                    )
                )

        for i, line in enumerate(lines):
            if underline_flags[i]:
                # Skip the underline row itself; heading was set on previous iteration
                continue

            _, is_h = heading_flags[i]
            if is_h:
                _flush(current_heading, current_lines)
                current_heading = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        _flush(current_heading, current_lines)
        return sections

    def _extract_title(
        self, sections: List[Section], raw_text: str
    ) -> Optional[str]:
        """
        Return the first detected heading as title.
        Fall back to the first non-blank line when no headings were found.
        """
        for section in sections:
            if section.heading:
                return section.heading
        # Fallback: first non-blank line
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:100]  # cap to 100 chars
        return None
