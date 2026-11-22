"""
src/ingestion/base.py
=====================
Abstract base class for all document parsers.

Adding a new format means:
  1. Subclass ``BaseParser``.
  2. Implement ``supported_extensions`` and ``parse``.
  3. Register the class in ``document_loader.py``.

Nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from src.ingestion.models import ParsedDocument


class BaseParser(ABC):
    """
    Abstract base for file-type-specific parsers.

    Every concrete parser must declare which extensions it handles and
    implement the ``parse`` method that returns a ``ParsedDocument``.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Return lowercase file extensions this parser handles (without dot).

        Example: ``["pdf"]`` or ``["md", "markdown"]``.
        """

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a file and return a structured ``ParsedDocument``.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the file.

        Returns
        -------
        ParsedDocument
            Fully populated document model.

        Raises
        ------
        FileNotFoundError
            When ``file_path`` does not exist.
        PermissionError
            When the file cannot be read.
        ValueError
            When the file content is fundamentally unparseable.
        """

    def can_handle(self, file_path: Path) -> bool:
        """Return True when this parser can handle the given file extension."""
        return file_path.suffix.lstrip(".").lower() in self.supported_extensions
