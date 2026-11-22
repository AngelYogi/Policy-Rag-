"""
src/ingestion/document_loader.py
==================================
Central document loader: discovers files, dispatches to the correct parser,
and returns a list of :class:`ParsedDocument` objects.

Design
------
* ``DocumentLoader`` maintains a registry of ``BaseParser`` subclasses.
* File extension → parser is resolved at load time, not at startup.
* A single bad file never aborts the full batch; errors are logged and
  collected in the ``IngestionResult``.

Command-line usage
------------------
  python -m src.ingestion.document_loader [directory]

  When no directory is given, the value of ``settings.raw_docs_dir`` is used.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from src.ingestion.base import BaseParser
from src.ingestion.markdown_parser import MarkdownParser
from src.ingestion.models import ParsedDocument
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_parser import TextParser

# ---------------------------------------------------------------------------
# Default parser registry — add new parsers here, nothing else changes.
# ---------------------------------------------------------------------------

_DEFAULT_PARSERS: List[Type[BaseParser]] = [
    MarkdownParser,
    TextParser,
    PDFParser,
]


@dataclass
class IngestionResult:
    """
    Summary of a document loading run.

    Attributes
    ----------
    documents:
        Successfully parsed documents.
    failed_files:
        Map of ``{file_path_str: error_message}`` for files that could not
        be parsed.
    skipped_files:
        Files whose extension is not supported.
    duration_seconds:
        Wall-clock time taken for the entire ingestion run.
    """

    documents: List[ParsedDocument] = field(default_factory=list)
    failed_files: Dict[str, str] = field(default_factory=dict)
    skipped_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def total_discovered(self) -> int:
        return len(self.documents) + len(self.failed_files) + len(self.skipped_files)

    @property
    def success_count(self) -> int:
        return len(self.documents)

    @property
    def failed_count(self) -> int:
        return len(self.failed_files)

    @property
    def total_sections(self) -> int:
        return sum(d.section_count for d in self.documents)

    @property
    def empty_document_count(self) -> int:
        return sum(1 for d in self.documents if d.is_empty)

    def print_summary(self) -> None:
        """Print a human-readable ingestion summary to stdout."""
        print("\n" + "=" * 60)
        print("  DOCUMENT INGESTION SUMMARY")
        print("=" * 60)
        print(f"  Duration          : {self.duration_seconds:.2f}s")
        print(f"  Files discovered  : {self.total_discovered}")
        print(f"  Successfully parsed: {self.success_count}")
        print(f"  Failed            : {self.failed_count}")
        print(f"  Skipped (unsupported): {len(self.skipped_files)}")
        print(f"  Total sections    : {self.total_sections}")
        print(f"  Empty documents   : {self.empty_document_count}")

        if self.failed_files:
            print("\n  FAILED FILES:")
            for path, err in self.failed_files.items():
                print(f"    ✗ {path}")
                print(f"      {err}")

        if self.skipped_files:
            print("\n  SKIPPED (unsupported extension):")
            for path in self.skipped_files:
                print(f"    - {path}")

        print("=" * 60 + "\n")


class DocumentLoader:
    """
    Discovers and parses documents from a directory (or a single file).

    Parameters
    ----------
    extra_parsers:
        Additional parser classes to register beyond the defaults.  Useful
        for extending support without modifying this file.
    """

    def __init__(
        self, extra_parsers: Optional[List[Type[BaseParser]]] = None
    ) -> None:
        self._parsers: List[BaseParser] = []
        self._ext_map: Dict[str, BaseParser] = {}

        all_parser_classes = _DEFAULT_PARSERS + (extra_parsers or [])
        for cls in all_parser_classes:
            instance = cls()
            self._parsers.append(instance)
            for ext in instance.supported_extensions:
                self._ext_map[ext.lower()] = instance

        logger.debug(
            "DocumentLoader initialised with extensions: {}",
            sorted(self._ext_map.keys()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> IngestionResult:
        """
        Discover and parse all supported files in ``directory``.

        Parameters
        ----------
        directory:
            Path to scan.
        recursive:
            When True, descend into sub-directories.

        Returns
        -------
        IngestionResult
            Contains all parsed documents plus error/skip information.

        Raises
        ------
        FileNotFoundError
            When ``directory`` does not exist.
        NotADirectoryError
            When ``directory`` exists but is not a directory.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {directory}"
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        logger.info("Scanning directory: {}", directory)

        pattern = "**/*" if recursive else "*"
        all_files = [
            p for p in directory.glob(pattern) if p.is_file()
        ]

        logger.info("Found {} file(s) in '{}'", len(all_files), directory)

        result = IngestionResult()
        start = time.perf_counter()

        for file_path in sorted(all_files):
            self._process_file(file_path, result)

        result.duration_seconds = time.perf_counter() - start
        logger.info(
            "Ingestion complete: {}/{} parsed in {:.2f}s",
            result.success_count,
            result.total_discovered,
            result.duration_seconds,
        )
        return result

    def load_file(self, file_path: Path) -> ParsedDocument:
        """
        Parse a single file and return the ``ParsedDocument``.

        Parameters
        ----------
        file_path:
            Path to the file to parse.

        Returns
        -------
        ParsedDocument

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When the file extension is not supported.
        Any parser-specific exception propagates unchanged.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lstrip(".").lower()
        parser = self._ext_map.get(ext)
        if parser is None:
            raise ValueError(
                f"Unsupported file type '.{ext}' for file: {file_path}"
            )

        return parser.parse(file_path)

    @property
    def supported_extensions(self) -> List[str]:
        """Return the list of registered file extensions."""
        return sorted(self._ext_map.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_file(self, file_path: Path, result: IngestionResult) -> None:
        """
        Attempt to parse a single file and update ``result`` in place.

        Errors are caught and recorded; they never abort the whole batch.
        """
        ext = file_path.suffix.lstrip(".").lower()

        # Skip hidden files and system files
        if file_path.name.startswith("."):
            return

        parser = self._ext_map.get(ext)
        if parser is None:
            logger.debug("Skipping unsupported file: {}", file_path.name)
            result.skipped_files.append(str(file_path))
            return

        try:
            doc = parser.parse(file_path)

            if doc.is_empty:
                logger.warning(
                    "Document '{}' has no extractable text content.",
                    file_path.name,
                )

            result.documents.append(doc)

        except (FileNotFoundError, PermissionError, ValueError) as exc:
            logger.error(
                "Failed to parse '{}': {} — {}",
                file_path.name, type(exc).__name__, exc,
            )
            result.failed_files[str(file_path)] = f"{type(exc).__name__}: {exc}"

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected error parsing '{}': {}",
                file_path.name, exc,
            )
            result.failed_files[str(file_path)] = f"UnexpectedError: {exc}"


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.ingestion.document_loader [path]
# ---------------------------------------------------------------------------

def _main() -> None:
    """Run ingestion on a directory and print the summary."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run document ingestion and print a summary."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help=(
            "Directory to ingest. Defaults to the value of "
            "settings.raw_docs_dir from config/settings.py."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        default=False,
        help="Disable recursive directory scanning.",
    )
    args = parser.parse_args()

    if args.directory:
        target = Path(args.directory)
    else:
        try:
            from config.settings import settings
            target = settings.raw_docs_dir
        except Exception as exc:
            logger.error("Could not load settings: {}", exc)
            sys.exit(1)

    loader = DocumentLoader()
    logger.info("Supported extensions: {}", loader.supported_extensions)

    try:
        result = loader.load_directory(target, recursive=not args.no_recursive)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("{}", exc)
        sys.exit(1)

    result.print_summary()

    # Print per-document details
    for doc in result.documents:
        print(
            f"  [{doc.file_type.value.upper():8}] "
            f"{doc.filename:<40} "
            f"{doc.section_count:>3} section(s)  "
            f"{len(doc.content):>7} chars  "
            f"title={doc.title!r}"
        )


if __name__ == "__main__":
    _main()
