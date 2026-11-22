"""
src/chunking/chunker.py
========================
Central dispatcher that selects the correct chunking strategy and provides
the CLI entry point.

Usage (programmatic)
--------------------
    from src.chunking.chunker import chunk_documents
    from src.chunking.config import PRESET_MEDIUM

    chunks = chunk_documents(documents, config=PRESET_MEDIUM)

Usage (CLI)
-----------
    python -m src.chunking.chunker [directory] [--strategy fixed|recursive|section]
                                   [--preset small|medium|large]
                                   [--chunk-size N] [--overlap N]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from loguru import logger

from src.chunking.base import BaseChunker
from src.chunking.config import ALL_PRESETS, ChunkingConfig, PRESET_MEDIUM
from src.chunking.fixed_size import FixedSizeChunker
from src.chunking.models import Chunk
from src.chunking.recursive import RecursiveChunker
from src.chunking.section_based import SectionBasedChunker
from src.ingestion.models import ParsedDocument

# ---------------------------------------------------------------------------
# Strategy registry — add new strategies here
# ---------------------------------------------------------------------------

_STRATEGY_MAP: Dict[str, Type[BaseChunker]] = {
    "fixed": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "section": SectionBasedChunker,
}


def get_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Instantiate and return the appropriate :class:`BaseChunker` for
    the strategy named in ``config``.

    Parameters
    ----------
    config:
        ``ChunkingConfig`` whose ``strategy`` field selects the implementation.

    Returns
    -------
    BaseChunker

    Raises
    ------
    ValueError
        When ``config.strategy`` is not registered.
    """
    cls = _STRATEGY_MAP.get(config.strategy)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy {config.strategy!r}. "
            f"Available: {sorted(_STRATEGY_MAP)}"
        )
    return cls(config)


def chunk_document(
    document: ParsedDocument,
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Chunk a single document with the given configuration.

    Parameters
    ----------
    document:
        ``ParsedDocument`` from Stage 2.
    config:
        Chunking configuration.

    Returns
    -------
    List[Chunk]
    """
    chunker = get_chunker(config)
    return chunker.chunk_document(document)


def chunk_documents(
    documents: List[ParsedDocument],
    config: ChunkingConfig,
) -> List[Chunk]:
    """
    Chunk a batch of documents with the same configuration.

    Parameters
    ----------
    documents:
        List of ``ParsedDocument`` objects from Stage 2.
    config:
        Shared chunking configuration.

    Returns
    -------
    List[Chunk]
        All chunks from all documents, in document order.
    """
    chunker = get_chunker(config)
    all_chunks: List[Chunk] = []
    for doc in documents:
        try:
            doc_chunks = chunker.chunk_document(doc)
            all_chunks.extend(doc_chunks)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to chunk document '{}': {}",
                doc.filename,
                exc,
            )
    return all_chunks


# ---------------------------------------------------------------------------
# CLI summary helpers
# ---------------------------------------------------------------------------

def _print_chunk_summary(
    chunks: List[Chunk],
    documents: List[ParsedDocument],
    config: ChunkingConfig,
) -> None:
    """Print a human-readable summary of a chunking run."""
    if not chunks:
        print("\n  No chunks generated.")
        return

    lengths = [c.char_length for c in chunks]
    total = len(chunks)
    avg_len = sum(lengths) / total
    min_len = min(lengths)
    max_len = max(lengths)
    no_section_id = sum(1 for c in chunks if not c.has_section_id)

    # Per-document count
    doc_chunk_counts: Dict[str, int] = {}
    for c in chunks:
        doc_chunk_counts[c.filename] = doc_chunk_counts.get(c.filename, 0) + 1

    print("\n" + "=" * 65)
    print("  CHUNKING SUMMARY")
    print("=" * 65)
    print(f"  Strategy          : {config.strategy}")
    print(f"  Chunk size        : {config.chunk_size} chars")
    print(f"  Overlap           : {config.overlap} chars")
    print(f"  Documents         : {len(documents)}")
    print(f"  Total chunks      : {total}")
    print(f"  Avg chunk length  : {avg_len:.0f} chars")
    print(f"  Min chunk length  : {min_len} chars")
    print(f"  Max chunk length  : {max_len} chars")
    print(f"  Chunks w/o sec ID : {no_section_id}")
    print()
    print("  Per-document breakdown:")
    for doc in documents:
        count = doc_chunk_counts.get(doc.filename, 0)
        print(f"    {doc.filename:<45}  {count:>4} chunk(s)")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point:  python -m src.chunking.chunker [directory]
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run document chunking and print a summary."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help=(
            "Directory to ingest and chunk.  "
            "Defaults to settings.raw_docs_dir or ./corpus."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "section"],
        default="recursive",
        help="Chunking strategy (default: recursive).",
    )
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        default=None,
        help="Use a named preset (overrides --strategy/--chunk-size/--overlap).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Chunk size in characters (default: 700).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap in characters (default: 100).",
    )
    args = parser.parse_args()

    # Resolve directory
    if args.directory:
        target = Path(args.directory)
    else:
        try:
            from config.settings import settings
            target = settings.raw_docs_dir
        except Exception:
            target = Path("corpus")

    # Build config
    if args.preset:
        config = ALL_PRESETS[args.preset]
        # Allow strategy override on top of preset
        config = ChunkingConfig(
            strategy=args.strategy,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            min_chunk_length=config.min_chunk_length,
        )
    else:
        try:
            config = ChunkingConfig(
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        except ValueError as exc:
            logger.error("Invalid configuration: {}", exc)
            sys.exit(1)

    # Ingest documents
    from src.ingestion.document_loader import DocumentLoader

    loader = DocumentLoader()
    logger.info("Ingesting from: {}", target)

    try:
        result = loader.load_directory(target)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("{}", exc)
        sys.exit(1)

    if not result.documents:
        logger.warning("No documents ingested from '{}'.", target)
        sys.exit(0)

    # Chunk
    logger.info(
        "Chunking {} document(s) with strategy={!r} size={} overlap={}",
        len(result.documents),
        config.strategy,
        config.chunk_size,
        config.overlap,
    )
    chunks = chunk_documents(result.documents, config)

    # Summary
    _print_chunk_summary(chunks, result.documents, config)

    # Sample: print first 3 chunks
    print("  Sample chunks (first 3):")
    for c in chunks[:3]:
        preview = c.content[:120].replace("\n", "↵")
        print(
            f"    [{c.chunk_index}] id={c.chunk_id}  "
            f"sec={c.section_id!r}  "
            f"chars={c.char_length}\n"
            f"        {preview!r}"
        )
    print()


if __name__ == "__main__":
    _main()
