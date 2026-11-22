"""
src/chunking/config.py
======================
Chunking configuration dataclass and built-in experimental presets.

All tuneable parameters live in ``ChunkingConfig``.  The three presets
(small / medium / large) are the starting points for before-and-after
retrieval quality comparisons.

Usage
-----
    from src.chunking.config import ChunkingConfig, PRESET_MEDIUM

    cfg = PRESET_MEDIUM
    # or override:
    cfg = ChunkingConfig(strategy="fixed", chunk_size=512, overlap=80)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Valid strategy names — referenced by multiple modules
ChunkingStrategy = Literal["fixed", "recursive", "section"]


@dataclass
class ChunkingConfig:
    """
    Parameters controlling how a document is split into chunks.

    Attributes
    ----------
    strategy:
        ``"fixed"``    – Strategy A: fixed-size character windows.
        ``"recursive"`` – Strategy B: hierarchical separator splitting.
        ``"section"``  – Strategy C: one-chunk-per-section (with overflow
                         splitting via the recursive strategy).
    chunk_size:
        Target maximum chunk size in characters.
    overlap:
        Number of characters to repeat between consecutive chunks.
        Must be strictly less than ``chunk_size``.
    min_chunk_length:
        Chunks shorter than this value are silently discarded (avoids
        near-empty trailing chunks).

    Raises
    ------
    ValueError
        When ``overlap >= chunk_size`` or either value is non-positive.
    """

    strategy: ChunkingStrategy = "recursive"
    chunk_size: int = 700
    overlap: int = 100
    min_chunk_length: int = 20

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {self.chunk_size}"
            )
        if self.overlap < 0:
            raise ValueError(
                f"overlap must be non-negative, got {self.overlap}"
            )
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_length <= 0:
            raise ValueError(
                f"min_chunk_length must be positive, got {self.min_chunk_length}"
            )
        valid_strategies = ("fixed", "recursive", "section")
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}. "
                f"Choose from {valid_strategies}."
            )

    # ------------------------------------------------------------------
    # Convenience repr for experiment logs
    # ------------------------------------------------------------------

    def label(self) -> str:
        """Human-readable label for use in experiment reports."""
        return f"{self.strategy}_size{self.chunk_size}_overlap{self.overlap}"

    def __repr__(self) -> str:
        return (
            f"ChunkingConfig(strategy={self.strategy!r}, "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap})"
        )


# ---------------------------------------------------------------------------
# Experimental presets  (Table from the spec)
# ---------------------------------------------------------------------------

PRESET_SMALL = ChunkingConfig(
    strategy="recursive",
    chunk_size=350,
    overlap=50,
    min_chunk_length=20,
)

PRESET_MEDIUM = ChunkingConfig(
    strategy="recursive",
    chunk_size=700,
    overlap=100,
    min_chunk_length=20,
)

PRESET_LARGE = ChunkingConfig(
    strategy="recursive",
    chunk_size=1000,
    overlap=150,
    min_chunk_length=20,
)

ALL_PRESETS = {
    "small": PRESET_SMALL,
    "medium": PRESET_MEDIUM,
    "large": PRESET_LARGE,
}
