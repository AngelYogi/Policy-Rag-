"""
src/ingestion/__init__.py
=========================
Public API for the ingestion module.

Importable at any level:
    from src.ingestion import DocumentLoader, ParsedDocument, FileType
"""

from src.ingestion.base import BaseParser
from src.ingestion.document_loader import DocumentLoader, IngestionResult
from src.ingestion.markdown_parser import MarkdownParser
from src.ingestion.models import FileType, ParsedDocument, Section
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_parser import TextParser

__all__ = [
    "BaseParser",
    "DocumentLoader",
    "FileType",
    "IngestionResult",
    "MarkdownParser",
    "PDFParser",
    "ParsedDocument",
    "Section",
    "TextParser",
]
