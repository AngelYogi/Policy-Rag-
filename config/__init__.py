"""
config/__init__.py
Exposes the singleton `settings` object for clean imports:

    from config import settings
"""

from config.settings import settings  # noqa: F401

__all__ = ["settings"]
