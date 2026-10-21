"""Compatibility import; prefer :mod:`auspex_v2.data.browser`."""

from .data.browser import DatasetEntry, FolderContents, inspect_folder

__all__ = ["DatasetEntry", "FolderContents", "inspect_folder"]
