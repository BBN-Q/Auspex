"""Read-only discovery for the optional offline Auspex dataset browser."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .data import AuspexStore, _component, MAX_HEADER


@dataclass(frozen=True)
class DatasetEntry:
    """A published dataset and its small header, without loaded numerical arrays."""

    group: str
    name: str
    header: dict | None
    error: str | None = None

    @property
    def plottable(self):
        """Only ordinary one-dimensional values are plotted in the first browser."""
        if self.error or self.header is None or self.header.get("ragged") is not False:
            return False
        try:
            shape = self.header["arrays"]["value"]["shape"]
            dtype = np.dtype(self.header["arrays"]["value"]["dtype"])
            return (isinstance(shape, list) and len(shape) == 1 and
                    type(shape[0]) is int and shape[0] >= 0 and dtype.kind in "biufc")
        except (KeyError, TypeError, ValueError):
            return False

    @property
    def reason(self):
        if self.error:
            return self.error
        return "Plotting not yet supported (multidimensional or ragged dataset)"


@dataclass(frozen=True)
class FolderContents:
    """One refreshed manifest snapshot, optional run metadata and listed datasets."""

    path: Path
    manifest: dict
    metadata: object
    entries: tuple[DatasetEntry, ...]
    metadata_error: str | None = None


def inspect_folder(path: str | Path) -> FolderContents:
    """List published files from one manifest, reading headers but no array bytes."""
    path = Path(path).expanduser().resolve()
    with AuspexStore(path) as store:
        manifest = store.read_manifest()
        pairs = sorted(store.list_datasets())
    metadata = None
    metadata_error = None
    file = path / "metadata.json"
    if file.exists():
        try:
            if file.stat().st_size > MAX_HEADER:
                raise ValueError("Run metadata exceeds size limit")
            with file.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (ValueError, OSError) as exc:
            metadata_error = str(exc)
    entries = []
    for group, name in pairs:
        try:
            _component(group)
            _component(name)
            expected = f"{group}/{name}.dat"
            if manifest["datasets"][group][name]["file"] != expected:
                raise ValueError("Invalid dataset path in manifest")
            header = AuspexStore.read_file_header(path / expected)
            entries.append(DatasetEntry(group, name, header))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            entries.append(DatasetEntry(group, name, None, str(exc)))
    return FolderContents(path, manifest, metadata, tuple(entries), metadata_error)
