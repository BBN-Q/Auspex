"""Convenient standalone saving and loading of completed datasets."""

from datetime import date as calendar_date
import json
from numbers import Integral
from pathlib import Path
import re
import shutil

import numpy as np

from .data import Axis, AuspexStore, Dataset, MAX_HEADER, _validated_dataset


def save_data(filename, data, *, comments=None, force_write=False, **axes) -> Path:
    """Save one numeric array at ``data/data`` in a browser-ready store.

    Append ``.auspex`` if absent and return the resulting Path. Axis keywords,
    in insertion order, correspond to the data dimensions. Each value is either
    a coordinate array or an Axis with a matching name (and optional units).
    Exactly one axis per dimension is required; ragged data is not supported.
    Optional string comments are stored as ``metadata.json``'s ``comments``.

    Existing destinations raise FileExistsError unless force_write=True, which
    deletes the ENTIRE existing store, including other datasets and metadata.
    Only valid stores without an active writer can be replaced. This is an
    offline convenience, not an atomic replacement or concurrent-run API: do not
    use it on live run folders. Data, axes and comments are validated before
    deletion; later serialization/I/O failures may leave an incomplete new store.
    """
    path = Path(filename).expanduser()
    if path.suffix != ".auspex":
        path = path.with_name(path.name + ".auspex")
    resolved_axes = []
    for name, points in axes.items():
        if isinstance(points, Axis):
            if points.name != name:
                raise ValueError(f"Axis name {points.name!r} does not match keyword {name!r}")
            resolved_axes.append(points)
        else:
            resolved_axes.append(Axis(name, points))
    dataset = _validated_dataset(Dataset(np.asarray(data), tuple(resolved_axes)))
    if comments is not None:
        if not isinstance(comments, str):
            raise TypeError("comments must be a string or None")
        # Check the same encoded limit as the writer before any forced deletion.
        encoded = json.dumps({"comments": comments}, allow_nan=False, indent=2).encode("utf-8")
        if len(encoded) > MAX_HEADER:
            raise ValueError("Run metadata exceeds size limit")
    if path.exists():
        if not force_write:
            raise FileExistsError(path)
        with AuspexStore(path, mode="a"):
            shutil.rmtree(path)
    with AuspexStore(path, mode="w") as store:
        store.write_dataset("data", "data", dataset.data, resolved_axes)
        if comments is not None:
            store.write_run_metadata({"comments": comments})
    return path


def _pick_data_folder():
    """Keep Qt optional and its application alive through the modal folder picker."""
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:
        raise ImportError("The folder picker requires auspex-v2[gui]; alternatively pass path=") from exc
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return QtWidgets.QFileDialog.getExistingDirectory(None, "Open an .auspex data folder")


def open_data(*, path=None, number=None, date=None, group=None, dataset=None) -> Dataset | None:
    """Load one published Dataset, without retaining an open store or GUI objects.

    With no path, show an OS folder picker (requires optional GUI dependencies,
    called from the main thread); cancellation returns None. Otherwise path is
    the exact .auspex folder, unless number is supplied. Number lookup requires
    path and searches its immediate children for <name>_N.auspex directories,
    matching N numerically regardless of zero-padding. An optional date selects
    a YYYY-MM-DD child directory; 'today' uses the current local date. Date=None
    searches path itself. Date without number is an error.

    Select the sole published dataset, optionally filtered by group and/or
    dataset name. Missing folders/datasets raise FileNotFoundError; ambiguous
    folder or dataset matches raise ValueError with the available matches.
    Explicit-path loading never imports Qt and supports every stored layout.
    """
    if number is None and date is not None:
        raise ValueError("date requires number")
    if number is not None:
        if path is None:
            raise ValueError("number requires path")
        if isinstance(number, bool) or not isinstance(number, Integral) or number < 0:
            raise ValueError("number must be a nonnegative integer")
    if path is None:
        path = _pick_data_folder()
        if not path:
            return None
    path = Path(path).expanduser()
    if number is not None:
        if date is not None:
            if date == "today":
                date = calendar_date.today().isoformat()
            if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError("date must be None, 'today', or YYYY-MM-DD")
            calendar_date.fromisoformat(date)  # Reject impossible calendar dates, too.
            path = path / date
        matches = []
        for child in sorted(path.iterdir()):
            match = re.fullmatch(r".+_(\d+)\.auspex", child.name)
            if match and int(match[1]) == number and child.is_dir():
                matches.append(child)
        if not matches:
            raise FileNotFoundError(f"No .auspex folder numbered {number} in {path}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous folder number {number} in {path}: "
                             + ", ".join(p.name for p in matches))
        path = matches[0]
    with AuspexStore(path) as store:
        available = store.list_datasets()
        matches = [(g, n) for g, n in available
                   if (group is None or g == group) and (dataset is None or n == dataset)]
        choices = ", ".join(f"{g}/{n}" for g, n in available) or "(none)"
        if not matches:
            raise FileNotFoundError(f"No matching dataset in {path}; available: {choices}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous dataset in {path}; specify group/dataset. Available: {choices}")
        return store.read_dataset(*matches[0])
