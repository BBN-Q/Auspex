# V2 `.auspex` datasets (format version 2)

`AuspexStore` is independently usable from ordinary scripts and notebooks—no experiment, instrument service, or filter pipeline required. See [standalone writing and manual processing](filters/manual.md) for examples, or run `examples/manual_data.py`. Direct writes publish complete arrays, not appendable datasets.

One run folder contains a `manifest.json` index and self-describing numeric `.dat` files. The **old V2 format version 1 is disposable and is not read**; this is not a V1 reader either. The runner/coordinator and automatic capture of run settings/events are still to be implemented. For example:

```text
experiment_001.auspex/
  manifest.json
  metadata.json            # optional, runner-provided settings snapshot
  data/reference.dat
  data/sweep_000.dat
```

Each `.dat` file starts with the eight-byte magic `AUSPEX2\n`, a little-endian unsigned 64-bit JSON-header length, the bounded UTF-8 JSON header, then contiguous C-order numeric arrays. Header array descriptors give dtype, shape, byte offset (relative to the array region), and length. It includes version, value name/unit, named pipeline input, producing output port, label, dimensions, coordinate names/units/dimension dependencies, and optional ragged offsets. Only numeric/bool arrays; no pickle. `AuspexStore.read_file(path)` validates and reads a `.dat` even without its manifest. Do not mistake the value's dimensions for coordinate names: a coordinate may depend on multiple dimensions.

Three layouts use the **same** envelope:

- Shared grid: value `[point, frequency]`; point/swept-parameter and frequency coordinates are 1D.
- Dependent grid: value `[point, frequency]`; frequency coordinate is 2D, indexed by `(point, frequency)`.
- Ragged: flat value and sample-coordinate arrays, offsets of length `point_count + 1`, 1D point/swept-parameter coordinates. `Dataset.trace(i)` returns one row without object-dtype arrays.

`Axis` and `write_dataset(group, name, data, axes)` still support writing one completed regular array. For dependent/ragged data pass `dimensions`, `coordinates` (`Coordinate(name, values, dimensions, unit)`), and optionally `offsets`. `Dataset.coordinates` includes all coordinate arrays; `Dataset.axes` is a compatibility convenience for separable dimension coordinates only. An irregularly spaced 1D axis is a normal coordinate. No data is interpolated or padded.

## Save one manual array

For a single regular array, use the standalone `save_data` helper:

```python
import numpy as np
from auspex_v2.data import Axis, AuspexStore, save_data

bias = np.array([-0.2, 0.0, 0.3])
frequency = np.linspace(4.9, 5.1, 101)
z = np.exp(1j * bias[:, None]) * np.ones((3, 101))
path = save_data("manual_scan", z,
                 bias=bias, frequency=Axis("frequency", frequency, "GHz"),
                 comments="Manual complex transmission scan")
# path is Path('manual_scan.auspex'); directly openable in DataBrowser.
with AuspexStore(path) as store:
    dataset = store.read_dataset("data", "data")
    comments = store.read_run_metadata()["comments"]
```

Signature: `save_data(filename, data, *, comments=None, force_write=False, **axes)`.
Axis keywords follow **dimension order**, with exactly one 1D coordinate array per
value dimension and matching lengths. Plain arrays/lists have no units; `Axis`
objects preserve units and must have the same name as their keyword. Numeric/bool
values (including complex values) are supported; coordinates must be finite and
real. No ragged or dependent coordinates. Scalar data needs no axes.

The helper appends `.auspex` unless already present, creates the folder and returns
its `Path`. It saves one dataset at `data/data`, with the default value name
`value`. Optional free-form string comments are stored under `comments` in run
metadata; omitting them creates no metadata file.

An existing destination raises `FileExistsError`. **`force_write=True` deletes the
entire existing store**, including other datasets and metadata, before writing the
replacement. Only valid stores without an active writer can be replaced. Data,
axes and comments are validated before replacement, but this is not an atomic
operation: later serialization/I/O failure can leave an incomplete new store.
Use only for offline manual data, never live run folders. For multiple datasets,
value names/units or richer metadata, use `AuspexStore` directly.

## Open one dataset

```python
from pathlib import Path
from auspex_v2.data import open_data

measurement = open_data()  # OS folder picker; None if cancelled
measurement = open_data(path=Path("manual_scan.auspex"))
measurement = open_data(path=Path("/lab/runs"), number=7, date="today")
measurement = open_data(path=Path("/lab/runs"), number=7, date="2026-04-03",
                        group="readout", dataset="transmission")
values = measurement.data
frequency = measurement.coordinates["frequency"].values
```

Signature: `open_data(*, path=None, number=None, date=None, group=None, dataset=None)`.
The return is one fully loaded `Dataset`, not a store or a `(values, descriptor)`
pair. The store is closed before returning; values remain independently mutable.
All supported storage layouts, including ragged datasets, can be loaded.

Without `number`, `path` is the exact `.auspex` folder. Without `path`, the OS
folder picker lets you choose that folder; it requires optional `auspex-v2[gui]`
dependencies and must run on the main thread. Cancellation returns `None`.
Explicit-path use does not load Qt or require GUI dependencies.

With `number`, `path` is required and is the base directory. `date=None` searches
that directory; `date="today"` searches its current local-date `YYYY-MM-DD`
subfolder; an explicit `YYYY-MM-DD` string selects that date subfolder. Date is
only meaningful with `number`. Search is **not recursive** and matches only
immediate directories named `<any name>_N.auspex`. Numeric comparison ignores
zero-padding: `_7.auspex` and `_007.auspex` both match 7. Missing folders raise
`FileNotFoundError`; multiple matches raise `ValueError` listing the candidates.

The sole published dataset is loaded automatically. For multi-dataset stores,
use `group` and/or `dataset` to select a unique match. Missing matches raise
`FileNotFoundError`; ambiguous matches raise `ValueError` listing available
`group/dataset` names. No dataset is silently preferred, and unpublished files
are not scanned.

## Access axes and coordinates by position or name

Both collections are read-only sequences of metadata objects, with additional
case-sensitive string lookup:

```python
dataset = AuspexStore.read_file("run.auspex/data/trace.dat")
axis = dataset.axes["frequency"]          # Axis, not its numeric points
points = axis.points
coordinate = dataset.coordinates["frequency"]  # Coordinate, not its numeric values
values = coordinate.values
assert dataset.axes["frequency"] is dataset.axes[0]  # For a single frequency axis
```

Integer and negative indices, length, iteration, tuple equality and slicing retain
their sequence behavior. Slices return ordinary tuples, as before. Iteration yields
objects, not dictionary keys. Name lookup returns the same object as positional
access, without copying its numerical payload. Collections cannot be appended to or
assigned into; this does not make their contained lists/NumPy arrays read-only.
Missing names raise `KeyError`; invalid positions raise `IndexError`. Duplicate
names within either collection are rejected, including for directly constructed
`Dataset` objects. An axis and its corresponding coordinate may share a name.

`axes` still contains only separable dimension coordinates. A dependent frequency
grid or ragged sample coordinate is accessible through
`dataset.coordinates["frequency"]`, but is not invented as an independent axis.
This is an in-memory convenience; the version-2 file format is unchanged.

The optional JSON-compatible settings snapshot is written once through `write_run_metadata()` / read with `read_run_metadata()`; a future runner must supply it. Dynamic settings changes and events are not logged yet. Only the single writer holding an exclusive advisory lock can publish. Publication writes and syncs a temporary file, renames it, then atomically publishes the manifest; interrupted writes may leave an unlisted file, which is never silently overwritten. This is **not** an all-or-nothing transaction across a whole run, or a guarantee against disk failure. Readers opened before a new publication see their existing manifest snapshot. A lost manifest can be reconstructed by examining individual `.dat` files; automated repair is not yet provided. There is no streaming append to a published file, V1 import, automatic run settings capture or events, checksums, or digitizer-scale throughput guarantee.
