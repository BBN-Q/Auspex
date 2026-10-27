# Data and storage

A V2 `.auspex` directory stores completed numerical datasets and their coordinates.
`AuspexStore` is usable independently of experiments, services, or Qt.

## Run artifacts

A service-managed run normally contains:

```text
source_vna_000.auspex/
  manifest.json
  metadata.json
  plan.json
  status.json
  execution.log
  measurement/
    s21.dat
```

The manifest lists published datasets. Each `.dat` is self-describing: it contains
numeric values, coordinate arrays, dtype/shape descriptions, dimension dependencies,
names/units, and optional ragged offsets. Plan format 6 and dataset format 2 are
independent version schemes. V1 and old V2 format-1 data are not read by this store.

The service saves a plan and run metadata, not a complete physical settings history.
`execution.log` is best-effort diagnostics, not a durable audit trail or automatic
capture of every intermediate calculation. Status can report failure even when
an earlier dataset was published successfully.

## Save a manual array with `save_data`

For one completed regular array, no experiment or pipeline is needed:

```python
import numpy as np
from auspex_v2.data import Axis, save_data, open_data

bias = np.array([-0.2, 0.0, 0.3])
frequency = np.linspace(4.9, 5.1, 101)
z = np.exp(1j * bias[:, None]) * np.ones((3, 101))
path = save_data("manual_scan", z,
                 bias=bias, frequency=Axis("frequency", frequency, "GHz"),
                 comments="Manual complex transmission scan")
measurement = open_data(path=path)
np.testing.assert_array_equal(measurement.data, z)
```

`save_data(filename, data, *, comments=None, force_write=False, **axes)` creates a
browser-ready folder, appends `.auspex` if absent, and returns its `Path`. The
example writes `manual_scan.auspex/data/data.dat`, published as group `data`,
dataset `data`.

Axis keywords follow **data dimension order**, with exactly one matching-length
1D axis per dimension. Values can be plain coordinate arrays/lists or `Axis`
objects preserving units; an `Axis.name` must match its keyword. Coordinates must
be finite and real; values may be real, complex or boolean. Ragged/dependent axes
are not supported by this helper. Scalars require no axes.

Optional free-form string comments are saved under `comments` in `metadata.json`:

```python
from auspex_v2.data import AuspexStore

with AuspexStore(path) as store:
    comments = store.read_run_metadata()["comments"]
```

Omitting comments creates no metadata file. For multiple datasets, value names/units or richer metadata, use the
[direct store API](#write-complete-arrays-directly).

An existing destination raises `FileExistsError`. **`force_write=True` deletes the
entire existing store**, including other datasets and metadata. Only valid stores
without an active writer can be replaced. Inputs are validated first, but
replacement is not atomic: later serialization/I/O failures can leave an
incomplete new store. Use this only for offline manual data, never live run folders.

## Load a dataset with `open_data`

`open_data(*, path=None, number=None, date=None, group=None, dataset=None)` returns
one fully loaded `Dataset`, not a store or a `(values, descriptor)` pair. Its store
is closed before return; the numerical arrays remain independently mutable. It
can read every supported stored layout, including ragged datasets.

```python
from pathlib import Path
from auspex_v2.data import open_data

measurement = open_data(path=Path("manual_scan.auspex"))
values = measurement.data
frequency = measurement.coordinates["frequency"].values

# Exact dataset selection in a multi-dataset store:
measurement = open_data(path=Path("run.auspex"),
                        group="measurement", dataset="s21")

# Numbered-folder lookup under an explicit base directory:
measurement = open_data(path=Path("/lab/runs"), number=7)
measurement = open_data(path=Path("/lab/runs"), number=7, date="today")
measurement = open_data(path=Path("/lab/runs"), number=7, date="2026-04-03")
```

Without `number`, `path` is the exact `.auspex` folder. Number lookup requires a
base `path` and matches immediate directories named `<any name>_N.auspex`, ignoring
zero-padding: `_7.auspex` and `_007.auspex` both match 7. Search is **not recursive**.
`date=None` searches the base directory itself; `"today"` uses the current local
`YYYY-MM-DD` subfolder; an explicit date string selects that subfolder. Date is
only accepted with `number`.

The sole published dataset is selected automatically. Optional `group` and/or
`dataset` filters must identify one match. Missing folders/datasets raise
`FileNotFoundError`; ambiguous folder/dataset matches raise `ValueError` listing
candidates. Unpublished files are not scanned.

With no path, `open_data()` shows an OS folder picker. This requires the optional
`gui` extra and must run on the main thread. Cancellation returns `None`:

```python
measurement = open_data()
if measurement is not None:
    values = measurement.data
```

Explicit-path use requires neither Qt nor GUI dependencies.

## Read values and coordinates

```python
from auspex_v2.data import AuspexStore

with AuspexStore("data/docs-demo/source_vna_000.auspex") as store:
    dataset = store.read_dataset("measurement", "s21")

first_trace = dataset.data[0]
frequency = dataset.coordinates["frequency"].values
source_frequency = dataset.axes["source_frequency"].points
```

`axes` and `coordinates` accept names or integer positions and return metadata
objects, not arrays directly. They preserve sequence iteration, negative indices,
and ordinary tuple slices. Missing names raise `KeyError`; invalid positions raise
`IndexError`. Collections are read-only, but their numerical payloads retain their
normal list/NumPy mutability.

`coordinates` contains every coordinate. `axes` contains only separable dimension
coordinates. A changing frequency grid may depend on both point and frequency;
use `coordinates`, not a fabricated shared axis. Ragged 1D scans store flat values
and offsets; `dataset.trace(i)` selects one row. No padding or interpolation is
performed implicitly.

## Write complete arrays directly

```python
from auspex_v2.data import Axis, AuspexStore

with AuspexStore("manual.auspex", mode="w") as store:
    store.write_dataset("measurement", "trace", [1., 2., 3.],
                        axes=[Axis("time", [0., 0.1, 0.3], "s")])
```

Only one writer owns a store. New stores and existing dataset names are never
overwritten implicitly. Publication syncs a temporary file and updates the manifest;
this is not a transaction across a whole run or protection against disk failure.
Readers retain their opened manifest snapshot; reopen to see later publications.
There is no append-to-published-array API.

## Inspect and recover cautiously

[DataBrowser](../utilities/data-browser.md) reads published datasets and metadata.
It plots non-ragged 1D arrays and rectilinear 2D maps with monotonic coordinates.
Use Python for dependent-coordinate, ragged or higher-rank data selection.

`AuspexStore.read_file(path)` validates an individual `.dat` without its manifest.
This can help inspect an orphaned file, but is not automated store repair. Unlisted
files and staging directories are not evidence of a completed publication. Retain
originals when investigating interrupted writes.

## Repository notes

- [Dataset envelope and coordinates](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/data/data.md)
- [Standalone writing](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/manual.md)
- [DataBrowser behavior](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/data/data_browser.md)
