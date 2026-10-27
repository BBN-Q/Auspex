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
It plots only non-ragged 1D arrays; use Python for higher-dimensional selection.

`AuspexStore.read_file(path)` validates an individual `.dat` without its manifest.
This can help inspect an orphaned file, but is not automated store repair. Unlisted
files and staging directories are not evidence of a completed publication. Retain
originals when investigating interrupted writes.

## Repository notes

- [Dataset envelope and coordinates](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/data/data.md)
- [Standalone writing](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/manual.md)
- [DataBrowser behavior](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/data/data_browser.md)
