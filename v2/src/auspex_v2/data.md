# V2 `.auspex` dataset storage

This is a new format with **conceptual**, not binary or metadata, compatibility with V1. V2 does not read V1 `.auspex` folders. A V2 folder has a root `manifest.json` containing `format_version: 1` and a `datasets` mapping by group and name. Each published dataset has its own group/name `.dat` file (raw C-order NumPy values) and a manifest descriptor with `file`, `shape`, `dtype`, and ordered `axes` (`name`, `unit`, actual numeric `points`). Unsupported format versions are rejected.

```python
from auspex_v2.data import Axis, AuspexStore

with AuspexStore("measurement.auspex", mode="w") as store:
    store.write_dataset("vna", "s21", complex_trace,
                        [Axis("frequency", actual_frequencies, "Hz")])

with AuspexStore("measurement.auspex") as store:
    dataset = store.read_dataset("vna", "s21")
    print(dataset.data, dataset.axes)
```

`w` creates a new folder; `a` opens an existing V2 folder for additional *new* datasets; `r` reads a snapshot of its published manifest. Writers hold an exclusive advisory file lock for their lifetime. Every write saves the binary file before atomically publishing a new manifest; readers never see an in-progress dataset. Failed writes can leave an unpublished file; these are not overwritten or automatically recovered. Published binary lengths are checked on read. Files and manifest are synced before publication, but this is **not** a transaction across storage devices or a guarantee against disk failure. Keep the writer context short and close it when done.

Only completed numeric/bool arrays with one finite numeric coordinate vector per dimension are currently supported; scalar arrays have no axes. Group/dataset names are single safe path components. There is no append/streaming, concurrent-writer merge, adaptive-ragged array format, legacy reader, or run-level plan/status/event schema yet. Those can be added to the versioned folder in a later reviewed slice without changing the existing dataset contract.
