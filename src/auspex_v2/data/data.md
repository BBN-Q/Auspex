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

The optional JSON-compatible settings snapshot is written once through `write_run_metadata()` / read with `read_run_metadata()`; a future runner must supply it. Dynamic settings changes and events are not logged yet. Only the single writer holding an exclusive advisory lock can publish. Publication writes and syncs a temporary file, renames it, then atomically publishes the manifest; interrupted writes may leave an unlisted file, which is never silently overwritten. This is **not** an all-or-nothing transaction across a whole run, or a guarantee against disk failure. Readers opened before a new publication see their existing manifest snapshot. A lost manifest can be reconstructed by examining individual `.dat` files; automated repair is not yet provided. There is no streaming append to a published file, V1 import, automatic run settings capture or events, checksums, or digitizer-scale throughput guarantee.
