# Live plots

`plot_client.py` is a detached, read-only viewer. Install the `gui` extra and leave
one window open while experiments run:

```sh
python utils/plot_client.py
python utils/plot_client.py --socket /path/to/experiments.sock
python utils/plot_client.py --no-discovery /path/to/run.auspex
```

The default client discovers known runs through ExperimentServer. It reads each
run's `preview.json` to find its private local preview socket; numerical frames
never cross the experiment-control connection. Explicit run folders also work
with manual pipelines and after service cleanup. There is no third persistent
plot daemon, TCP discovery or automatic GUI launch.

## Window behavior

- Left: experiment/run names, newest first, with their plot endpoints underneath.
- Center: one tab per plot; new runs add tabs **without stealing focus**.
- Right: selected run metadata/status, plot source/schema, sweep position,
  averaging count, configuration revision and snapshot timestamp.

Closing a tab stops fetching that plot but retains its listing and last snapshot.
Double-click to reopen. **Close plot** above the left pane removes the selected
plot and suppresses automatic rediscovery for this client session only. Nothing
is deleted on disk and no acquisition command is sent.

Completed, failed and cancelled plots remain available. At most **64 plots** are
listed, including closed tabs. Admission evicts the oldest terminal-run plot
first. If all retained plots belong to active runs, new plots are declined with
a visible warning and reconsidered on later refreshes. Explicitly removed and
automatically evicted entries stay suppressed for this session.

Rendering is shared with the [Data Browser](data-browser.md): actual coordinates,
complex components, plasma-default colormaps and single-panel map cuts. Array
updates retain view ranges, component/colormap selection and cut positions.

## Declare endpoints

Import inert authoring nodes from `auspex_v2.filters.definitions`:

```python
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, ComplexAverage, WriteToFile, Plot,
)

trace = PipelineInput("trace")
average = ComplexAverage(name="average", repeats=3)
writer = WriteToFile(name="disk", group="measurement", dataset="s21")
current = Plot(name="current_trace")
image = Plot(name="bias_map", mode="map")
pipeline = PipelineDefinition(inputs=[trace], edges=[
    (trace, average.sink),
    (average.final_average, writer.sink),
    (average.partial_average, current.sink),
    (average.final_average, image.sink),
])
```

In an experiment's `setup_pipeline`, connect its output to `trace` as usual.
At each swept bias, publish three complete traces with identical actual frequency
coordinates. The current trace shows partial averages, labeled with their count;
only finalized averages add rows to the map. The run must have exactly one
independent fixed sweep with strictly monotonic points. The runner supplies the
planned sweep coordinates so unacquired cells are missing (NaN), never zero-filled.

For `ManualPipeline`, supply the input schema normally and optionally
`preview_sweep=Axis("bias", bias_points, "V")`. Without this declaration a map grows
in acquired order. Changed trace grids, repeated finalized positions or unsupported
sweep paths disable the map with a diagnostic; required files preserve actual data.
Initial plotting requires a writer-backed pipeline and rank-one trace sources.
Map endpoints must share the same source as a required writer.

## Bounds and failure semantics

`Plot(max_bytes=...)` bounds retained numerical display arrays, including axes
(default **8 MiB** per plot). Declared limits sum to at most **64 MiB per run** and
at most 64 endpoints. These are payload bounds, not total RSS: queues, encoded
frames, rendering and temporary copies also use memory. The client retains at
most 64 plot snapshots; their individual sizes still matter.

Current trace delivery is lossy. Finalized rows enter the writer-owned map state
before display frames are coalesced, at most five frames/second per plot. A late
or reconnecting client receives a complete acquired snapshot, not a stream of
irreplaceable row increments. Slow viewers use their own socket/GUI work and do
not hold shared-memory slots, cause required backpressure, or participate in
completion acknowledgments. Display setup/update failures warn and disable only
the affected preview. Preview construction/encoding has bounded work but is not
zero-cost; this is not a throughput benchmark.

On ordinary pipeline closure, best-effort final snapshots are cached under
`.preview/` with an updated `preview.json`. These bounded auxiliary files permit
late viewing; they are **not published datasets** and may be absent after abrupt
termination. Saved data and `status.json` remain authoritative. There is no
historical run-folder scan or reconstruction of a crashed preview service.

The GUI labels disconnected/stale views. Having acquired a row, publishing a
dataset and overall run success are distinct. For a manual pipeline without
`status.json`, `CLOSED` means the preview endpoint closed—not verified experiment
success or safe hardware state. Neither closing the GUI nor removing plots
cancels a run, changes settings, releases ownership or deletes files.

## Attended visual test

```sh
python tests/visual_plot_client.py
# Optional: keep synthetic output in a new directory:
python tests/visual_plot_client.py --directory /path/to/new-demo
```

Two finite simulated acquisitions drive actual spawned averaging and writer
processes. Check partial trace counts, progressive map rows, complex display
modes, cuts, metadata, new-run focus, close/reopen and Close plot. The second run
adds tabs in the same window. Closing that window does not cancel the finite
acquisition. No instrument services or hardware are used; all values are synthetic.
