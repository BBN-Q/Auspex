# Data browser

`data_browser.py` is a read-only viewer for published V2 `.auspex` datasets.
Install the `gui` extra; no instrument or experiment service is required.

```sh
python -m pip install -e ".[gui]"
python utils/data_browser.py
python utils/data_browser.py data/docs-demo
python utils/data_browser.py data/docs-demo/source_vna_000.auspex
```

## Browse and plot

Add directories containing runs, or individual run folders. Directory discovery
covers immediate children only, not recursive date subdirectories. The tree shows
folder → run → group → dataset. Double-click a supported dataset to open its tab.

Non-ragged **1D arrays and rectilinear 2D maps** are plotted using stored
coordinates, including nonuniform monotonic grids. Complex maps offer real,
imaginary, magnitude, phase, stacked and four-panel modes; select a colormap or
drag optional X/Y cuts in single-panel views. The quickstart's `(3, 201)` trace
dataset opens as a heatmap. Ragged, dependent-coordinate and higher-rank datasets
remain metadata-only. Rendering is shared with the detached [live plot client](live-plots.md).

The metadata pane shows run metadata, manifest information, and the self-describing
dataset header. Arrays are loaded for plotted tabs, not merely to populate the
tree. Many large open tabs can still consume substantial memory.

## Refresh and close

Use **Refresh**, the File menu, or Ctrl+R to rescan folders/manifests/headers.
Refresh is manual, not a live acquisition subscription; existing open plots remain
unchanged for retained runs. Removing a watched folder or closing a plot never
deletes data. A run's context menu can close all its dataset tabs.

The browser does not fit, export, modify, or reprocess data. It lists published
manifest entries, not staging/orphan files, and does not read V1 or old V2 format-1
stores. For individual-file investigation, see [data recovery](../components/data.md#inspect-and-recover-cautiously).

See the retained [DataBrowser guide](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/data/data_browser.md).
