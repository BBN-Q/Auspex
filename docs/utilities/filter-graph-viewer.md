# Filter graph viewer

`filter_graph_viewer.py` attaches to a running FilterServer's explicitly enabled
inspection socket. It shows graph structure, schemas, and last-acknowledged filter
configuration. It is not a plotter or editor.

```sh
python -m pip install -e ".[gui]"
python utils/filter_graph_viewer.py /private/path/filter.sock
```

There is no automatically discovered global graph socket. ExperimentServer does
not currently expose a command-line option to enable one. In a manual acquisition
script, pass `inspect_socket` to FilterServer, or through ManualPipeline's options:

```python
from auspex_v2.filters import FilterServer

# schema, nodes, edges, and results describe your graph.
with FilterServer(schema, nodes, edges, results,
                  inspect_socket="/private/path/filter.sock") as graph:
    ...  # Keep the graph open while the separate viewer is attached.
```

The parent directory must exist, belong to the current user, and exclude group/other
access (mode 0700). The endpoint uses mode 0600, refuses an existing path, and is
removed when the graph closes. Configuration can contain sensitive laboratory
values; restrict access accordingly.

## Use the view

Select a node or connection to inspect details. Scroll to zoom, middle-drag to
pan, and use **Fit graph** to reset framing. Solid edges are required; dashed edges
represent lossy previews.

**Refresh** requests one complete snapshot. There is no automatic polling, data
subscription, configuration mutation, or exact queue/worker-state display. A
disconnected viewer retains its last snapshot labeled as such. A configuration
shown as acknowledged is not a measurement of physical hardware state.

For headless access, `auspex_v2.filters.inspection.fetch_snapshot(path)` obtains
the same kind of snapshot. See the retained
[inspection guide](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/filters/inspection.md)
for the endpoint contract.
