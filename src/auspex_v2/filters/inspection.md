# Read-only filter graph inspection (local prototype)

`FilterServer` can expose a **read-only, one-shot graph snapshot** via an explicit Unix socket path. This is not a plot server, graph editor, run coordinator, or data subscription. The endpoint exists only while its FilterServer is open. Qt is optional and never imported by the headless graph or filter workers.

```python
from auspex_v2.filters import FilterServer

# Create a user-owned private directory (0700) first, e.g. ~/.auspex.
with FilterServer(schema, nodes, edges, results,
                  inspect_socket="/home/user/.auspex/my-filter.sock") as graph:
    # Submit measurements as usual; the separate viewer can attach meanwhile.
    ...
```

With `auspex-v2` installed and the optional `gui` dependencies (`pip install -e '.[gui]'`), launch from the repository root:

```sh
python utils/filter_graph_viewer.py /home/user/.auspex/my-filter.sock
```

For **manual visual testing without a server**, run `python tests/visual_filter_graph_viewer.py` from the repository root. This standalone fixture is deliberately not named `test_*.py` and is not collected by `unittest discover`. It renders nine generated filters, five results and three writers; Refresh changes a synthetic filter parameter while preserving the diagram layout. The window prominently labels the data as synthetic.

The viewer uses a dark, selectable left-to-right DAG diagram. Click acquisition, a filter, a result/preview, a writer, or a connection for details; scroll to zoom, middle-drag to pan, and press **Fit graph** to reset framing. Solid lines are required; dashed lines denote lossy previews. **Refresh** requests a new complete snapshot manually; there are no subscriptions, polling, editing controls, or measurement arrays. A disconnected viewer keeps its last snapshot labeled as such. Parameter-only refreshes retain graph positions.

`FilterServer.inspection_snapshot()` returns the same JSON-compatible data directly without opening a socket. Each filter's `config` is its **last acknowledged** configuration; rejected changes do not appear. The snapshot contains node kinds/ports, schemas (including units/limits), ingress name, graph connections, terminal and writer destinations, revision numbers, and a capture timestamp. It does not report actual hardware state, exact worker parameter internals, per-edge occupancy, or filter progress. Lifecycle is captured once and is not a live indicator. Current configuration may contain sensitive values: allow only trusted local users to access the socket.

The socket's parent directory must be user-owned and inaccessible to group/other users; existing socket paths are never replaced. The socket is chmod `0600` before listening and is removed when the graph closes. The bounded, versioned request/reply protocol serves only `SNAPSHOT` (`{"version":1,"op":"SNAPSHOT"}` followed by a newline); it has no mutation command. Use `auspex_v2.filters.inspection.fetch_snapshot(path)` for headless tests. An explicit socket per graph avoids a misleading global “current graph.” Remote access/discovery through the future coordinator and TCP plotting are separate, deferred work.
