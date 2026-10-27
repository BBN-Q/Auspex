# Utilities

Run entry points from the repository root with the same Python environment as
Auspex. `utils/src/` contains shared implementation helpers, not additional tools.
Use each program's `--help` for its current options.

| Utility | Role | Extra |
| --- | --- | --- |
| [launch_auspex.py](launch-auspex.md) | Start/supervise both services; shared operator console | `tui` for `--tui`, `gui` for `--gui` |
| [instrument_server.py](instrument-server.md) | Instrument service and interactive inventory commands | `visa` for VISA hardware/discovery; optional `gui` |
| [instrument_server_gui.py](instrument-server-gui.md) | Attach an inventory/discovery window | `gui` |
| [experiment_server.py](experiment-server.md) | Execute submitted plans | None for simulated runs |
| [auspex_status.py](auspex-status.md) | Read-only service/ownership inspection | None |
| [data_browser.py](data-browser.md) | Read saved data/metadata; plot supported 1D/2D arrays | `gui` |
| [plot_client.py](live-plots.md) | Follow optional live trace/map endpoints in one window | `gui` |
| [filter_graph_viewer.py](filter-graph-viewer.md) | Inspect an explicitly enabled running graph | `gui` |

## Shared paths

`AUSPEX_HOME` selects a base directory; otherwise HOME is used and startup announces
the fallback. Sockets and per-launch diagnostic logs normally live in its
`.auspex/` subdirectory. The discovery whitelist instead lives directly at
`$AUSPEX_HOME/instrument_whitelist.txt` (or `$HOME/instrument_whitelist.txt`).

Use the same environment setting in launchers, scripts, notebooks, and inspection
tools. Socket overrides do not relocate the diagnostic log directory. Logs are
truncated on successful service startup and are not durable operator history.
Run execution logs remain separately in each run directory.

V2 services are local programs, not authenticated network services. Do not create
independent registrations for the same physical device through multiple endpoints
or unmanaged driver instances.

The retained [utility map](https://github.com/BBN-Q/Auspex/blob/auspex-v2/utils/README.md)
provides additional implementation notes.
