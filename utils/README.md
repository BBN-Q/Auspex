# Auspex V2 utilities

Run these entry points from the repository root with `python utils/<name>.py`. GUI programs require the optional GUI dependencies; `launch_auspex.py --tui` requires the optional Textual dependency.

| Program | Purpose |
| --- | --- |
| `launch_auspex.py` | Start and supervise both local servers. Offers modal instrument/experiment commands on an interactive terminal, or an opt-in `--tui`; stays headless with redirected input. |
| `instrument_server.py` | Start only InstrumentServer, with its interactive operator commands and optional `--gui`. |
| `experiment_server.py` | Start only ExperimentServer for submitted experiment plans. |
| `instrument_server_gui.py` | Attach the standalone instrument inventory/connection GUI to a running server. |
| `auspex_status.py` | Read-only status of both services, optionally as JSON. |
| `data_browser.py` | Browse `.auspex` run folders, inspect metadata, and plot supported 1D datasets. |
| `filter_graph_viewer.py` | Attach a read-only graph/configuration viewer to an enabled filter inspection socket. |

`src/` contains the non-executable launcher helpers: `operator_commands.py` shares IPC command parsing between plain and Textual modes; `auspex_tui.py` implements the optional Textual display. Service, driver, and data-library code lives in `../src/auspex_v2/`.
