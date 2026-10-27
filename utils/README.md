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

In either launcher mode, **Ctrl+C shuts down the launcher**, not just the current
prompt: it cancels an active run, waits for teardown/finalization, then stops the
services in order while preserving registrations. The TUI stays visible and streams
service output until shutdown completes. **Ctrl+C again skips the cooperative wait**
and proceeds to service shutdown; it does not guarantee hardware outputs are off,
and leases may need reconciliation. Command entry is disabled during TUI shutdown.

Instrument discovery uses `enumerate visa [--serial] [--prologix]
[--show-permission-denied]` in either launcher console and the standalone instrument
terminal. Serial and Prologix candidates are hidden by default; confirmed
permission-denied resources are also hidden. Whitelisted addresses from the server's
`$AUSPEX_HOME/instrument_whitelist.txt` (HOME fallback) remain visible. The file is
one address per line, ignoring blank/`#` comment lines, and is reloaded on enumeration.
`identify all` confirms and probes visible candidates only, always skipping serial,
Prologix and registered resources. `identify <letter>` can explicitly retry a hidden
candidate; letters stay stable until enumeration. See the
[operator guide](../src/auspex_v2/instruments/server/instrument_server.md#discovery-whitelist-and-permission-errors).

For `--tui`, the launcher starts service-output readers immediately after spawning
each service, before readiness checks. A shared buffer retains the newest **5,000
pending lines** independently of the UI; overflow is reported with a dropped-line
count. The TUI displays buffered startup output and polls for new lines. Service
log files remain independent of this best-effort terminal display.

After ordered service shutdown, output readers get **one second total** to reach
EOF. If output remains open (for example, inherited by a surviving descendant),
the TUI warns to inspect hardware/leases and exits without waiting indefinitely.
No descendant is killed merely for keeping stdout open. Existing service-stop
timeouts/escalation still apply. Startup failures also print retained output even
if the TUI never mounted. The plain CLI output path is unchanged.

`src/` contains the non-executable console helpers. `operator_commands.py` owns
operator syntax, contextual help, confirmations, registration prompts, inventory and
discovery snapshots, IPC execution, and common result text for the standalone
instrument terminal and both launcher consoles. Frontends read/display input and
act on `CommandResult.text` / `exit_console`; the command module never reads stdin
or controls UI threads. Explicit `launcher` / `standalone` contexts keep experiment
mode and standalone inventory-clearing `shutdown` distinct. Launcher Ctrl+C still
uses its separate cooperative, registration-preserving shutdown policy.

`auspex_tui.py` implements the optional Textual display; `service_output.py` owns
bounded service-pipe capture without depending on Textual. Service, driver, and
data-library code lives in `../src/auspex_v2/`.
