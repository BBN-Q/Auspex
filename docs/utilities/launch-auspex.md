# Combined launcher

`launch_auspex.py` starts InstrumentServer, waits for readiness, then starts
ExperimentServer. It owns service shutdown, not experimental decisions.

```sh
python utils/launch_auspex.py
python utils/launch_auspex.py --tui
python utils/launch_auspex.py --gui
python utils/launch_auspex.py --tui --gui
```

`--tui` requires the `tui` extra and an interactive terminal; `--gui` requires the
`gui` extra. The plain launcher remains the default. Redirected plain operation
has no interactive command prompt.

## Operator console

Type `instruments` or `experiment` to select a mode, then `help` for its commands.
Instrument mode supports inventory/discovery/connection actions. Experiment mode
supports `status`, `runs`, and `cancel <run-id>`. Cancellation leaves the launcher
running; Ctrl+C shuts the launcher down.

Use `--instrument-socket PATH` and `--experiment-socket PATH` for explicit endpoints.
The launcher refuses services already using those endpoints rather than attaching
to or replacing them. It never performs a forced `--clean` reset.

## Shutdown

First Ctrl+C requests cooperative cancellation if needed, keeps output visible,
waits for cleanup/finalization, and stops services in order. Registrations remain.
A second press skips the cooperative wait and proceeds to shutdown; inspection
and lease reconciliation may be needed afterward. Command entry is disabled during
TUI shutdown. Closing the optional GUI alone does not stop the services.

A stuck instrument call or Python calculation can delay cooperative cancellation.
Service-stop escalation may terminate processes, but does not guarantee that
hardware output is off.

## Output

The TUI drains service output independently of display rendering. Pending output
is bounded to the newest 5,000 lines; overflow is reported, not silently hidden.
After service shutdown it waits one second total for output EOF, then warns if a
descendant may still hold a stream open. Such a warning requires inspection; it is
not proof that a physical operation stopped. Diagnostic files are separate from
this display buffer.

See the [quickstart](../quickstart.md) and the retained
[launcher notes](https://github.com/BBN-Q/Auspex/blob/auspex-v2/utils/README.md).
