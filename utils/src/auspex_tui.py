"""Optional full-screen operator console for the two independently running services."""

import os
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog, Static

from .operator_commands import OperatorCommands


class OperatorConsole(App):
    """Keep one chronological service log above a mode-specific command line."""

    CSS = """
    Screen { layout: vertical; }
    #scroll { height: 1fr; border: solid $primary; }
    #mode { height: 1; }
    #command { dock: bottom; }
    """
    BINDINGS = [("ctrl+c", "shutdown", "Cancel run, wait, then stop services; press again to skip wait")]

    def __init__(self, instrument_socket, experiment_socket, children, *, output, shutdown=None):
        super().__init__()
        self.commands = OperatorCommands(instrument_socket, experiment_socket, context="launcher")
        self.services = children
        self.shutdown_callback = shutdown
        self.shutting_down = False
        self.stop_wait = threading.Event()
        self.output = output
        self.pending_confirmation = None
        self.pending_registration = None
        self.busy = False
        self.color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"

    def compose(self) -> ComposeResult:
        yield RichLog(id="scroll", max_lines=5000, wrap=True, highlight=False,
                      markup=False, auto_scroll=True)
        yield Static("", id="mode")
        yield Input(placeholder="Type help for commands", id="command")

    def on_mount(self):
        self._prompt()
        self._drain_output()
        self._log("[Auspex] Both services ready. Type help. Ctrl+C shuts down the launcher: cancels an active run, "
                  "waits for teardown, then stops services. Press again to skip the wait. "
                  "The console stays visible until services stop; Ctrl+C does not just clear the prompt.")
        self.set_interval(0.05, self._drain_output)
        self.set_interval(0.2, self._check_children)
        self.query_one(Input).focus()

    def _drain_output(self):
        """Display buffered service output; pipe readers never depend on the UI event loop."""
        for line in self.output.drain():
            self._log(line)

    def _check_children(self):
        if not self.shutting_down and any(process.poll() is not None for process, _ in self.services[:2]):
            self.exit()

    def action_shutdown(self):
        """First Ctrl+C starts visible shutdown; a second only skips the cooperative wait."""
        if self.shutting_down:
            if not self.stop_wait.is_set():
                self.stop_wait.set()
                self._log("[Auspex] Second interrupt: skipping cooperative wait and stopping supervision. "
                          "Hardware may remain active; leases may need reconciliation.")
            return
        if self.shutdown_callback is None:
            self.exit()  # Standalone embedding may leave shutdown to its caller.
            return
        self.shutting_down = True
        self.pending_confirmation = None
        self.pending_registration = None
        field = self.query_one(Input)
        field.value = ""
        field.disabled = True
        self.query_one("#mode", Static).update("Shutting down — Ctrl+C again skips the cooperative wait")
        self._log("[Auspex] Shutdown requested: cancelling any active run and waiting for teardown. "
                  "Ctrl+C again skips the wait; neither action guarantees hardware outputs are off.")
        threading.Thread(target=self._shutdown_services, daemon=True).start()

    def _shutdown_services(self):
        """Run launcher cleanup off the event loop and drain service output before exiting."""
        def report(message):
            self.call_from_thread(self._log, f"[Auspex] {message}")

        result = 0
        try:
            self.shutdown_callback(self.stop_wait, report)
            self.call_from_thread(self._drain_output)
            report("Services stopped. Exiting console.")
        except Exception as exc:
            result = 1
            report(f"Shutdown failed: {exc}. Inspect hardware and leases; launcher cleanup will follow.")
        self.call_from_thread(self.exit, result)

    def _log(self, message):
        if message.startswith("[InstrumentServer]"):
            style = "cyan"
        elif message.startswith("[ExperimentServer]"):
            style = "magenta"
        else:
            style = ""
        self.query_one(RichLog).write(Text(message, style=style if self.color else ""))

    def _prompt(self):
        mode = self.commands.mode
        self.query_one("#mode", Static).update(Text(
            f"{mode} >>", style=("bold cyan" if mode == "Instruments" else "bold magenta")
            if self.color else ""))

    def on_input_submitted(self, event: Input.Submitted):
        line = event.value.strip()
        event.input.value = ""
        if self.shutting_down:
            return
        if self.pending_registration is not None:
            parts = self.pending_registration
            self.pending_registration = None
            if line:
                self._dispatch([*parts, line])
            else:
                self._log("[Auspex] Cancelled")
            return
        if not line:
            return
        self._log(f"{self.commands.mode} >> {line}")
        if self.pending_confirmation is not None:
            action = self.pending_confirmation
            self.pending_confirmation = None
            if line.lower() != "y":
                self._log("[Auspex] Cancelled (answer y to confirm).")
            else:
                self._dispatch(action)
            return
        if self.busy:
            self._log("[Auspex] Wait for the previous command to finish.")
            return
        try:
            parts = self.commands.parse(line)
        except ValueError as exc:
            self._log(f"[Auspex] {exc}")
            return
        try:
            prompt = self.commands.registration_prompt(parts)
            confirmation_text = self.commands.confirmation_text(parts)
        except ValueError as exc:
            self._log(f"[Auspex] {exc}")
            return
        if prompt is not None:
            self.pending_registration = parts
            self._log(f"[Auspex] {prompt}")
            return
        if self.commands.confirmation(parts):
            self.pending_confirmation = parts
            self._log(f"[Auspex] Confirm {confirmation_text}? Type y or n:")
            return
        if self.commands.is_local(parts):
            self._show_result(self.commands.execute(parts))
            self._prompt()
            return
        self._dispatch(parts)

    def _show_result(self, result):
        """Render common command text; only the frontend decides how its console exits."""
        self._log(f"[Auspex] {result.text}")
        if result.exit_console:
            self.exit(0)

    def _dispatch(self, parts):
        self.busy = True
        self.query_one(Input).disabled = True
        threading.Thread(target=self._execute, args=(parts,), daemon=True).start()

    def _execute(self, parts):
        """Keep synchronous IPC off the Textual event loop, including slow connect/disconnect."""
        try:
            try:
                result = self.commands.execute(parts)
            except Exception as exc:
                self.call_from_thread(self._log, f"[Auspex] Command failed: {exc}")
            else:
                self.call_from_thread(self._show_result, result)
            self.call_from_thread(self._command_finished)
        except RuntimeError:
            pass  # The operator closed the console while a request was in flight.

    def _command_finished(self):
        self.busy = False
        field = self.query_one(Input)
        field.disabled = self.shutting_down
        if not self.shutting_down:
            field.focus()
