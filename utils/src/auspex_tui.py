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
    BINDINGS = [("ctrl+c", "quit", "Stop launcher (cooperative cancellation)")]

    def __init__(self, instrument_socket, experiment_socket, children):
        super().__init__()
        self.commands = OperatorCommands(instrument_socket, experiment_socket)
        self.services = children
        self.pending_confirmation = None
        self.busy = False
        self.color = "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"

    def compose(self) -> ComposeResult:
        yield RichLog(id="scroll", max_lines=5000, wrap=True, highlight=False,
                      markup=False, auto_scroll=True)
        yield Static("", id="mode")
        yield Input(placeholder="Type help for commands", id="command")

    def on_mount(self):
        self._prompt()
        self._log("[Auspex] Both services ready. Type help; Ctrl+C cancels a run and stops supervision.")
        for process, name in self.services[:2]:
            threading.Thread(target=self._read_output, args=(process,), daemon=True).start()
        self.set_interval(0.2, self._check_children)
        self.query_one(Input).focus()

    def _read_output(self, process):
        """Drain each child continuously so a chatty service never blocks on its pipe."""
        for line in process.stdout:
            try:
                self.call_from_thread(self._log, line.rstrip("\n"))
            except RuntimeError:
                return  # Console closed; the launcher still owns service shutdown.

    def _check_children(self):
        if any(process.poll() is not None for process, _ in self.services[:2]):
            self.exit()

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
        if self.commands.confirmation(parts):
            self.pending_confirmation = parts
            self._log(f"[Auspex] Confirm {line}? Type y or n:")
            return
        if parts[0].lower() in ("experiment", "instruments", "help"):
            self._log(f"[Auspex] {self.commands.execute(parts)}")
            self._prompt()
            return
        self._dispatch(parts)

    def _dispatch(self, parts):
        self.busy = True
        self.query_one(Input).disabled = True
        threading.Thread(target=self._execute, args=(parts,), daemon=True).start()

    def _execute(self, parts):
        """Keep synchronous IPC off the Textual event loop, including slow connect/disconnect."""
        try:
            try:
                message = f"[Auspex] {self.commands.execute(parts)}"
            except Exception as exc:
                message = f"[Auspex] {parts[0]} failed: {exc}"
            self.call_from_thread(self._log, message)
            self.call_from_thread(self._command_finished)
        except RuntimeError:
            pass  # The operator closed the console while a request was in flight.

    def _command_finished(self):
        self.busy = False
        field = self.query_one(Input)
        field.disabled = False
        field.focus()
