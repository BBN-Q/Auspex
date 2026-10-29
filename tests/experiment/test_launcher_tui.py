"""Optional Textual console tests without starting instrument workers."""

import importlib.util
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from utils.src.service_output import ServiceOutput
from utils.src.operator_commands import CommandResult


def capture(children):
    """Start launcher-style capture without involving Textual or reading a mock pipe twice."""
    output = ServiceOutput()
    seen = set()
    for process, name in children:
        if id(process.stdout) not in seen:
            output.attach(process, name)
            seen.add(id(process.stdout))
    return output


@unittest.skipUnless(importlib.util.find_spec("textual"), "install auspex-v2[tui] to test the terminal console")
class OperatorConsoleTest(unittest.IsolatedAsyncioTestCase):
    """Exercise input modes and real Textual layout with synthetic service output."""

    async def test_modes_scrollback_and_command_results(self):
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "utils"))
        try:
            from src.auspex_tui import OperatorConsole
            from textual.widgets import Input, RichLog, Static
        finally:
            sys.path.remove(str(root / "utils"))

        instrument = Mock(stdout=io.StringIO("[InstrumentServer] connected\n"))
        instrument.poll.return_value = None
        experiment = Mock(stdout=io.StringIO("[ExperimentServer] running\n"))
        experiment.poll.return_value = None
        app = OperatorConsole("instrument.sock", "experiment.sock",
                              [(instrument, "InstrumentServer"), (experiment, "ExperimentServer")],
                              output=capture([(instrument, "InstrumentServer"), (experiment, "ExperimentServer")]))
        def fake_execute(parts):
            if parts[0] in ("experiment", "instruments"):
                app.commands.mode = parts[0].title()
            return CommandResult("ok")

        with patch.object(app.commands, "execute", side_effect=fake_execute):
            async with app.run_test() as pilot:
                self.assertIsNotNone(app.query_one(RichLog))
                self.assertIn("Instruments >>", str(app.query_one(Static).render()))
                await pilot.press("l", "i", "s", "t", "enter")
                await pilot.pause(.1)
                self.assertFalse(app.busy)
                await pilot.press("e", "x", "p", "e", "r", "i", "m", "e", "n", "t", "enter")
                self.assertEqual(app.commands.mode, "Experiment")
                await pilot.press("s", "t", "a", "t", "u", "s", "enter")
                await pilot.pause(.1)
                self.assertFalse(app.busy)
                self.assertIn("Experiment >>", str(app.query_one(Static).render()))
                self.assertEqual(app.query_one(Input).value, "")

    async def test_discovery_confirmation_and_registration_name(self):
        """Identification needs yes; registration collects a name without blocking the UI."""
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "utils"))
        try:
            from src.auspex_tui import OperatorConsole
            from textual.widgets import Input
        finally:
            sys.path.remove(str(root / "utils"))
        child = Mock(stdout=io.StringIO())
        child.poll.return_value = None
        app = OperatorConsole("i", "e", [(child, "i"), (child, "e")], output=capture([(child, "i")]))
        with patch.object(app.commands, "execute", return_value=CommandResult("done")) as execute, \
             patch.object(app.commands, "registration_prompt", side_effect=lambda parts:
                          "Inventory name:" if parts[0] == "register" else None):
            async with app.run_test() as pilot:
                async def submit(text):
                    app.query_one(Input).value = text
                    await pilot.press("enter")
                    await pilot.pause(.1)

                await submit("identify all")
                execute.assert_not_called()
                await submit("n")
                execute.assert_not_called()
                await submit("identify all")
                await submit("y")
                execute.assert_called_once_with(["identify", "all"])
                execute.reset_mock()
                await submit("register a AgilentN5230A")
                self.assertIsNotNone(app.pending_registration)
                execute.assert_not_called()
                await submit("")
                execute.assert_not_called()
                await submit("register a AgilentN5230A")
                await submit("bench_vna")
                execute.assert_called_once_with(["register", "a", "AgilentN5230A", "bench_vna"])
                self.assertFalse(app.busy)

    async def test_clear_and_live_release_require_confirmation(self):
        """Confirm revocation with cached owner identity without blocking on IPC."""
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "utils"))
        try:
            from src.auspex_tui import OperatorConsole
        finally:
            sys.path.remove(str(root / "utils"))
        child = Mock(stdout=io.StringIO())
        child.poll.return_value = None
        app = OperatorConsole("instruments.sock", "experiments.sock",
                              [(child, "InstrumentServer"), (child, "ExperimentServer")],
                              output=capture([(child, "InstrumentServer")]))
        with patch.object(app.commands, "execute", return_value=CommandResult("cleared")) as command:
            async with app.run_test() as pilot:
                await pilot.press("c", "l", "e", "a", "r", "space", "a", "l", "l", "enter")
                self.assertIsNotNone(app.pending_confirmation)
                command.assert_not_called()
                await pilot.press("n", "enter")
                command.assert_not_called()
                await pilot.press("c", "l", "e", "a", "r", "space", "a", "l", "l", "enter")
                await pilot.press("y", "enter")
                await pilot.pause(.1)
                command.assert_called_once()
                command.reset_mock()
                app.commands.names = ["vna"]
                app.commands.owners = {"vna": {"id": "abc1234", "label": "notebook",
                                               "revoking": False}}
                with patch.object(app, "_log", wraps=app._log) as log:
                    await pilot.press("r", "e", "l", "e", "a", "s", "e", "space", "1", "enter")
                    self.assertIsNotNone(app.pending_confirmation)
                    self.assertIn("vna: notebook [abc1234]", log.call_args.args[0])
                    command.assert_not_called()
                    await pilot.press("n", "enter")
                    command.assert_not_called()
                    await pilot.press("r", "e", "l", "e", "a", "s", "e", "space", "1", "enter")
                    await pilot.press("y", "enter")
                    await pilot.pause(.1)
                    command.assert_called_once_with(["release", "1"])
