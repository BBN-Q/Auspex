"""Optional Textual console tests without starting instrument workers."""

import importlib.util
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from utils.src.service_output import ServiceOutput


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
            return "ok"

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

    async def test_instrument_ids_and_experiment_requests(self):
        """Keep command targets tied to the last list and route experiment queries separately."""
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "utils"))
        try:
            from src import auspex_tui
        finally:
            sys.path.remove(str(root / "utils"))

        class Instruments:
            """Minimal session-free inventory for command routing."""

            def __init__(self, path):
                self.state = {"vna": "DISCONNECTED"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def status(self):
                return {"run": None, "instruments": {"vna": {"driver": "SimulatedVNA",
                        "physical": "SIM::VNA", "state": self.state["vna"], "live_sessions": 0}}}

            def connect(self, name):
                self.state[name] = "CONNECTED"

        class Experiments:
            """Report a known active run without submitting anything."""

            def __init__(self, path):
                pass

            def status(self):
                return {"active": "run-1"}

            def list_runs(self):
                return [{"run_id": "run-1"}]

        app = auspex_tui.OperatorConsole("i", "e", [], output=ServiceOutput())
        from src import operator_commands
        with patch.object(operator_commands, "InstrumentServerClient", Instruments), \
             patch.object(operator_commands, "ExperimentServerClient", Experiments):
            with self.assertRaisesRegex(ValueError, "list first"):
                app.commands.execute(["connect", "1"])
            self.assertIn("1. vna", app.commands.execute(["list"]))
            self.assertIn("connect vna", app.commands.execute(["connect", "1"]))
            app.commands.execute(["experiment"])
            self.assertIn("run-1", app.commands.execute(["status"]))
            self.assertIn("run-1", app.commands.execute(["runs"]))
            app.commands.execute(["instruments"])
            with self.assertRaisesRegex(ValueError, "Recovery must be individual"):
                app.commands.parse("recover all")

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
        with patch.object(app.commands, "execute", return_value="done") as execute, \
             patch.object(app.commands, "registration_prompt", return_value="Inventory name:"):
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

    async def test_clear_requires_confirmation(self):
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
        with patch.object(app.commands, "execute", return_value="cleared") as command:
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
