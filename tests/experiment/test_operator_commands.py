"""Plain launcher console and shared command parsing without a TUI dependency."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "utils"))
from src import operator_commands
spec = importlib.util.spec_from_file_location("auspex_launcher_prompt", ROOT / "utils" / "launch_auspex.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
sys.path.remove(str(ROOT / "utils"))


class OperatorCommandsTest(unittest.TestCase):
    """Verify that plain-console controls use the same state and confirmations as the TUI."""

    def test_prompt_modes_ids_and_confirmations(self):
        """Resolve listed IDs, route experiment queries, and confirm destructive actions."""
        class InstrumentClient:
            """Record one registered device and one clear request."""
            calls = []

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def status(self):
                return {"run": None, "instruments": {"vna": {"driver": "SimulatedVNA",
                        "physical": "SIM::VNA", "state": "DISCONNECTED", "live_sessions": 0}}}

            def connect(self, name):
                self.calls.append(("connect", name))

            def deregister(self, name):
                self.calls.append(("clear", name))

        class ExperimentClient:
            """Distinguish status from run-list dispatch without starting services."""
            calls = []

            def __init__(self, path):
                pass

            def status(self):
                self.calls.append("status")
                return {"active": "run-1"}

            def list_runs(self):
                self.calls.append("runs")
                return [{"run_id": "run-1"}]

        inputs = iter(("connect 1", "list", "connect 1", "clear 1", "n", "clear 1", "y",
                       "experiment", "status", "runs", "instruments", "recover all"))
        output = io.StringIO()

        def read_input(prompt):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None
        with patch.object(operator_commands, "InstrumentServerClient", InstrumentClient), \
             patch.object(operator_commands, "ExperimentServerClient", ExperimentClient), \
             patch("builtins.input", side_effect=read_input), \
             contextlib.redirect_stdout(output), patch.dict(launcher.os.environ, {"NO_COLOR": "1"}):
            launcher.plain_operator_prompt("instrument.sock", "experiment.sock")
        self.assertEqual(InstrumentClient.calls, [("connect", "vna"), ("clear", "vna")])
        self.assertEqual(ExperimentClient.calls, ["status", "runs"])
        self.assertIn("list first", output.getvalue())
        self.assertIn("1. vna", output.getvalue())
        self.assertIn("connect vna", output.getvalue())
        self.assertIn("ExperimentServer: {'active': 'run-1'}", output.getvalue())
        self.assertIn("[{'run_id': 'run-1'}]", output.getvalue())
        self.assertIn("Recovery must be individual", output.getvalue())
        self.assertNotIn("\033[", output.getvalue())

    def test_parse_rejects_wrong_mode_and_unsupported_shutdown(self):
        """Context gates help, parsing and execution independently of default socket paths."""
        commands = operator_commands.OperatorCommands("instruments", "experiments")
        self.assertNotIn("; shutdown", commands.execute(["help"]).text)
        with self.assertRaises(ValueError):
            commands.parse("shutdown")
        with self.assertRaises(ValueError):
            commands.execute(["shutdown"])
        with self.assertRaises(ValueError):
            commands.parse("recover all")
        commands.execute(["experiment"])
        with self.assertRaises(ValueError):
            commands.parse("connect 1")
        self.assertEqual(commands.parse("cancel run-id"), ["cancel", "run-id"])
        standalone = operator_commands.OperatorCommands(None, context="standalone")
        help_result = standalone.execute(["help"])
        self.assertIn("; shutdown", help_result.text)
        self.assertNotIn("; experiment", help_result.text)
        self.assertFalse(help_result.exit_console)
        self.assertTrue(standalone.confirmation(standalone.parse("shutdown")))
        for line in ("experiment", "instruments", "status", "runs", "cancel run-id"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                standalone.parse(line)
        with self.assertRaises(ValueError):
            standalone.execute(["experiment"])
