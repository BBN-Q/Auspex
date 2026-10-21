"""Operator messages follow durable state and cannot change run outcomes."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auspex_v2.experiment.service import Registry, announce


class TerminalLoggingTest(unittest.TestCase):
    def test_status_messages_follow_persistence_and_include_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(None)
            entry = {"id": "test-run", "folder": SimpleNamespace(path=Path(directory))}
            with patch("auspex_v2.experiment.service.announce") as output:
                registry._status(entry, "ACCEPTED")
                self.assertIn(directory, output.call_args.args[0])
                registry._status(entry, "RUNNING")
                self.assertIn("test-run: RUNNING", output.call_args.args[0])
                registry._status(entry, "FAILED", reason="example failure")
                self.assertIn("example failure", output.call_args.args[0])
                with patch("auspex_v2.experiment.service._atomic_json", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        registry._status(entry, "SUCCEEDED")
                self.assertEqual(entry["status"]["state"], "FAILED")
                self.assertIn("Cannot persist SUCCEEDED", output.call_args.args[0])

    def test_terminal_prefix_has_no_timestamp(self):
        with patch("builtins.print") as output:
            announce("Run finished")
        output.assert_called_once_with("[ExperimentServer] Run finished", flush=True)

    def test_broken_terminal_is_not_a_run_failure(self):
        with patch("builtins.print", side_effect=BrokenPipeError):
            announce("Run finished")
