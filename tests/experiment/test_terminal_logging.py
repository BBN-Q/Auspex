"""Operator messages follow durable state and cannot change run outcomes."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auspex_v2.experiment.service import Registry, announce


class TerminalLoggingTest(unittest.TestCase):
    def test_status_messages_follow_persistence_and_include_identity(self):
        """Human labels omit hashes/paths while durable status retains canonical identity."""
        with tempfile.TemporaryDirectory() as directory:
            directory = str(Path(directory) / "run with 'quotes'.auspex")
            Path(directory).mkdir()
            registry = Registry(None)
            run_id = "1abc4de" + "0" * 25
            entry = {"id": run_id, "folder": SimpleNamespace(path=Path(directory))}
            label = "run with 'quotes' [1abc4de]"
            with patch("auspex_v2.experiment.service.announce") as output:
                registry._status(entry, "ACCEPTED")
                self.assertIn(label + ": ACCEPTED", output.call_args.args[0])
                registry._status(entry, "RUNNING")
                self.assertIn(label + ": RUNNING", output.call_args.args[0])
                registry._status(entry, "FAILED", reason="example failure")
                self.assertTrue(any("example failure" in call.args[0] for call in output.call_args_list))
                self.assertIn(label + ": FAILED", output.call_args.args[0])
                for call in output.call_args_list:
                    self.assertNotIn(run_id, call.args[0])
                    self.assertNotIn(directory, call.args[0])
                saved = json.loads((Path(directory) / "status.json").read_text())
                self.assertEqual(saved["run_id"], run_id)
                self.assertEqual(saved["path"], directory)
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
