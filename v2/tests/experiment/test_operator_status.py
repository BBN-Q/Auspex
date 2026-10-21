"""Read-only status formatting and bounded unavailable-service behavior."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from auspex_v2.status import inspect_services, instrument_status, format_status, progress_text


class OperatorStatusTest(unittest.TestCase):
    def test_blocked_status_socket_has_bounded_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silent.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(path))
                server.listen()
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    instrument_status(path, timeout=.05)
                self.assertLess(time.monotonic() - start, 1)

    def test_one_unavailable_service_does_not_hide_the_other(self):
        with patch("auspex_v2.status.instrument_status", side_effect=ConnectionError("offline")), \
             patch("auspex_v2.status.ExperimentServerClient") as client:
            client.return_value.status.return_value = {"active": None, "degraded": "disk error"}
            status = inspect_services("instruments", "experiments")
        text = format_status(status)
        self.assertIn("offline", text)
        self.assertIn("disk error", text)
        self.assertIn("No active experiment", text)

    def test_ownership_blockers_and_progress_are_visible(self):
        status = {"instruments": {"reachable": True, "run": {"run_id": "orphan", "orphaned": True},
                  "instruments": {"source": {"driver": "SimulatedSource", "state": "UNKNOWN",
                    "live_sessions": 1, "busy": True, "operation": "connecting"}}},
                  "experiments": {"reachable": True, "run": None, "degraded": None}}
        text = format_status(status)
        for word in ("orphan", "reconcile", "recovery", "live session", "connecting"):
            self.assertIn(word, text)
        progress = {"phase": "acquisition", "completed": 2, "total": 5, "coordinates": {"bias": .2}}
        self.assertIn("2/5 points completed", progress_text(progress))
        self.assertIn("40%", progress_text(progress))
        progress["total"] = None
        self.assertNotIn("%", progress_text(progress))

    def test_json_cli_reports_both_unreachable_services(self):
        utility = Path(__file__).resolve().parents[2] / "utils" / "auspex_status.py"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(utility), "--json", "--instrument-socket",
                                     str(Path(directory) / "i"), "--experiment-socket", str(Path(directory) / "e")],
                                    capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1)
        status = json.loads(result.stdout)
        self.assertFalse(status["instruments"]["reachable"])
        self.assertFalse(status["experiments"]["reachable"])
