"""Presentation and prefix resolution contracts independent of hardware timing."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from auspex_v2.experiment.client import ExperimentServerClient
from auspex_v2.experiment.protocol import PROTOCOL_VERSION
from auspex_v2.experiment.service import Registry
from auspex_v2.run_display import format_experiment_summary, format_runs, run_label


class RunDisplayTest(unittest.TestCase):
    """Keep display abbreviations separate from canonical catalog identities."""

    def test_prefix_collision_is_rejected_before_any_action(self):
        """Two identical displayed prefixes must never select an arbitrary run."""
        registry = Registry(None)
        ids = ("1abc4de0" + "0" * 24, "1abc4de1" + "0" * 24)
        for index, run_id in enumerate(ids):
            path = Path(f"/data/vna_sweep_{index:03}.auspex")
            registry.runs[run_id] = {"id": run_id, "folder": SimpleNamespace(path=path),
                                     "status": {"run_id": run_id, "path": str(path), "state": "SUCCEEDED"}}
        registry.active = ids[0]
        for operation in ("status", "log", "plan", "cancel", "cleanup"):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "Ambiguous"):
                registry.dispatch({"version": PROTOCOL_VERSION, "op": operation, "run_id": "1abc4de"})
        self.assertEqual(registry.active, ids[0])
        self.assertEqual(tuple(registry.runs), ids)
        for query, full in ((ids[0], ids[0]), ("1abc4de0", ids[0]), ("1abc4de1", ids[1])):
            result = registry.dispatch({"version": PROTOCOL_VERSION, "op": "status", "run_id": query})
            self.assertEqual(result["run_id"], full)
        for invalid in (None, "", 42, "missing"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                registry.dispatch({"version": PROTOCOL_VERSION, "op": "status", "run_id": invalid})

    def test_readable_summary_empty_degraded_and_failure_details(self):
        """Display diagnostics and progress without dumping transport dictionaries or path fields."""
        run_id = "1abc4de" + "0" * 25
        path = "/private/data/vna_sweep_003.auspex"
        run = {"run_id": run_id, "path": path, "state": "FAILED", "reason": "Acquisition failed",
               "diagnostic": "Log unavailable", "progress": {"phase": "teardown", "completed": 2, "total": 5}}
        text = format_experiment_summary({"active": None, "degraded": "Status persistence unavailable"}, [run])
        for expected in ("ExperimentServer: degraded", "Active run: none", "vna_sweep_003 [1abc4de]",
                         "FAILED", "2/5 points completed", "40%", "Acquisition failed", "Log unavailable"):
            self.assertIn(expected, text)
        self.assertNotIn(path, text)
        self.assertNotIn(run_id, text)
        self.assertEqual(run_label(None, "/data/manual.auspex"), "manual")
        self.assertEqual(run_label(run_id), "[1abc4de]")
        self.assertEqual(format_runs([]), "  No known runs.")
        self.assertIn("ExperimentServer: ready", format_experiment_summary({"active": None}, []))
        with tempfile.TemporaryDirectory() as directory:
            client = ExperimentServerClient(Path(directory) / "absent.sock")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIsNone(client.summary())
            self.assertIn("ExperimentServer unavailable:", output.getvalue())
