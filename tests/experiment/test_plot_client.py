"""Retention, GUI lifecycle and shared-renderer checks for detached live plots."""

import copy
import json
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from utils.src.plot_client_model import PlotCatalog


class PlotRetentionTest(unittest.TestCase):
    """The agreed 64-plot policy is independent of Qt and transport."""

    def test_cap_terminal_eviction_active_rejection_and_session_suppression(self):
        """Count closed listings too; eviction is terminal-only and removal suppresses rediscovery."""
        catalog = PlotCatalog()
        for index in range(64):
            self.assertTrue(catalog.admit((str(index), "trace"), {"order": index, "state": "RUNNING"})[0])
        key = ("new", "trace")
        self.assertEqual(catalog.admit(key, {"order": 100, "state": "RUNNING"}), (False, None))
        self.assertEqual(len(catalog.plots), 64)
        catalog.plots[("3", "trace")]["state"] = "FAILED"
        catalog.plots[("7", "trace")]["state"] = "SUCCEEDED"
        self.assertEqual(catalog.admit(key, {"order": 100, "state": "RUNNING"}), (True, ("3", "trace")))
        self.assertIn(("3", "trace"), catalog.suppressed)
        self.assertEqual(catalog.admit(("3", "trace"), {}), (False, None))
        catalog.remove(key)
        self.assertNotIn(key, catalog.plots)
        self.assertEqual(catalog.admit(key, {}), (False, None))
        self.assertTrue(PlotCatalog().admit(key, {})[0])


@unittest.skipUnless(importlib.util.find_spec("PySide6") and importlib.util.find_spec("pyqtgraph"),
                     "install auspex-v2[gui] for live plot tests")
class PlotClientGuiTest(unittest.TestCase):
    """Use real pipeline frames; test UI behavior without mocking external services."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
        import plot_client
        cls.module = plot_client
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_real_pipeline_frames_tabs_metadata_and_shared_view_state(self):
        """New runs do not steal focus; updates preserve view choices and cuts."""
        from auspex_v2.data import Axis
        from auspex_v2.filters import ManualPipeline
        from auspex_v2.filters.preview import encode_frame, decode_frame
        from auspex_v2.filters.preview_client import poll_run
        from tests.filters.test_live_preview import recipe
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.auspex"
            with ManualPipeline(recipe(repeats=1), store_path=path,
                                preview_sweep=Axis("bias", [0., 1., 3.], "V")) as pipeline:
                for bias in [0., 1., 3.]:
                    pipeline.submit(np.array([1+2j, 3+4j, 5+6j]) + bias,
                        axes=[Axis("frequency", [2., 4., 7.], "Hz")],
                        sweep={"bias": bias}, sweep_units={"bias": "V"})
            run_id = "1abc4de" + "0" * 25
            (path / "status.json").write_text(json.dumps({"run_id": run_id, "state": "SUCCEEDED"}))
            window = self.module.PlotClient([path], discover=False, auto_poll=False)
            try:
                initial = poll_run(path)
                window.apply_updates([initial])
                self.assertEqual(window.tabs.count(), 2)
                self.assertIn("run [1abc4de]", window.tree.topLevelItem(0).text(0))
                self.assertIn("run [1abc4de]", window.tabs.tabText(0))
                self.assertNotIn(run_id, window.tree.topLevelItem(0).text(0))
                run = poll_run(path, wanted=["bias_map", "current_trace"])
                window.apply_updates([run])
                window.show()
                self.app.processEvents()
                trace_key, map_key = (run["id"], "current_trace"), (run["id"], "bias_map")
                trace = window.open_tabs[trace_key].renderer
                image = window.open_tabs[map_key].renderer
                self.assertIsNotNone(trace)
                self.assertIsNotNone(image)
                self.assertIsInstance(trace, self.module.DataTab)
                trace.component.setCurrentText("Imaginary")
                image.component.setCurrentText("Phase")
                image.colormap.setCurrentText("viridis")
                image.x_cut.setChecked(True)
                image.y_cut.setChecked(True)
                image.lines["X"].setValue(1.)
                plot = image.panels[0][0]
                plot.setRange(xRange=[2.5, 6.0], yRange=[0.2, 2.2], padding=0)
                bounds = plot.viewRange()
                updated = copy.copy(run)
                updated["frames"] = {}
                for name, (dataset, metadata) in run["frames"].items():
                    frame = replace(dataset, data=dataset.data * (1 + 1j))
                    updated["frames"][name] = decode_frame(encode_frame(frame, {**metadata, "sequence": metadata["sequence"] + 1}))
                window.apply_updates([updated])
                self.assertIs(window.open_tabs[map_key].renderer, image)
                self.assertEqual(image.component.currentText(), "Phase")
                self.assertEqual(image.colormap.currentText(), "viridis")
                self.assertEqual(image.lines["X"].value(), 1.)
                np.testing.assert_allclose(plot.viewRange(), bounds)
                np.testing.assert_allclose(image.cut_curves["X"][1].yData,
                                           np.angle(updated["frames"]["bias_map"][0].data)[1])
                self.assertEqual(trace.component.currentText(), "Imaginary")
                window.tabs.setCurrentWidget(window.open_tabs[map_key])
                focused = window.tabs.currentWidget()
                second_id = "2abc4de" + "0" * 25
                second = {**run, "id": second_id, "name": "New experiment",
                          "path": str(Path(directory) / "new_experiment_003.auspex"),
                          "status": {"run_id": second_id, "state": "SUCCEEDED"},
                          "order": run["order"] + 1}
                window.apply_updates([second])
                self.assertEqual(window.tabs.count(), 4)
                self.assertIs(window.tabs.currentWidget(), focused)
                self.assertIn("new_experiment_003 [2abc4de]", window.tree.topLevelItem(0).text(0))
                self.assertGreater(window.metadata.topLevelItemCount(), 0)
                window.close_tab(window.tabs.indexOf(focused))
                self.assertIn(map_key, window.catalog.plots)
                self.assertNotIn(map_key, window.open_tabs)
                window.apply_updates([updated])
                self.assertNotIn(map_key, window.open_tabs)  # no automatic reopening
                window.open_plot(map_key)
                self.assertEqual(window.tabs.currentWidget().key, map_key)
                window.tab_selected(window.tabs.currentIndex())
                window.remove_selected()
                self.assertNotIn(map_key, window.catalog.plots)
                window.apply_updates([updated])
                self.assertNotIn(map_key, window.catalog.plots)
                self.assertIn(map_key, window.catalog.suppressed)
            finally:
                window.close()

    def test_visual_demo_runs_real_pipelines_and_client_can_poll_late(self):
        """The attended script's data path is executable without desktop interaction."""
        from auspex_v2.filters.preview_client import poll_run
        script = Path(__file__).resolve().parents[1] / "visual_plot_client.py"
        spec = importlib.util.spec_from_file_location("visual_plot_client", script)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"demo_{i}.auspex" for i in range(2)]
            demo.simulated_runs(paths, delay=0, points=3)
            window = self.module.PlotClient(paths, discover=False, auto_poll=False)
            try:
                window.tick()
                window.worker.join(5)
                window.tick()
                self.assertEqual(window.tabs.count(), 4)
                window.worker.join(5)
                window.tick()
                self.assertTrue(all(p.renderer is not None for p in window.open_tabs.values()))
                for path in paths:
                    run = poll_run(path, wanted=["bias_map"])
                    self.assertEqual(run["frames"]["bias_map"][0].data.shape, (3, 201))
            finally:
                window.close()
