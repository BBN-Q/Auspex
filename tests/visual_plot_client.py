"""Attended live-plot demo, using real spawned filters and synthetic acquisition.

Run: python tests/visual_plot_client.py
Two sequential runs add trace/map tabs without stealing focus. Try component
modes, colormaps, cuts, closing/reopening tabs and permanent Close plot. Closing
the GUI does not cancel the finite acquisition. No services/hardware are used.
"""

import argparse
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
from PySide6 import QtWidgets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from plot_client import PlotClient, dark_theme
from auspex_v2.data import Axis
from auspex_v2.filters import ManualPipeline, PortSchema
from auspex_v2.filters.definitions import (
    ComplexAverage, PipelineDefinition, PipelineInput, Plot, WriteToFile,
)


def simulated_runs(paths, *, delay=0.08, points=41):
    """Drive actual filter/store processes; preview clients are optional observers."""
    rng = np.random.default_rng(418)
    frequency = 5 + 0.1 * np.linspace(-1, 1, 201)**3
    biases = np.linspace(-0.6, 0.6, points)
    for run_number, path in enumerate(paths):
        source = PipelineInput("synthetic_vna", PortSchema(
            "complex128", 1, ("frequency",), ("GHz",), 65536, value_name="S21"))
        average = ComplexAverage(name="average", repeats=3)
        writer = WriteToFile(name="disk", group="measurement", dataset="s21")
        trace = Plot(name="current_trace")
        image = Plot(name="bias_map", mode="map")
        definition = PipelineDefinition(inputs=[source], edges=[
            (source, average.sink), (average.final_average, writer.sink),
            (average.partial_average, trace.sink), (average.final_average, image.sink)])
        with ManualPipeline(definition, store_path=path,
                            preview_sweep=Axis("bias", biases, "V"),
                            run_metadata={"name": f"Synthetic resonance {run_number + 1}", "synthetic": True,
                                          "comments": "Not physical measurements; partial trace and finalized map."}) as pipeline:
            for bias in biases:
                center = 5 + 0.07 * bias + run_number * 0.008
                response = (1 - 0.8 / (1 + 1j * (frequency - center) / 0.006)) * np.exp(1j * bias * 8)
                for _ in range(3):
                    noise = rng.normal(0, 0.006, frequency.size) + 1j * rng.normal(0, 0.006, frequency.size)
                    pipeline.submit(response + noise, axes=[Axis("frequency", frequency, "GHz")],
                                    sweep={"bias": bias}, sweep_units={"bias": "V"})
                    time.sleep(delay)
        print(f"Synthetic pipeline complete: {path}", flush=True)


def main(argv=None):
    """Run a finite desktop demo without overwriting existing synthetic or real data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help="New destination directory; defaults to temporary data")
    args = parser.parse_args(argv)
    temporary = tempfile.TemporaryDirectory(prefix="auspex-live-demo-") if args.directory is None else None
    root = Path(temporary.name) if temporary else args.directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"resonance_{i:03d}.auspex" for i in range(2)]
    if any(path.exists() for path in paths):
        parser.error("Demo destinations already exist; choose a new directory")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dark_theme(app)
    window = PlotClient(paths, discover=False)
    window.setWindowTitle("Auspex V2 — Live plotting demo (SYNTHETIC)")
    worker = threading.Thread(target=simulated_runs, args=(paths,))
    worker.start()
    window.show()
    result = app.exec()
    worker.join()  # GUI closure is not acquisition cancellation.
    if temporary:
        temporary.cleanup()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
