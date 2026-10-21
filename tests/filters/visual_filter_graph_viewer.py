"""Manual visual fixture for the standalone filter graph viewer (NOT a unittest).

Run explicitly from the repository root with a desktop session and V2's GUI extra:
    python tests/visual_filter_graph_viewer.py

Creates no FilterServer or socket. Refresh generates a fresh *synthetic* snapshot
with changed, acknowledged-looking parameters to exercise stable layout and details.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count
from pathlib import Path
import sys

# The viewer is deliberately a standalone script in utils, not an importable GUI
# dependency of the headless filter package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "utils"))
import filter_graph_viewer as viewer

from PySide6 import QtWidgets


SCHEMA = {
    "dtype": "<c8", "rank": 1, "axis_names": ["frequency"],
    "axis_units": ["Hz"], "max_bytes": 65536,
    "value_name": "S21", "value_unit": "dimensionless",
}


def make_snapshot(revision=0):
    """Build a plausible fan-out DAG with multiple filters, previews, and writers."""
    nodes = {}

    def add(name, kind="passthrough", config=None):
        nodes[name] = {
            "kind": kind, "config": config or {}, "revision": revision if config else 0,
            "schema": dict(SCHEMA), "inputs": ["in"],
            "outputs": ["out"] if kind == "passthrough" else ["partial", "final"],
        }

    add("raw_trace")
    add("reference_trace")
    add("archive_raw")
    add("average_5", "complex_average", {"repeats": 5 + revision, "group_name": "mean"})
    add("average_3", "complex_average", {"repeats": 3, "group_name": "comparison"})
    add("processed_mean")
    add("quality_check")
    add("reference_check")
    add("decision_input")

    edges = []

    def link(source, port, target, preview=False):
        edges.append({"source": source, "port": port, "target": target,
                      "target_port": "in", "preview": preview})

    link("$input", "out", "raw_trace")
    link("$input", "out", "reference_trace")
    link("raw_trace", "out", "archive_raw")
    link("raw_trace", "out", "average_5")
    link("raw_trace", "out", "average_3")
    link("average_5", "final", "processed_mean")
    link("average_3", "final", "quality_check")
    link("reference_trace", "out", "reference_check")
    link("processed_mean", "out", "decision_input")
    link("average_5", "partial", "result:live_mean", preview=True)
    link("average_3", "partial", "result:comparison_preview", preview=True)
    link("quality_check", "out", "result:quality")
    link("decision_input", "out", "result:decision")
    link("reference_check", "out", "result:reference")
    link("archive_raw", "out", "writer:raw")
    link("processed_mean", "out", "writer:averaged")
    link("reference_check", "out", "writer:reference")

    return {
        "snapshot_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lifecycle": "DEMO (synthetic, not connected)",
        "input": {"name": "simulated_vna_s21", "schema": dict(SCHEMA)},
        "nodes": nodes, "edges": edges,
        "results": {
            "live_mean": {"node": "average_5", "port": "partial", "preview": True},
            "comparison_preview": {"node": "average_3", "port": "partial", "preview": True},
            "quality": {"node": "quality_check", "port": "out", "preview": False},
            "decision": {"node": "decision_input", "port": "out", "preview": False},
            "reference": {"node": "reference_check", "port": "out", "preview": False},
        },
        "writers": {
            "raw": {"node": "archive_raw", "port": "out", "group": "sweeps",
                    "prefix": "raw_", "max_scan_bytes": 67108864, "max_scan_windows": 8192},
            "averaged": {"node": "processed_mean", "port": "out", "group": "averages",
                         "prefix": "mean_", "max_scan_bytes": 67108864, "max_scan_windows": 8192},
            "reference": {"node": "reference_check", "port": "out", "group": "reference",
                          "prefix": "", "max_scan_bytes": 67108864, "max_scan_windows": 8192},
        },
    }


def main():
    """Open the real viewer against generated snapshots without starting services."""
    app = QtWidgets.QApplication(sys.argv)
    viewer.dark_theme(app)
    sequence = count()
    viewer.fetch_snapshot = lambda _socket: make_snapshot(next(sequence))
    window = viewer.ViewerWindow("DEMO — no socket")
    window.setWindowTitle("Auspex V2 — Filter Graph Visual Fixture (not connected)")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
