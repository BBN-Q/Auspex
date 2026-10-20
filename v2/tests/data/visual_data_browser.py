"""Non-unittest, desktop visual demo of DataBrowser with generated V2 runs.

From the repository root, with the optional V2 GUI dependencies installed:
    python v2/tests/data/visual_data_browser.py

Generates the ignored fixture files in test/test_data/ if absent, then opens three
independently closable 1D tabs across two .auspex folders. No lab hardware or
services are accessed. Run test/gen_test_data.py explicitly to generate only data.
"""

from __future__ import annotations

from pathlib import Path
import sys

from PySide6 import QtWidgets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "v2" / "utils"))

from gen_test_data import generate_test_data
from data_browser import DataBrowser, dark_theme, ROLE_KEY


def make_demo_window():
    """Open generated run folders and select their noisy Lorentzian plot."""
    folders = generate_test_data()
    window = DataBrowser()
    window.add_directory(folders[0].parent)
    window.setWindowTitle("Auspex V2 — DataBrowser visual demo (synthetic data)")
    first = None
    os_folder = window.tree.topLevelItem(0)
    for run_index in range(os_folder.childCount()):
        run = os_folder.child(run_index)
        for group_index in range(run.childCount()):
            group = run.child(group_index)
            for dataset_index in range(group.childCount()):
                item = group.child(dataset_index)
                key = item.data(0, ROLE_KEY)
                if key[0] == "dataset" and window._entry(key).plottable:
                    window._open_item(item, 0)
                    if key[3] == "three_lorentzians":
                        first = (item, (Path(key[1]), key[2], key[3]))
    if first is not None:
        window.tabs.setCurrentWidget(window.open_tabs[first[1]])
        window.tree.setCurrentItem(first[0])
    return window


def main():
    """Launch the real DataBrowser with two synthetic, independently browsable runs."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dark_theme(app)
    window = make_demo_window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
