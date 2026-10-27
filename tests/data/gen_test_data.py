"""Generate deterministic, synthetic V2 .auspex folders for manual DataBrowser demos.

Run from the repository root: python tests/data/gen_test_data.py
This is not a unittest. The generated .auspex folders are intentionally ignored
by the repository's data-file gitignore rules; rerun the script to recreate them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from auspex_v2.data import Axis, AuspexStore

DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "test_data"
FIXTURE_ID = "auspex-v2-data-browser-demo-1"


def lorentzian(x, center, width):
    """Return a unit-height Lorentzian with width specified at half maximum."""
    return width**2 / ((x - center)**2 + width**2)


def resonances(path):
    """Create nonuniform-grid VNA magnitude and complex transmission traces."""
    rng = np.random.default_rng(12017)
    normalized_frequency = np.linspace(4.63, 5.37, 1801)
    # Preserve monotonicity while avoiding an artificially uniform readback grid.
    frequency = (normalized_frequency + 1.1e-5 * np.sin(np.linspace(0, 20 * np.pi, 1801))) * 1e9
    baseline = 0.97 + 0.022 * (normalized_frequency - 5.0) + 0.008 * np.sin(5 * normalized_frequency)
    transmission = (baseline - 0.43 * lorentzian(normalized_frequency, 4.771, 0.006)
                    - 0.27 * lorentzian(normalized_frequency, 5.026, 0.018)
                    - 0.52 * lorentzian(normalized_frequency, 5.254, 0.0045)
                    + rng.normal(0, 0.006, frequency.size))
    # The complex trace has dispersive phase as well as absorption; it exercises
    # the browser's real/imaginary/magnitude/phase display selector.
    complex_trace = (0.9 - 0.26 / (1 + 1j * (normalized_frequency - 4.771) / 0.006)
                     - 0.17 / (1 + 1j * (normalized_frequency - 5.026) / 0.018)
                     - 0.31 / (1 + 1j * (normalized_frequency - 5.254) / 0.0045))
    complex_trace *= np.exp(1j * (normalized_frequency - 5) * 1.4)
    complex_trace += (rng.normal(0, 0.002, frequency.size)
                      + 1j * rng.normal(0, 0.002, frequency.size))
    with AuspexStore(path, mode="w") as store:
        store.write_run_metadata({
            "fixture_id": FIXTURE_ID, "synthetic": True,
            "experiment": "Three simulated microwave resonances",
            "settings": {"source_power_dBm": -38, "vna_points": len(frequency),
                         "if_bandwidth_Hz": 1000},
            "notes": "Deterministic simulation, NOT measurements from a physical VNA.",
        })
        axis = [Axis("frequency", frequency, "Hz")]
        store.write_dataset("traces", "three_lorentzians", transmission.astype(np.float64), axis,
                            value_name="S21 magnitude", value_unit="dimensionless",
                            label="three dips plus noise", input_name="SimulatedVNA",
                            source="reference_generator.lorentzians")
        store.write_dataset("traces", "complex_s21", complex_trace.astype(np.complex128), axis,
                            value_name="S21", value_unit="dimensionless",
                            label="dispersive resonances", input_name="SimulatedVNA",
                            source="reference_generator.complex_response")


def ramsey(path):
    """Create a decaying, beating Ramsey-like trace with a late revival."""
    rng = np.random.default_rng(20240419)
    time_us = np.linspace(0, 48, 1501)
    time_us += 7e-4 * np.sin(np.linspace(0, 19 * np.pi, time_us.size))
    envelope = np.exp(-(time_us / 27)**1.35)
    chirp = np.cos(2 * np.pi * (0.23 * time_us + 0.0008 * time_us**2))
    beating = np.cos(2 * np.pi * 0.027 * time_us)
    revival = 0.085 * np.exp(-((time_us - 35) / 3.6)**2) * np.cos(2 * np.pi * 0.18 * (time_us - 35))
    excited = np.clip(0.5 + 0.36 * envelope * chirp * beating + revival
                      + rng.normal(0, 0.006, time_us.size), 0, 1)
    with AuspexStore(path, mode="w") as store:
        store.write_run_metadata({
            "fixture_id": FIXTURE_ID, "synthetic": True,
            "experiment": "Simulated detuned Ramsey interferometry",
            "settings": {"nominal_detuning_MHz": 0.23, "readout_shots_per_point": 4096},
            "notes": "Deterministic synthetic beating, decay and revival; not a fit or measured data.",
        })
        store.write_dataset("qubit", "ramsey_beating", excited.astype(np.float64),
                            [Axis("delay", time_us, "us")], value_name="excited-state probability",
                            value_unit="dimensionless", label="beating with a late revival",
                            input_name="simulated_readout", source="reference_generator.ramsey")


def complex_map(path):
    """Create one nonuniform-grid complex resonance map for visual 2D checks."""
    frequency = 5 + 0.12 * np.linspace(-1, 1, 181)**3
    bias = 0.8 * np.linspace(-1, 1, 91)**3
    detuning = (frequency[None, :] - (5 + 0.07 * bias[:, None])) / 0.008
    response = (1 - 0.8 / (1 + 1j * detuning)) * np.exp(1j * 9 * bias[:, None])
    with AuspexStore(path, mode="w") as store:
        store.write_run_metadata({
            "fixture_id": FIXTURE_ID, "synthetic": True,
            "experiment": "Complex 2D resonance map",
            "notes": "Nonuniform grid; diagonal resonance and wrapped phase. Not measured data.",
        })
        store.write_dataset("maps", "complex_resonance", response,
                            [Axis("bias", bias, "V"), Axis("frequency", frequency, "GHz")],
                            value_name="S21", value_unit="dimensionless")


def generate_test_data(directory=DEFAULT_DIRECTORY):
    """Create missing demo folders; refuse to overwrite unrelated existing runs."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    folders = ((directory / "resonances.auspex", resonances, {"traces/three_lorentzians", "traces/complex_s21"}),
               (directory / "ramsey.auspex", ramsey, {"qubit/ramsey_beating"}),
               (directory / "complex_map.auspex", complex_map, {"maps/complex_resonance"}))
    for path, generate, expected in folders:
        if path.exists():
            # Never overwrite a run folder, including one somebody edited later.
            with AuspexStore(path) as store:
                try:
                    metadata = store.read_run_metadata()
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Existing {path} is not the expected demo; move it before regenerating") from exc
                if (not isinstance(metadata, dict) or metadata.get("fixture_id") != FIXTURE_ID or
                        {f"{group}/{name}" for group, name in store.list_datasets()} != expected):
                    raise ValueError(f"Existing {path} is not the expected demo; move it before regenerating")
        else:
            generate(path)
    return tuple(path for path, _, _ in folders)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate deterministic Auspex V2 DataBrowser demo runs")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY,
                        help="Output directory (default: tests/data/test_data)")
    args = parser.parse_args(argv)
    for path in generate_test_data(args.directory):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
