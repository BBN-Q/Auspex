"""Standalone storage and manual whole-window processing, with synthetic data.

Run: python v2/examples/manual_data.py OUTPUT_DIRECTORY
Creates two new .auspex folders; never opens an instrument or starts a service.
"""
import argparse
from pathlib import Path

import numpy as np

from auspex_v2.data import Axis, AuspexStore
from auspex_v2.filters import BufferSpec, ManualPipeline, NodeSpec, PortSchema, WriterSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    frequency = np.linspace(6e9, 6.2e9, 101)
    response = (1 - 0.7 / (1 + 1j * (frequency - 6.1e9) / 1e6)).astype(np.complex128)
    axes = [Axis("frequency", frequency, "Hz")]

    # Complete arrays from any acquisition library can be saved directly.
    with AuspexStore(args.directory / "standalone.auspex", mode="w") as store:
        store.write_run_metadata({"source": "manual_data.py", "synthetic": True})
        store.write_dataset("data", "trace", response, axes)

    schema = PortSchema("complex128", 1, ("frequency",), ("Hz",), 8192,
                        value_name="S21")
    with ManualPipeline(
        schema,
        {"average": NodeSpec("complex_average", schema, {"repeats": 2})},
        [("$input", "out", "average")],
        writers={"disk": WriterSpec("average", "final")},
        buffers={"memory": BufferSpec("average", "final", capacity=8)},
        store_path=args.directory / "processed.auspex",
        run_metadata={"synthetic": True},
    ) as pipeline:
        for bias in (0., 0.1, 0.2):
            for repeat in range(2):
                # Replace only this acquisition step with your own Python code.
                measured = response * (1 + bias)
                pipeline.submit(measured, axes=axes, sweep={"bias": bias}, sweep_units={"bias": "V"})

    # Copies remain accessible after processing workers and shared memory close.
    buffered = pipeline.buffers["memory"]
    print(f"{len(buffered.view())} completed averages retained")
    for result in buffered.drain():
        print(result.scan, result.window, result.sweep, result.data.shape)


if __name__ == "__main__":
    main()
