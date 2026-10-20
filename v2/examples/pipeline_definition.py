"""Synthetic whole-window acquisition through a reusable, inert pipeline.

Run: python v2/examples/pipeline_definition.py /path/to/new_run.auspex
No instrument server or ExperimentServer is needed. The destination must not exist.
"""
import argparse
import numpy as np

from auspex_v2.data import Axis
from auspex_v2.filters import ManualPipeline, PortSchema
from auspex_v2.filters.definitions import (
    PipelineInput, PipelineDefinition, Passthrough, WriteToFile, DataBuffer,
)


def trace_pipeline():
    trace = PipelineInput("trace")
    forward = Passthrough(name="forward")
    disk = WriteToFile(name="disk", group="measurement", dataset="response")
    memory = DataBuffer(name="memory", capacity=16)
    return PipelineDefinition(inputs=[trace], edges=[
        (trace, forward.sink),
        (forward.source, disk.sink),
        (forward.source, memory.sink),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="New .auspex run directory")
    args = parser.parse_args()
    definition = trace_pipeline()
    schema = PortSchema("complex128", 1, ("frequency",), ("Hz",), max_bytes=65536,
                        value_name="response")
    compiled = definition.compile(inputs={"trace": schema})
    frequency = np.linspace(4e9, 6e9, 101)
    with ManualPipeline(compiled, store_path=args.path,
                        run_metadata={"pipeline": compiled.document}) as running:
        for bias in (0., 0.1, 0.2):
            response = 1 / (1 + 1j * (frequency - (5e9 + bias * 1e9)) / 1e8)
            running.submit(response, axes=[Axis("frequency", frequency, "Hz")],
                           sweep={"bias": bias}, sweep_units={"bias": "V"})
    print(f"Published measurement/response in {args.path}")
    print(f"Retained {len(running.buffers['memory'].drain())} complete windows")


if __name__ == "__main__":
    main()
