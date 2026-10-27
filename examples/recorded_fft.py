"""Record an FFT, select a symbolic bin, and publish a scalar parameter update.

Run: python examples/recorded_fft.py /path/to/new_fft.auspex
No instruments or services are needed. NumPy outside seq.module runs during
compilation; recorded NumPy runs synchronously in the offline interpreter.
The same saved plan can be submitted to ExperimentServer for writer-only execution.
"""
import argparse
from pathlib import Path

import numpy as np

from auspex_v2.experiment import Experiment, FloatParameter, IntParameter, OutputConnector, PipelineExecutor
from auspex_v2.filters.definitions import PipelineDefinition, PipelineInput, WriteToFile


class RecordedFFT(Experiment):
    """Measure the normalized real FFT coefficient of a deterministic cosine."""

    bin_index = IntParameter(default=3, bounds=(0, 31))
    component = FloatParameter(default=0., bounds=(-1., 1.))
    coefficient = OutputConnector(dtype="float64")

    def setup_pipeline(self):
        source = PipelineInput("coefficient")
        PipelineDefinition(inputs=[source], edges=[
            (source, WriteToFile(name="disk", dataset="fft_component").sink)])
        self.connect(self.coefficient, source)

    def sequence(self, seq):
        samples = np.cos(2 * np.pi * 3 * np.arange(32) / 32)  # Now: ordinary NumPy.
        recorded_numpy = seq.module("numpy")
        spectrum = recorded_numpy.fft.fft(samples)  # Later, even with concrete inputs.
        seq.set_parameter(self.component, spectrum[self.bin_index].real / len(samples))
        seq.publish(self.coefficient, data=self.component)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="New .auspex directory")
    args = parser.parse_args()
    plan = RecordedFFT().compile()  # Warns: NumPy has no author purity assertion.
    print(plan.explain())
    run = PipelineExecutor().run(plan, store_path=args.path)
    plan.save(Path(args.path) / "plan.json")
    if run.outcome != "SUCCEEDED":
        raise RuntimeError(run.reason)
    print(f"FFT bin 3 real coefficient: {run.params['component']:.6f} (expected 0.5)")


if __name__ == "__main__":
    main()
