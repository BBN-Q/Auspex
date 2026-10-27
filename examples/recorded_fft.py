"""Record an FFT, select a symbolic bin, and publish a scalar parameter update.

Start the Auspex services, then submit to an existing server-host data directory:
python examples/recorded_fft.py /path/to/data
No instruments are used. NumPy outside seq.module runs during compilation;
recorded NumPy executes synchronously in the ExperimentServer runner.
"""
import argparse
import numpy as np

from auspex_v2.experiment import Experiment, ExperimentServerClient, FloatParameter, IntParameter, OutputConnector
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
    parser.add_argument("directory", help="Existing absolute data directory on the server host")
    parser.add_argument("--socket")
    args = parser.parse_args()
    plan = RecordedFFT().compile()  # Warns: NumPy has no author purity assertion.
    print(plan.explain())
    run = ExperimentServerClient(args.socket).submit(
        plan, name="fft", dir=args.directory, date=False)
    print(f"Submitted {run.run_id}\nFolder: {run.path}")
    print("Dataset data/fft_component should contain 0.5 after successful execution.")


if __name__ == "__main__":
    main()
