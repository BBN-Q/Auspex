"""Compile an explicit publication pipeline and submit it to ExperimentServer.

Start both services, then explicitly register lab_source / SimulatedSource /
SIM::SOURCE and lab_vna / SimulatedVNA / SIM::VNA before submitting. No injected
command callbacks or test executors are part of this example.
"""
import argparse

from auspex_v2.experiment import (
    Experiment, ExperimentServerClient, OutputConnector, InstrumentRef, FloatParameter,
)
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, Passthrough, WriteToFile
from auspex_v2.instruments.drivers.simulated import SimulatedSource, SimulatedVNA


class VnaSweep(Experiment):
    """Publish one complete complex trace at each requested source frequency."""

    vna = InstrumentRef(driver=SimulatedVNA)
    source = InstrumentRef(driver=SimulatedSource)
    source_frequency = FloatParameter(default=5e9, unit="Hz").bind(source, "frequency", wait=0.01)
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"})

    def setup_pipeline(self):
        trace = PipelineInput("trace")
        forward = Passthrough(name="forward")
        writer = WriteToFile(name="disk", group="measurement", dataset="response")
        PipelineDefinition(inputs=[trace], edges=[
            (trace, forward.sink), (forward.source, writer.sink)])
        self.connect(self.trace, trace)

    def setup(self, seq):
        self.source.output = False
        seq.set_parameter(self.source_frequency, self.source_frequency)
        self.source.output = True
        self.vna.num_points = 101

    def sequence(self, seq):
        result = self.vna.get_trace()
        # Select tuple fields explicitly; symbolic results cannot be unpacked.
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        self.source.output = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Existing absolute data directory on the server host")
    parser.add_argument("--socket")
    args = parser.parse_args()
    experiment = VnaSweep()
    experiment.bind_instrument("vna", "lab_vna")
    experiment.bind_instrument("source", "lab_source")
    experiment.add_sweep(experiment.source_frequency, [5e9, 5.1e9, 5.2e9])
    plan = experiment.compile()
    run = ExperimentServerClient(args.socket).submit(
        plan, name="publication", dir=args.directory, date=False)
    print(f"Submitted {run.run_id}\nFolder: {run.path}\nDataset: measurement/response")


if __name__ == "__main__":
    main()
