"""Source/VNA example used by the draft V2 manual.

Start the services and explicitly register lab_source/SimulatedSource/SIM::SOURCE
and lab_vna/SimulatedVNA/SIM::VNA. Then submit to an existing absolute directory:
python examples/documented_vna_sweep.py /path/to/data

The simulators are independent: source frequency does not alter the synthetic VNA
response. Runtime NumPy estimates a sampled minimum, not a fitted resonance.
"""
import argparse

# --8<-- [start:imports]
from auspex_v2.experiment import (
    Experiment, ExperimentServerClient, FloatParameter, InstrumentRef, OutputConnector,
)
from auspex_v2.filters.definitions import PipelineDefinition, PipelineInput, WriteToFile
from auspex_v2.instruments.drivers.simulated import SimulatedSource, SimulatedVNA
# --8<-- [end:imports]


# --8<-- [start:declarations]
class SourceVnaSweep(Experiment):
    """Save a complete S21 trace at each source-frequency point."""

    source = InstrumentRef(driver=SimulatedSource)
    vna = InstrumentRef(driver=SimulatedVNA)
    source_frequency = FloatParameter(default=5e9, unit="Hz", bounds=(1e6, 20e9)).bind(
        source, "frequency", wait=0.01)
    minimum_magnitude = FloatParameter(bounds=(0., 1.))
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=65536)
# --8<-- [end:declarations]

# --8<-- [start:pipeline]
    def setup_pipeline(self):
        trace = PipelineInput("trace")
        writer = WriteToFile(name="disk", group="measurement", dataset="s21")
        PipelineDefinition(inputs=[trace], edges=[(trace, writer.sink)])
        self.connect(self.trace, trace)
# --8<-- [end:pipeline]

# --8<-- [start:setup]
    def setup(self, seq):
        self.source.output = False
        seq.set_parameter(self.source_frequency, self.source_frequency)
        self.source.output = True
        self.vna.frequency_center = 6.1e9
        self.vna.frequency_span = 20e6
        self.vna.num_points = 201
# --8<-- [end:setup]

# --8<-- [start:sequence]
    def sequence(self, seq):
        result = self.vna.get_trace(measurement="S21")
        frequency, response = result[0], result[1]
        numerical = seq.module("numpy")
        minimum = numerical.min(numerical.abs(response))
        seq.set_parameter(self.minimum_magnitude, minimum)
        seq.publish(self.trace, data=response, axes={"frequency": frequency})
# --8<-- [end:sequence]

# --8<-- [start:teardown]
    def teardown(self, seq):
        self.source.output = False
# --8<-- [end:teardown]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Existing absolute data directory on the server host")
    parser.add_argument("--socket", help="ExperimentServer socket")
    parser.add_argument("--source", default="lab_source")
    parser.add_argument("--vna", default="lab_vna")
    args = parser.parse_args()
# --8<-- [start:submission]
    experiment = SourceVnaSweep()
    experiment.bind_instrument("source", args.source)
    experiment.bind_instrument("vna", args.vna)
    experiment.add_sweep(experiment.source_frequency, [5e9, 5.1e9, 5.2e9])
    plan = experiment.compile()
    print(plan.explain())
    run = ExperimentServerClient(args.socket).submit(
        plan, name="source_vna", dir=args.directory, date=False)
    print(f"Submitted {run.run_id}\nFolder: {run.path}")
# --8<-- [end:submission]


if __name__ == "__main__":
    main()
