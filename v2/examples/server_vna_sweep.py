"""Submit a saved fixed sweep and exit; execution belongs to ExperimentServer.

Start instrument_server.py and experiment_server.py separately. Register
lab_source / SimulatedSource / SIM::SOURCE and lab_vna / SimulatedVNA / SIM::VNA
explicitly through InstrumentServerClient before submitting (no implicit registration).
"""
import argparse

from auspex_v2.experiment import (Experiment, OutputConnector, InstrumentRef,
                                  FloatParameter, ExperimentServerClient)
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, WriteToFile
from auspex_v2.instruments.drivers.simulated import SimulatedSource, SimulatedVNA


class Sweep(Experiment):
    source = InstrumentRef(kind="MICROWAVE_SOURCE", driver=SimulatedSource)
    vna = InstrumentRef(kind="VNA", driver=SimulatedVNA)
    source_frequency = FloatParameter(default=5e9, unit="Hz")
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=65536)

    def setup_pipeline(self):
        source = PipelineInput("trace")
        PipelineDefinition(inputs=[source], edges=[
            (source, WriteToFile(name="disk", dataset="response").sink)])
        self.connect(self.trace, source)

    def setup(self, seq):
        self.source.output = True
        self.vna.num_points = 201

    def sequence(self, seq):
        seq.wait(0.2)
        result = self.vna.get_trace()
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        self.source.output = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Absolute existing data directory on the server host")
    parser.add_argument("--socket")
    args = parser.parse_args()
    experiment = Sweep()
    experiment.bind_instrument("source", "lab_source")
    experiment.bind_instrument("vna", "lab_vna")
    experiment.source_frequency.bind(experiment.source, "frequency")
    experiment.add_sweep(experiment.source_frequency, [5e9, 5.1e9, 5.2e9, 5.3e9])
    plan = experiment.compile()
    run = ExperimentServerClient(args.socket).submit(plan, name="vna", dir=args.directory, date=True)
    print(f"Submitted {run.run_id}\nFolder: {run.path}\nThis client now exits; reconnect by run ID.")


if __name__ == "__main__":
    main()
