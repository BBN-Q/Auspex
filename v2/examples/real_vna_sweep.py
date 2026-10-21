"""Submit an attended fixed sweep using either ported Agilent VNA.

Review resource names, measurement routing, RF levels and output policy in the lab.
The named VNA must already be registered as AgilentE8363C or AgilentN5230A.
Configure a linear frequency sweep: the returned axis interpolates read-back endpoints.
"""
import argparse

from auspex_v2.experiment import Experiment, ExperimentServerClient, FloatParameter, InstrumentRef, OutputConnector
from auspex_v2.filters.definitions import PipelineDefinition, PipelineInput, WriteToFile
from auspex_v2.instruments import AgilentE8363C


class VnaSweep(Experiment):
    vna = InstrumentRef().bind(AgilentE8363C)  # Any registered VNA KIND may fill this role.
    center = FloatParameter(default=6e9, unit="Hz")
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=8 * 1024 * 1024)

    def setup_pipeline(self):
        source = PipelineInput("trace")
        PipelineDefinition(inputs=[source], edges=[
            (source, WriteToFile(name="disk", group="measurement", dataset="s21").sink)])
        self.connect(self.trace, source)

    def setup(self, seq):
        self.vna.set_measurements(["S21"])
        self.vna.num_points = 401
        self.vna.frequency_span = 20e6
        self.vna.output_enable = True

    def sequence(self, seq):
        result = self.vna.get_trace(measurement="S21")
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        self.vna.output_enable = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--vna", default="vna", help="registered VNA name")
    parser.add_argument("--socket")
    args = parser.parse_args()
    experiment = VnaSweep()
    experiment.bind_instrument("vna", args.vna)
    experiment.center.bind(experiment.vna, "frequency_center")
    experiment.add_sweep(experiment.center, [5.98e9, 5.99e9, 6.00e9, 6.01e9, 6.02e9])
    run = ExperimentServerClient(args.socket).submit(
        experiment.compile(), name="vna_sweep", dir=args.directory, date=True)
    print(run.run_id, run.path)


if __name__ == "__main__":
    main()
