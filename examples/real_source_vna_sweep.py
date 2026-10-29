"""Submit an attended N5183A frequency sweep with a VNA trace per point.

Review source power, cabling, VNA configuration and output policy before use.
Setup intentionally interrupts existing RF output, configures power and frequency,
then enables output. Configure the VNA for a linear frequency sweep.
"""
import argparse

from auspex_v2.experiment import Experiment, ExperimentServerClient, FloatParameter, InstrumentRef, OutputConnector
from auspex_v2.filters.definitions import PipelineDefinition, PipelineInput, WriteToFile
from auspex_v2.instruments import AgilentE8363C, AgilentN5183A


class SourceVnaSweep(Experiment):
    source = InstrumentRef(driver=AgilentN5183A)
    vna = InstrumentRef(driver=AgilentE8363C)
    source_frequency = FloatParameter(default=5e9, unit="Hz").bind(source, "frequency")
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=8 * 1024 * 1024)

    def setup_pipeline(self):
        source = PipelineInput("trace")
        PipelineDefinition(inputs=[source], edges=[
            (source, WriteToFile(name="disk", group="measurement", dataset="s21").sink)])
        self.connect(self.trace, source)

    def setup(self, seq):
        self.source.output = False
        self.source.power = -40.0
        self.source.frequency = self.source_frequency
        self.source.output = True
        self.vna.set_measurements(["S21"])
        self.vna.num_points = 401
        self.vna.output_enable = True

    def sequence(self, seq):
        seq.wait(0.05)
        result = self.vna.get_trace(measurement="S21")
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        self.source.output = False
        self.vna.output_enable = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--source", default="source")
    parser.add_argument("--vna", default="vna")
    parser.add_argument("--socket")
    args = parser.parse_args()
    experiment = SourceVnaSweep()
    experiment.bind_instrument("source", args.source)
    experiment.bind_instrument("vna", args.vna)
    experiment.add_sweep(experiment.source_frequency, [4.98e9, 4.99e9, 5.00e9, 5.01e9, 5.02e9])
    run = ExperimentServerClient(args.socket).submit(
        experiment.compile(), name="source_vna_sweep", dir=args.directory, date=True)
    print(run)


if __name__ == "__main__":
    main()
