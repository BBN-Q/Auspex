"""Submit an attended GS200 bias sweep with a VNA trace per point.

The example uses voltage mode and small demonstration values. Review ranges,
protection, wiring, ramp rate and output policy for the actual device under test.
Setup intentionally interrupts any existing bias: output off, voltage mode,
zero volts, then output on. Configure the VNA for a linear frequency sweep.
"""
import argparse

from auspex_v2.experiment import Experiment, ExperimentServerClient, FloatParameter, InstrumentRef, OutputConnector
from auspex_v2.filters.definitions import PipelineDefinition, PipelineInput, WriteToFile
from auspex_v2.instruments import AgilentE8363C, YokogawaGS200


class BiasVnaSweep(Experiment):
    bias_source = InstrumentRef(driver=YokogawaGS200)
    vna = InstrumentRef(driver=AgilentE8363C)
    bias = FloatParameter(default=0.0, unit="V", bounds=(-0.1, 0.1))
    trace = OutputConnector(dtype="complex64", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"}, max_bytes=8 * 1024 * 1024)

    def setup_pipeline(self):
        source = PipelineInput("trace")
        PipelineDefinition(inputs=[source], edges=[
            (source, WriteToFile(name="disk", group="measurement", dataset="s21").sink)])
        self.connect(self.trace, source)

    def setup(self, seq):
        self.bias_source.output = False
        self.bias_source.mode = "voltage"
        self.bias_source.level = 0.0
        self.bias_source.output = True
        self.vna.set_measurements(["S21"])
        self.vna.num_points = 401
        self.vna.output_enable = True

    def sequence(self, seq):
        seq.wait(0.05)
        result = self.vna.get_trace(measurement="S21")
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        self.bias_source.ramp = 0.0
        self.bias_source.output = False
        self.vna.output_enable = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--bias-source", default="bias_source")
    parser.add_argument("--vna", default="vna")
    parser.add_argument("--socket")
    args = parser.parse_args()
    experiment = BiasVnaSweep()
    experiment.bind_instrument("bias_source", args.bias_source)
    experiment.bind_instrument("vna", args.vna)
    experiment.bias.bind(experiment.bias_source, "ramp")
    experiment.add_sweep(experiment.bias, [-0.01, -0.005, 0.0, 0.005, 0.01])
    run = ExperimentServerClient(args.socket).submit(
        experiment.compile(), name="bias_vna_sweep", dir=args.directory, date=True)
    print(run.run_id, run.path)


if __name__ == "__main__":
    main()
