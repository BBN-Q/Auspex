"""Compile explicit publications and run against synthetic command results.

Usage: python examples/compiled_publication.py /path/to/new_run.auspex
This is an offline integration harness: waits are virtual and no hardware/server
is contacted. The compiled plan and pipeline are included in run metadata.
"""
import argparse
from pathlib import Path
import numpy as np

from auspex_v2.experiment import Experiment, OutputConnector, InstrumentRef, FloatParameter
from auspex_v2.experiment.pipeline import PipelineExecutor
from auspex_v2.filters.definitions import PipelineInput, PipelineDefinition, Passthrough, WriteToFile
from auspex_v2.instruments.drivers.simulated import SimulatedVNA


class VnaSweep(Experiment):
    """Publish a complete complex trace at each requested bias."""
    vna = InstrumentRef(kind="VNA", driver=SimulatedVNA)
    source = InstrumentRef(kind="MICROWAVE_SOURCE")
    bias = FloatParameter(default=0., unit="V")
    trace = OutputConnector(dtype="complex128", dimensions=("frequency",),
                            axis_units={"frequency": "Hz"})

    def setup_pipeline(self):
        trace = PipelineInput("trace")
        forward = Passthrough(name="forward")
        writer = WriteToFile(name="disk", group="measurement", dataset="response")
        PipelineDefinition(inputs=[trace], edges=[
            (trace, forward.sink), (forward.source, writer.sink)])
        self.connect(self.trace, trace)

    def sequence(self, seq):
        result = self.vna.get_trace()
        # SimulatedVNA returns a (frequency, response) tuple. Select explicitly;
        # symbolic results cannot be unpacked while recording Python hooks.
        seq.publish(self.trace, data=result[1], axes={"frequency": result[0]})

    def teardown(self, seq):
        seq.set(self.source, "voltage", 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="New .auspex directory")
    args = parser.parse_args()
    experiment = VnaSweep()
    experiment.bind_instrument("vna", "lab_vna")
    experiment.bind_instrument("source", "lab_source")
    experiment.bias.bind(experiment.source, "voltage", wait=0.01)
    experiment.add_sweep(experiment.bias, [0., 0.1, 0.2])
    plan = experiment.compile()
    bias = 0.

    def command(op, instrument, member, arguments, keywords):
        nonlocal bias
        if op == "SET" and member == "voltage":
            bias = arguments[0]
        if member == "get_trace":
            frequency = np.linspace(4e9, 6e9, 101)
            response = 1 / (1 + 1j * (frequency - (5e9 + bias * 1e9)) / 1e8)
            return frequency, response

    run = PipelineExecutor(command).run(plan, store_path=args.path,
                                       run_metadata={"experiment_plan": plan.document,
                                                     "plan_identity": plan.identity})
    plan.save(Path(args.path) / "plan.json")
    print(run.outcome, run.reason)
    print("Dataset: measurement/response; saved plan: plan.json")


if __name__ == "__main__":
    main()
