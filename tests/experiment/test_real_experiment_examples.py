"""Compile reviewed lab examples without importing VISA or touching hardware."""
from pathlib import Path
import runpy
import unittest

from auspex_v2.instruments import InstrumentKind


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class RealExampleTest(unittest.TestCase):
    def test_vna_example_compiles_for_abstract_vna_role(self):
        cls = runpy.run_path(str(EXAMPLES / "real_vna_sweep.py"))["VnaSweep"]
        experiment = cls()
        experiment.bind_instrument("vna", "lab_vna")
        experiment.center.bind(experiment.vna, "frequency_center")
        experiment.add_sweep(experiment.center, [5.99e9, 6.01e9])
        plan = experiment.compile()
        self.assertEqual(plan.document["instruments"]["vna"]["kind"], InstrumentKind.VNA.value)

    def test_source_and_gs200_examples_compile(self):
        cases = [
            ("real_source_vna_sweep.py", "SourceVnaSweep", "source_frequency",
             {"source": "lab_source", "vna": "lab_vna"}, "source", [4.9e9, 5.0e9]),
            ("real_gs200_vna_sweep.py", "BiasVnaSweep", "bias",
             {"bias_source": "lab_bias", "vna": "lab_vna"}, "bias_source", [-0.01, 0.01]),
        ]
        for filename, class_name, parameter_name, bindings, role, points in cases:
            with self.subTest(filename=filename):
                cls = runpy.run_path(str(EXAMPLES / filename))[class_name]
                experiment = cls()
                for instrument_role, registered_name in bindings.items():
                    experiment.bind_instrument(instrument_role, registered_name)
                parameter = getattr(experiment, parameter_name)
                parameter.bind(getattr(experiment, role), "ramp" if parameter_name == "bias" else "frequency")
                experiment.add_sweep(parameter, points)
                plan = experiment.compile()
                self.assertEqual(plan.document["version"], 5)
                setup = plan.document["setup"][:4]
                self.assertEqual([node["member"] for node in setup],
                                 ["output", "mode", "level", "output"] if role == "bias_source"
                                 else ["output", "power", "frequency", "output"])
                self.assertIs(setup[0]["value"]["value"], False)
                self.assertIs(setup[3]["value"]["value"], True)
                if role == "bias_source":
                    self.assertEqual(setup[2]["value"]["value"], 0.0)
