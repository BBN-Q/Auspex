"""Small offline checks for generated driver documentation; no device harness."""
import unittest
from unittest.mock import patch

from auspex_v2.instruments.drivers.core import Instrument
from auspex_v2.instruments.drivers.registry import DRIVERS
from utils.generate_instrument_api import generated_pages, generate


class DriverReferenceTest(unittest.TestCase):
    """Check effective inherited APIs and keep committed pages in sync with source."""

    def test_effective_api_and_command_declarations(self):
        """Real VNAs share operations but retain distinct ports; overrides stay local."""
        with patch.object(Instrument, "__init__", side_effect=AssertionError("No device construction")):
            pages = generated_pages()
        self.assertEqual(set(pages), {"index.md", *(name + ".md" for name in DRIVERS)})
        vna = pages["AgilentE8363C.md"]
        self.assertIn("get_trace(measurement=None, restart_measurement=True, timeout=None)", vna)
        self.assertIn("agilent._AgilentNetworkAnalyzer", vna)
        self.assertIn("logarithmic and segmented stimulus readback is not implemented", vna)
        self.assertIn("**ports:** `(1, 2)`", vna)
        self.assertIn("**ports:** `(1, 2, 3, 4)`", pages["AgilentN5230A.md"])
        self.assertIn("FloatCommand(", vna)
        self.assertIn("value_map={False: '0', True: '1'}", vna)
        self.assertIn("### measurements\n\n**Access:** GET\n", vna)
        self.assertNotIn("### connect", vna)
        self.assertNotIn("### get_frequency_start", vna)
        self.assertIn("**Equivalent names:** `value`", pages["YokogawaGS200.md"])
        ramp = pages["BOP2020M.md"]
        for text in ("RampCommand(", "value_range=(-20, 20)", "increment=0.1", "pause=0.02"):
            self.assertIn(text, ramp)
        self.assertIn("additional_args=('channel',)", pages["SimulatedSource.md"])
        pro = pages["DSInstrumentsSG12000Pro.md"]
        reference = pro.split("### reference\n", 1)[1].split("### ", 1)[0]
        self.assertIn("dsinstruments.DSInstrumentsSG12000Pro", reference)
        self.assertIn("### set_display", pro)
        self.assertNotIn("### set_low_power_mode", pro)  # Only the non-Pro model declares it.
        self.assertIn("**Experimental:** yes", pro)
        self.assertNotIn("HP33120A.md", pages)

    def test_generated_reference_is_current(self):
        self.assertTrue(generate(check=True), "Run python utils/generate_instrument_api.py")
