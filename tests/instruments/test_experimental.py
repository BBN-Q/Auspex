"""Experimental designation is metadata, not an alternate driver execution path."""

import unittest
import warnings

from auspex_v2.instruments import (
    AgilentE8363C, AgilentN5183A, AgilentN5230A, SimulatedSource,
    YokogawaGS200, SMB100B, experimental,
)
from auspex_v2.instruments.drivers import experimental as driver_experimental


class ExperimentalTest(unittest.TestCase):
    """Use the actual simulator lifecycle; no VISA hardware or mock transports."""

    def test_marker_preserves_class_and_warns_only_on_connection(self):
        """Keep declaration inspection silent and repeat normal use after reconnect."""
        class UnvalidatedSource(SimulatedSource):
            """A test-only experimental simulator, not a hardware-driver test."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(experimental, driver_experimental)
            self.assertIs(experimental(UnvalidatedSource), UnvalidatedSource)
            self.assertIs(UnvalidatedSource.properties["frequency"],
                          SimulatedSource.properties["frequency"])
            source = UnvalidatedSource("SIM::EXPERIMENTAL")
            self.assertTrue(UnvalidatedSource.EXPERIMENTAL)
            self.assertEqual(caught, [])
            for _ in range(2):
                with source:
                    source.frequency = 5e9
                    self.assertEqual(source.frequency, 5e9)
            self.assertEqual(len(caught), 2)
            self.assertTrue(all(item.category is UserWarning for item in caught))
            self.assertIn("UnvalidatedSource driver has not yet been validated with hardware",
                          str(caught[0].message))

    def test_subclasses_inherit_marker_without_changing_existing_drivers(self):
        @experimental
        class Parent(SimulatedSource):
            """Test-only unvalidated base."""

        class Child(Parent):
            """Validation cannot be inferred merely from subclassing."""

        self.assertTrue(Child.EXPERIMENTAL)
        for cls in (SimulatedSource, AgilentE8363C, AgilentN5183A,
                    AgilentN5230A, YokogawaGS200):
            self.assertFalse(cls.EXPERIMENTAL)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with SimulatedSource("SIM::VALIDATED"):
                pass
        self.assertEqual(caught, [])

    def test_ported_drivers_are_exported_and_admitted_without_hardware(self):
        """Inspect declarations only; do not exercise any hardware-driver methods."""
        from auspex_v2 import instruments
        from auspex_v2.instruments import drivers
        from auspex_v2.instruments.drivers.registry import DRIVERS

        names = ("SMB100B", "AMI430", "BNC845", "BOP2020M", "Picosecond10070A",
                 "DSInstrumentsSG12000", "DSInstrumentsSG12000Pro", "DAT64H",
                 "RelayDriver", "DigitalAttenuator", "Keithley2400")
        for name in names:
            with self.subTest(driver=name):
                canonical = DRIVERS[name]
                self.assertIs(getattr(instruments, name), canonical)
                self.assertIs(getattr(drivers, name), canonical)
                self.assertTrue(canonical.EXPERIMENTAL)
                self.assertEqual(canonical.__name__, name)
        self.assertIs(SMB100B, DRIVERS["SMB100B"])
        self.assertNotIn("DSInstruments", DRIVERS)

    def test_marker_requires_an_instrument_class(self):
        for value in (object, None, 42, SimulatedSource("SIM::INSTANCE")):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "Instrument subclass"):
                experimental(value)
