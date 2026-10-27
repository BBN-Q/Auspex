"""Unavailable driver declarations must not masquerade as usable instruments."""

import importlib
import unittest

from auspex_v2 import instruments
from auspex_v2.instruments import drivers
from auspex_v2.instruments.drivers.registry import DRIVERS


STUBS = {
    "agilent": ("HP33120A", "Agilent33220A", "Agilent33500B", "Agilent34970A", "AgilentN9010A"),
    "bbn": ("SpectrumAnalyzer",),
    "keysight": ("KeysightM8190A",),
    "lakeshore": ("LakeShore370", "LakeShore335"),
    "lecroy": ("HDO6104",),
    "stanford": ("SR830", "SR865"),
    "tektronix": ("DPO72004C", "DPO2024", "DPO2014", "RSA3308A"),
}


class DriverStubsTest(unittest.TestCase):
    """Check stub boundaries only, never hardware commands or mock transports."""

    def test_all_sixteen_stubs_are_module_local_and_not_admitted(self):
        """Direct module access is for future manual work, not service availability."""
        self.assertEqual(sum(map(len, STUBS.values())), 16)
        for vendor, names in STUBS.items():
            module = importlib.import_module(f"auspex_v2.instruments.drivers.{vendor}")
            for name in names:
                with self.subTest(driver=name):
                    self.assertEqual(getattr(module, name).__name__, name)
                    self.assertNotIn(name, module.__all__)
                    self.assertNotIn(name, DRIVERS)
                    for api in (instruments, drivers):
                        self.assertNotIn(name, api.__all__)
                        self.assertFalse(hasattr(api, name))

    def test_stub_construction_always_reports_manual_work_required(self):
        """Even normal resource/name arguments must fail before any transport exists."""
        for vendor, names in STUBS.items():
            module = importlib.import_module(f"auspex_v2.instruments.drivers.{vendor}")
            for name in names:
                cls = getattr(module, name)
                with self.subTest(driver=name):
                    for args, kwargs in (((), {}), (("UNIMPLEMENTED",), {"name": "pending"})):
                        with self.assertRaisesRegex(NotImplementedError, name + " requires manual V2 porting"):
                            cls(*args, **kwargs)
