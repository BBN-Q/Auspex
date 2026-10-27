"""One module export list controls Python completion and server driver admission."""

from contextlib import contextmanager
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from auspex_v2 import instruments
from auspex_v2.instruments import drivers
from auspex_v2.instruments.drivers.registry import DRIVERS, _discover_drivers
from auspex_v2.instruments.server import InstrumentServerClient, RemoteError

ROOT = Path(__file__).resolve().parents[2]
IMPORTS = '''from auspex_v2.instruments.drivers.core import Instrument, SCPIInstrument
from auspex_v2.instruments.kinds import InstrumentKind
'''
DECLARATION = '''class ExportedDriver(Instrument):
    """Only declaration inspection is permitted during discovery."""
    KIND = InstrumentKind.SENSOR
    def __init__(self, *args, **kwargs):
        raise AssertionError("Discovery constructed a driver")
'''


@contextmanager
def driver_package(modules):
    """Create real importable modules, with an isolated package namespace per check."""
    name = "_auspex_export_test"
    with tempfile.TemporaryDirectory(dir=ROOT / ".agents") as directory:
        package_path = Path(directory) / name
        package_path.mkdir()
        (package_path / "__init__.py").write_text("")
        for module, body in modules.items():
            (package_path / f"{module}.py").write_text(IMPORTS + body)
        sys.path.insert(0, directory)
        try:
            importlib.invalidate_caches()
            yield importlib.import_module(name)
        finally:
            sys.path.remove(directory)
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]


class DriverExportsTest(unittest.TestCase):
    """Check exports without opening hardware or depending on a VISA installation."""

    def test_current_exports_are_identical_at_every_import_path(self):
        for name, driver in DRIVERS.items():
            with self.subTest(driver=name):
                self.assertIs(getattr(drivers, name), driver)
                self.assertIs(getattr(instruments, name), driver)
                self.assertIn(name, drivers.__all__)
                self.assertIn(name, instruments.__all__)
                self.assertIn(name, dir(instruments))
        self.assertNotIn("_AgilentNetworkAnalyzer", instruments.__all__)
        self.assertNotIn("Instrument", DRIVERS)
        self.assertNotIn("InstrumentKind", DRIVERS)

    def test_discovery_is_explicit_deterministic_and_does_not_construct_drivers(self):
        """Adding a module export suffices; unexported helpers/subclasses remain private."""
        with driver_package({
            "z_vendor": DECLARATION + '__all__ = ["ExportedDriver"]\n',
            "a_vendor": DECLARATION.replace("ExportedDriver", "FirstDriver") + '__all__ = ["FirstDriver"]\n',
            "unexported": DECLARATION,
            "_private": 'raise AssertionError("Private module was imported")\n',
        }) as package:
            found = _discover_drivers(package)
            self.assertEqual(list(found), ["FirstDriver", "ExportedDriver"])
            self.assertEqual(found["ExportedDriver"].__module__, package.__name__ + ".z_vendor")

    def test_duplicate_names_and_invalid_exports_fail_visibly(self):
        """An export is admission intent, so malformed declarations cannot be skipped silently."""
        with driver_package({"one": DECLARATION + '__all__ = ["ExportedDriver"]\n',
                             "two": DECLARATION + '__all__ = ["ExportedDriver"]\n'}) as package:
            with self.assertRaisesRegex(ImportError, "Duplicate driver name"):
                _discover_drivers(package)
        invalid = [
            ('__all__ = "ExportedDriver"\n', "list/tuple"),
            ('__all__ = ["Missing"]\n', "locally defined concrete"),
            ('__all__ = ["Instrument"]\n', "locally defined concrete"),
            ('__all__ = ["SCPIInstrument"]\n', "locally defined concrete"),
            ('helper = 3\n__all__ = ["helper"]\n', "locally defined concrete"),
            ('__all__ = ["_private"]\n', "Invalid driver export"),
            ('class NoKind(Instrument): pass\n__all__ = ["NoKind"]\n', "valid InstrumentKind"),
            ('from abc import ABC, abstractmethod\nclass AbstractDriver(Instrument, ABC):\n'
             '    KIND = InstrumentKind.SENSOR\n    @abstractmethod\n    def read(self): pass\n'
             '__all__ = ["AbstractDriver"]\n', "locally defined concrete"),
            (DECLARATION + 'Alias = ExportedDriver\n__all__ = ["Alias"]\n', "locally defined concrete"),
            ('raise RuntimeError("broken driver import")\n', "broken driver import"),
        ]
        for body, message in invalid:
            with self.subTest(message=message), driver_package({"vendor": body}) as package:
                with self.assertRaisesRegex((ImportError, RuntimeError), message):
                    _discover_drivers(package)

    def test_new_vendor_export_is_importable_and_available_in_real_server_workers(self):
        """Use an isolated package copy to add a driver without editing either initializer or registry."""
        with tempfile.TemporaryDirectory(dir=ROOT / ".agents") as directory:
            root = Path(directory)
            destination = root / "auspex_v2"
            shutil.copytree(ROOT / "src" / "auspex_v2", destination,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            (destination / "instruments" / "drivers" / "new_vendor.py").write_text('''
from .simulated import SimulatedSource
__all__ = ["CatalogSource"]
class CatalogSource(SimulatedSource):
    """Synthetic exported vendor driver for the isolated integration fixture."""
class HiddenSource(SimulatedSource):
    """An unexported class is not admitted, even though it is importable by module."""
''')
            env = dict(os.environ, PYTHONPATH=str(root))
            # Validate a fresh interpreter's actual module attributes and namespace-collision check.
            code = '''from auspex_v2.instruments import CatalogSource
from auspex_v2.instruments.drivers.registry import DRIVERS
import auspex_v2.instruments as instruments
assert DRIVERS["CatalogSource"] is CatalogSource
assert "CatalogSource" in instruments.__all__
assert not hasattr(instruments, "HiddenSource")
assert "HiddenSource" not in DRIVERS
'''
            subprocess.run([sys.executable, "-B", "-c", code], check=True, env=env, capture_output=True, timeout=20)
            socket_path = root / "instrument.sock"
            process = subprocess.Popen([sys.executable, "-B", "-m", "auspex_v2.instruments.server",
                                        "--socket", str(socket_path), "--quiet"], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with InstrumentServerClient(socket_path) as client:
                            client.status()
                        break
                    except (OSError, ConnectionError):
                        if process.poll() is not None:
                            self.fail(process.communicate()[1].decode())
                        self.assertLess(time.monotonic(), deadline, "Server did not start")
                        time.sleep(0.02)
                with InstrumentServerClient(socket_path) as client:
                    client.register("auto", "CatalogSource", "SIM::AUTO")
                    with self.assertRaisesRegex(RemoteError, "not registered by canonical class name"):
                        client.register("hidden", "HiddenSource", "SIM::HIDDEN")
                    client.connect("auto")
                    with client.live("auto") as source:
                        source.frequency = 5e9
                        self.assertEqual(source.frequency, 5e9)
                    client.disconnect("auto")
            finally:
                process.terminate()
                try:
                    _, errors = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                    self.fail("InstrumentServer did not terminate")
            self.assertEqual(process.returncode, 0, errors.decode())
            # A valid driver class must also not overwrite a public declaration/enum name.
            (destination / "instruments" / "drivers" / "collision.py").write_text('''
from .simulated import SimulatedSource
__all__ = ["InstrumentKind"]
class InstrumentKind(SimulatedSource):
    """Invalid public name for a concrete driver."""
''')
            result = subprocess.run([sys.executable, "-B", "-c", "import auspex_v2.instruments"],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("collide with the instrument API", result.stderr)

    def test_imports_do_not_require_visa_and_completion_lists_exported_names(self):
        """Check normal import and Python import-name completion without a hardware backend."""
        code = '''import sys
sys.modules["pyvisa"] = None
from auspex_v2 import instruments
from auspex_v2.instruments.drivers.registry import DRIVERS
assert all(getattr(instruments, n) is driver for n, driver in DRIVERS.items())
import rlcompleter
completer = rlcompleter.Completer({"instruments": instruments})
assert "instruments.AgilentN5183A" in {name.rstrip("(") for name in completer.attr_matches("instruments.Agilent")}
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, timeout=20)

    @unittest.skipUnless(importlib.util.find_spec("IPython"), "IPython is optional")
    def test_real_ipython_from_import_completion(self):
        """Keep InteractiveShell out of the parent process used for spawned-worker tests."""
        code = '''import sys
sys.modules["pyvisa"] = None
from IPython.core.completer import provisionalcompleter
from IPython.core.interactiveshell import InteractiveShell
from traitlets.config import Config
shell = InteractiveShell(ipython_dir=sys.argv[1], user_ns={},
                         config=Config({"HistoryManager": {"hist_file": ":memory:"}}))
for jedi in (False, True):
    completer = shell.Completer  # Includes the shell's real from-import hooks.
    completer.use_jedi = jedi
    for prefix, expected in (("Agilent", {"AgilentN5183A", "AgilentE8363C", "AgilentN5230A"}),
                             ("Simulated", {"SimulatedSource", "SimulatedVNA"})):
        text = "from auspex_v2.instruments import " + prefix
        with provisionalcompleter():
            names = {match.text for match in completer.completions(text, len(text))}
        assert expected <= names, (jedi, expected - names, names)
'''
        with tempfile.TemporaryDirectory(dir=ROOT / ".agents") as directory:
            result = subprocess.run([sys.executable, "-B", "-c", code, directory],
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
