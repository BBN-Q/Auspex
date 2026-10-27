"""Bundled module exports are the single trusted driver admission catalog."""

import importlib
import inspect
import pkgutil
import sys

from .core import Instrument, SCPIInstrument
from ..kinds import instrument_kind


def _discover_drivers(package):
    """Collect explicit, locally defined concrete drivers without constructing them.

    Only direct modules in the bundled package are searched. A module without
    __all__ contributes nothing. Broken imports/exports are errors, not silently
    missing drivers. Optional backend imports must stay inside connect().
    """
    drivers = {}
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if info.ispkg or info.name.startswith("_") or info.name == "registry":
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        exports = getattr(module, "__all__", ())
        if not isinstance(exports, (list, tuple)):
            raise ImportError(f"{module.__name__}.__all__ must be a list/tuple of driver names")
        for name in exports:
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ImportError(f"Invalid driver export {name!r} in {module.__name__}")
            driver = getattr(module, name, None)
            if (not inspect.isclass(driver) or not issubclass(driver, Instrument)
                    or driver in (Instrument, SCPIInstrument) or inspect.isabstract(driver)
                    or driver.__module__ != module.__name__ or driver.__name__ != name):
                raise ImportError(f"{module.__name__}.{name} must be a locally defined concrete Instrument class")
            try:
                instrument_kind(driver.KIND)
            except ValueError as exc:
                raise ImportError(f"{module.__name__}.{name} needs a valid InstrumentKind") from exc
            if name in drivers:
                raise ImportError(f"Duplicate driver name {name}: {drivers[name].__module__} and {module.__name__}")
            drivers[name] = driver
    return drivers


DRIVERS = _discover_drivers(sys.modules[__package__])
