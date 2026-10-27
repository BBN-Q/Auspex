"""Instrument declarations and every explicitly exported bundled driver."""

from . import drivers as _drivers
from .kinds import InstrumentKind

# Concrete module attributes support normal import completion without a server,
# constructing a driver, or maintaining a second list of vendor exports.
__all__ = [*_drivers.__all__, "InstrumentKind"]
globals().update({name: getattr(_drivers, name) for name in _drivers.__all__})
