"""Direct-use declarations and bundled driver exports; imports never open hardware."""

from .core import (BoolCommand, Command, FloatCommand, Instrument, InstrumentError,
                   IntCommand, RampCommand, SCPIInstrument, StringCommand, experimental,
                   managed_property, operation)

__all__ = ["Instrument", "InstrumentError", "SCPIInstrument", "Command",
           "BoolCommand", "StringCommand", "FloatCommand", "IntCommand",
           "RampCommand", "managed_property", "operation", "experimental"]

from .registry import DRIVERS as _drivers

_conflicts = set(_drivers) & (set(globals()) | {"InstrumentKind", "drivers", "kinds", "server"})
if _conflicts:
    raise ImportError(f"Driver names collide with the instrument API: {sorted(_conflicts)}")
__all__ += list(_drivers)
globals().update(_drivers)
