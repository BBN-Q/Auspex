"""Unavailable VISA driver stubs; reference: v1/src/auspex/instruments/lakeshore.py."""

from .core import SCPIInstrument
from ..kinds import InstrumentKind

__all__ = []  # No admission or top-level exports until manually implemented.


class LakeShore370(SCPIInstrument):
    """Unavailable 370 bridge; review control units, bounds and heater setup."""

    KIND = InstrumentKind.TEMPERATURE_CONTROLLER

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LakeShore370 requires manual V2 porting and hardware validation")


class LakeShore335(SCPIInstrument):
    """Unavailable 335 controller; V1 heater-limit checks need manual review.

    Heater-resistance codes contradict comments; the 50-ohm limit checks a
    current-code field rather than the resistance field. Do not infer protection.
    """

    KIND = InstrumentKind.TEMPERATURE_CONTROLLER

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LakeShore335 requires manual V2 porting and hardware validation")
