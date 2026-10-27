"""Unavailable VISA driver stubs; reference: v1/src/auspex/instruments/stanford.py."""

from .core import SCPIInstrument
from ..kinds import InstrumentKind

__all__ = []  # No admission or top-level exports until manually implemented.


class SR830(SCPIInstrument):
    """Unavailable SR830 lock-in; review time-constant maps and buffer decoding."""

    KIND = InstrumentKind.LOCKIN

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SR830 requires manual V2 porting and hardware validation")


class SR865(SCPIInstrument):
    """Unavailable SR865 lock-in; review time-constant maps and capture semantics."""

    KIND = InstrumentKind.LOCKIN

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SR865 requires manual V2 porting and hardware validation")
