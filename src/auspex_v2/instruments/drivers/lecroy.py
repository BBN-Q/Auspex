"""Unavailable VISA driver stubs; reference: v1/src/auspex/instruments/lecroy.py."""

from .core import SCPIInstrument
from ..kinds import InstrumentKind

__all__ = []  # No admission or top-level exports until manually implemented.


class HDO6104(SCPIInstrument):
    """Unavailable HDO6104 scope; V1 waveform byte order needs manual review.

    V1 writes COMM_ORDER HI (MSB first) but decodes as little-endian.
    """

    KIND = InstrumentKind.OSCILLOSCOPE

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HDO6104 requires manual V2 porting and hardware validation")
