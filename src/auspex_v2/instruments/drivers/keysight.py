"""Unavailable VISA driver stubs; reference: v1/src/auspex/instruments/keysight.py.

Manual review must cover waveform packing, sequence/scenario tables and a
server-compatible upload API. No V1 helper or unfinished transfer code is ported.
"""

from .core import SCPIInstrument
from ..kinds import InstrumentKind

__all__ = []  # No admission or top-level exports until manually implemented.


class KeysightM8190A(SCPIInstrument):
    """Unavailable M8190A AWG; waveform/sequence support needs manual porting."""

    KIND = InstrumentKind.AWG

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("KeysightM8190A requires manual V2 porting and hardware validation")
