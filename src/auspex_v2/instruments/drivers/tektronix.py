"""Unavailable VISA driver stubs; reference: v1/src/auspex/instruments/tektronix.py.

V1 scope math acquisition is an empty stub; curve scaling, time axes and binary
transfer semantics need manual review rather than copying unfinished methods.
"""

from .core import SCPIInstrument
from ..kinds import InstrumentKind

__all__ = []  # No admission or top-level exports until manually implemented.


class DPO72004C(SCPIInstrument):
    """Unavailable DPO72004C scope; review binary curves and fast-frame shape."""

    KIND = InstrumentKind.OSCILLOSCOPE

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DPO72004C requires manual V2 porting and hardware validation")


class DPO2024(SCPIInstrument):
    """Unavailable DPO2024 scope; waveform acquisition needs manual porting."""

    KIND = InstrumentKind.OSCILLOSCOPE

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DPO2024 requires manual V2 porting and hardware validation")


class DPO2014(SCPIInstrument):
    """Unavailable DPO2014 scope; waveform acquisition needs manual porting."""

    KIND = InstrumentKind.OSCILLOSCOPE

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DPO2014 requires manual V2 porting and hardware validation")


class RSA3308A(SCPIInstrument):
    """Unavailable RSA3308A analyzer; trace acquisition needs manual porting."""

    KIND = InstrumentKind.SPECTRUM_ANALYZER

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RSA3308A requires manual V2 porting and hardware validation")
