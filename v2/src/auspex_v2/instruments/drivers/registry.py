"""Explicit trusted driver allowlist shared by instrument and experiment services."""
from .agilent import AgilentE8363C, AgilentN5183A, AgilentN5230A
from .simulated import SimulatedSource, SimulatedVNA
from .yokogawa import YokogawaGS200


DRIVERS = {driver.__name__: driver for driver in (
    SimulatedSource,
    SimulatedVNA,
    AgilentN5183A,
    AgilentE8363C,
    AgilentN5230A,
    YokogawaGS200,
)}
