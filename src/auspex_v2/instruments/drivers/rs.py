"""VISA drivers for Rohde & Schwarz instruments, ported from V1."""

from .core import SCPIInstrument, FloatCommand, StringCommand, experimental
from ..kinds import InstrumentKind

__all__ = ["SMB100B"]


@experimental
class SMB100B(SCPIInstrument):
    """Rohde & Schwarz SMB100B source; bare IPv4 uses HiSLIP instrument zero."""

    KIND = InstrumentKind.MICROWAVE_SOURCE
    IDN_STRING = "SMB100B"
    VISA_IPV4_DEVICE = "hislip0::INSTR"
    instrument_type = "Microwave Source"

    frequency = FloatCommand(scpi_string="SOUR1:FREQ")
    power = FloatCommand(scpi_string=":SOUR1:POWER")
    phase = FloatCommand(scpi_string=":SOUR1:PHASE")
    output = StringCommand(scpi_string="OUTP", value_map={True: "1", False: "0"})
    reference = StringCommand(scpi_string=":SOUR:ROSC:SOUR")

    def connect(self, resource_name=None):
        """Open VISA and configure line endings without changing RF settings."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
        except Exception:
            self.disconnect()
            raise
