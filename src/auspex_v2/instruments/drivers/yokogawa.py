"""Yokogawa GS200 source and sense meter."""

from .core import BoolCommand, FloatCommand, IntCommand, RampCommand, SCPIInstrument, StringCommand
from ..kinds import InstrumentKind


class YokogawaGS200(SCPIInstrument):
    """Control GS200 sourcing, sensing and bounded level ramps over VISA."""

    IDN_STRING = "GS200"
    KIND = InstrumentKind.DC_SOURCE
    instrument_type = "Current source"

    mode = StringCommand(scpi_string=":source:function",
                         value_map={"current": "CURR", "voltage": "VOLT"})
    level = FloatCommand(scpi_string=":source:level", aliases=("value",))
    output_range = FloatCommand(scpi_string=":source:range")
    protection_volts = FloatCommand(scpi_string=":source:protection:voltage")
    protection_current = FloatCommand(scpi_string=":source:protection:current")
    sense = BoolCommand(scpi_string=":sense:state", value_map={True: "1", False: "0"})
    output = BoolCommand(scpi_string=":output:state", value_map={True: "1", False: "0"})
    sense_value = FloatCommand(get_string=":fetch?")
    averaging_nplc = IntCommand(scpi_string=":sense:nplc")
    ramp = RampCommand(scpi_string=":source:level", increment=1e-4, pause=0.02,
                       value_range=(-0.1, 0.1))

    def connect(self, resource_name=None):
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write_termination = "\n"
            self.interface.write(":sense:trigger immediate")
        except Exception:
            self.disconnect()
            raise
