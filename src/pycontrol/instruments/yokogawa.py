from .instrument import Instrument, StringCommand, FloatCommand, IntCommand

class YokogawaGS200(Instrument):
    """YokogawaGS200 Current source"""

    function           = StringCommand(scpi_string=":source:function",
                          value_map={"current": "CURR", "voltage": "VOLT"})
    level              = FloatCommand(scpi_string=":source:level")
    protection_volts   = FloatCommand(scpi_string=":source:protection:voltage")
    protection_current = FloatCommand(scpi_string=":source:protection:current")
    sense              = StringCommand(scpi_string=":sense:state", value_map={True: "1", False: "0"})
    output             = StringCommand(scpi_string=":output:state", value_map={True: "1", False: "0"})
    sense_value        = FloatCommand(get_string=":fetch?")
    averaging_nplc     = IntCommand(scpi_string=":sense:nplc") # Number of power level cycles (60Hz)

    def __init__(self, resource_name, *args, **kwargs):
        super(YokogawaGS200, self).__init__(resource_name, *args, **kwargs)
        self.interface.write(":sense:trigger immediate")
        self.interface._resource.read_termination = "\n"
