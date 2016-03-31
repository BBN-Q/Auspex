from .instrument import Instrument, Command, FloatCommand, IntCommand

class YokogawaGS200(Instrument):
    """YokogawaGS200 Current source"""

    function = Command("function", scpi_string=":source:function", value_map={"current": "CURR", "voltage": "VOLT"})
    level = FloatCommand("level", scpi_string=":source:level")
    protection_volts   = FloatCommand("protection voltage", scpi_string=":source:protection:voltage")
    protection_current = FloatCommand("protection current", scpi_string=":source:protection:current")
    sense = Command("sense state", scpi_string=":sense:state", value_map={True: "1", False: "0"})
    output = Command("output", scpi_string=":output:state", value_map={True: "1", False: "0"})
    sense_value = FloatCommand("get sense value", get_string=":fetch?")
    averaging_nplc = IntCommand("set NPLC", scpi_string=":sense:nplc")

    def __init__(self, name, resource_name, *args, **kwargs):
        super(YokogawaGS200, self).__init__(name, resource_name, *args, **kwargs)
        self.interface.write(":sense:trigger immediate")
        self.interface._resource.read_termination = "\n"
