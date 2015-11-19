from .instrument import Instrument, Command, FloatCommand

class Picosecond10070A(Instrument):
    """Picosecond 10070A Pulser"""
    amplitude = FloatCommand("amplitude", get_string="amplitude?", set_string="amplitude {:g}")
    delay     = FloatCommand("delay", get_string="delay?", set_string="delay {:g}")
    duration  = FloatCommand("duration", get_string="duration?", set_string="duration {:g}")
    level     = FloatCommand("level", get_string="level?", set_string="level {:g}")
    period    = FloatCommand("period", get_string="period?", set_string="period {:g}")
    frequency = FloatCommand("frequency", get_string="frequency?", set_string="frequency {:g}")
    offset    = FloatCommand("offset", get_string="offset?", set_string="offset {:g}")

    def __init__(self, name, resource_name, *args, **kwargs):
        super(Picosecond10070A, self).__init__(name, resource_name, *args, **kwargs)
        self.interface.write("header off")
        self.interface.write("trigger GPIB")
        self.interface._resource.read_termination = u"\n"

    # This command is syntactically screwy
    @property
    def output(self):
        return self.interface.query("enable?") == "YES"
    @output.setter
    def output(self, value):
        if value:
            self.interface.write("enable")
        else:
            self.interface.write("disable")

    def trigger(self):
        self.interface.write("*TRG")
