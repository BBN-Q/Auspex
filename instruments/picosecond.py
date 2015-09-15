from .instrument import Instrument, Command, FloatCommand

class Picosecond10070A(Instrument):
    """Picosecond 10070A Pulser"""
    amplitude = Command("amplitude", get_string="amplitude?", set_string="amplitude {:g}")
    delay = FloatCommand("delay", get_string="delay?", set_string="delay {:g}")
    duration = FloatCommand("duration", get_string="duration?", set_string="duration {:g}")
    level = FloatCommand("level", get_string="level?", set_string="level {:g}")
    period = FloatCommand("period", get_string="period?", set_string="period {:g}")
    frequency = FloatCommand("frequency", get_string="frequency?", set_string="frequency {:g}")
    offset = FloatCommand("offset", get_string="offset?", set_string="offset {:g}")
    output = FloatCommand("offset", get_string="offset?", set_string="offset {:g}")

    def __init__(self, resourceName, *args, **kwargs):
        super(Picosecond10070A, self).__init__(resourceName, "Picosecond 10070A Pulser", *args, **kwargs)
        self.write("header off")
        self.write("trigger GPIB")

    # This command is syntactically screwy
    @property
    def output(self):
        return self.ask("enable?") == "YES"
    @output.setter
    def output(self, value):
        if value:
            self.write("enable")
        else:
            self.write("disable")