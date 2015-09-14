from .instrument import Instrument

class Picosecond10070A(Instrument):
    """Picosecond 10070A Pulser"""
    amplitude = Control("amplitude", get_string="amplitude?", set_string="amplitude {:g}")
    delay = Control("delay", get_string="delay?", set_string="delay {:g}")
    duration = Control("duration", get_string="duration?", set_string="duration {:g}")
    level = Control("level", get_string="level?", set_string="level {:g}")
    period = Control("period", get_string="period?", set_string="period {:g}")
    frequency = Control("frequency", get_string="frequency?", set_string="frequency {:g}")
    offset = Control("offset", get_string="offset?", set_string="offset {:g}")
    output = Control("offset", get_string="offset?", set_string="offset {:g}")

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