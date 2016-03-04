from .instrument import Instrument, Command, FloatCommand, IntCommand

class Keithley2400(Instrument):
    """Keithley2400 Sourcemeter"""

    current = FloatCommand("Source Current",  get_string=":sour:curr?",  set_string="sour:curr:lev {:g};")
    resistance = FloatCommand("Resistance Value", get_string=":read?")

    def __init__(self, name, resource_name, *args, **kwargs):
        super(Keithley2400, self).__init__(name, resource_name, *args, **kwargs)
        self.interface.write("format:data ascii")
        self.interface._resource.read_termination = "\n"

    def triad(self, freq=440, duration=0.2, minor=False):
        import time
        self.beep(freq, duration)
        time.sleep(duration)
        if minor:
            self.beep(freq*6.0/5.0, duration)
        else:
            self.beep(freq*5.0/4.0, duration)
        time.sleep(duration)
        self.beep(freq*6.0/4.0, duration)

    def beep(self, freq, dur):
        self.interface.write(":SYST:BEEP {:g}, {:g}".format(freq, dur))

    # One must configure the measurement before the source to avoid potential range issues
    def conf_meas_res(self, NPLC=1, res_range=1000.0, auto_range=True):
        self.interface.write(":sens:func \"res\";:sens:res:mode man;:sens:res:nplc {:f};:form:elem res;".format(NPLC))
        if auto_range:
            self.interface.write(":sens:res:rang:auto 1;")
        else:
            self.interface.write(":sens:res:rang:auto 0;:sens:res:rang {:g}".format(res_range))

    def conf_src_curr(self, comp_voltage=0.1, curr_range=1.0e-3, auto_range=True):
        if auto_range:
            self.interface.write(":sour:func curr;:sour:curr:rang:auto 1;")
        else:
            self.interface.write(":sour:func curr;:sour:curr:rang:auto 0;:sour:curr:rang {:g};".format(curr_range))
        self.interface.write(":sens:volt:prot {:g};".format(comp_voltage))
