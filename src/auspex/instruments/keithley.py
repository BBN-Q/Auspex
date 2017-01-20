# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

class Keithley2400(SCPIInstrument):
    """Keithley2400 Sourcemeter"""

    current    = FloatCommand(get_string=":sour:curr?",  set_string="sour:curr:lev {:g};")
    voltage    = FloatCommand(get_string=":sour:volt?",  set_string="sour:volt:lev {:g};")
    resistance = FloatCommand(get_string=":read?")

    def __init__(self, resource_name, *args, **kwargs):
        super(Keithley2400, self).__init__(resource_name, *args, **kwargs)
        self.name = "Keithley 2400 Sourcemeter"

    def connect(self, resource_name=None, interface_type=None):
        super(Keithley2400, self).connect(resource_name=resource_name, interface_type=interface_type)
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

    def conf_src_volt(self, comp_current=10e-6, volt_range=1.0, auto_range=True):
        if auto_range:
            self.interface.write(":sour:func volt;:sour:volt:rang:auto 1;")
        else:
            self.interface.write(":sour:func volt;:sour:volt:rang:auto 0;:sour:volt:rang {:g};".format(volt_range))
        self.interface.write(":sens:curr:prot {:g};".format(comp_current))
