# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Keithley2400']

import time
from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

class Keithley2400(SCPIInstrument):
    """Keithley2400 Sourcemeter"""

    SOUR_VALS  = ['VOLT','CURR']
    SENSE_VALS = ['VOLT','CURR','RES']

    source     = StringCommand(scpi_string=":SOUR:FUNC",allowed_values=SOUR_VALS)
    sense      = StringCommand(scpi_string=":SENS:FUNC",allowed_values=SENSE_VALS)
    current    = FloatCommand(get_string=":MEAS:CURR?")
    voltage    = FloatCommand(get_string=":MEAS:VOLT?")
    resistance = FloatCommand(get_string=":MEAS:RES?")


    def __init__(self, resource_name, *args, **kwargs):
        super(Keithley2400, self).__init__(resource_name, *args, **kwargs)
        self.name = "Keithley 2400 Sourcemeter"

    def connect(self, resource_name=None, interface_type=None):
        super(Keithley2400, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface.write("format:data ascii")
        self.interface._resource.read_termination = "\n"

    def triad(self, freq=440, duration=0.2, minor=False, down=False):
        beeps = [(freq, duration)]
        if minor:
            beeps.append((freq*6.0/5.0, duration))
        else:
            beeps.append((freq*5.0/4.0, duration))
        beeps.append((freq*6.0/4.0, duration))
        if down:
            beeps = beeps[::-1]
        for f, d in beeps:
            self.beep(f, d)
            time.sleep(duration)

    def beep(self, freq, dur):
        self.interface.write(":SYST:BEEP {:g}, {:g}".format(freq, dur))

#Level of Source

    @property
    def level(self):
        return self.interface.query(":SOUR:{}:LEV?".format(self.source))

    @level.setter
    def level(self, level):
        self.interface.write(":SOUR:{}:LEV {:g}".format(self.source,level))

#Range of Source

    @property
    def source_range(self):
        auto = self.interface.query(":SOUR:{}:RANG:AUTO?".format(self.source))
        if auto == 1:
            return "AUTO"
        else:
            return self.interface.query(":SOUR:{}:RANG?".format(self.source))

    @source_range.setter
    def source_range(self, range):
        source = self.source
        if range != "AUTO":
            self.interface.write(":SOUR:{}:RANG:AUTO 0;:SOUR:{}:RANG {:g}".format(source,source,range))
        else:
            self.interface.write(":SOUR:{}:RANG:AUTO 1".format(source))

#Compliance of Sense

    @property
    def compliance(self):
        return self.interface.query(":SENS:{}:PROT?".format(self.sense))

    @compliance.setter
    def compliance(self, comp):
        self.interface.write(":SENS:{}:PROT {:g}".format(self.sense,comp))

#Range of Sense

    @property
    def sense_range(self):
        auto = self.interface.query(":SENS:{}:RANG:AUTO?".format(self.source))
        if auto == 1:
            return "AUTO"
        else:
            return self.interface.query(":SENS:{}:RANG?".format(self.source))

    @sense_range.setter
    def sense_range(self, range):
        source = self.source
        if range != "AUTO":
            self.interface.write(":SOUR:{}:RANG:AUTO 0;:SOUR:{}:RANG {:g}".format(source,source,range))
        else:
            self.interface.write(":SOUR:{}:RANG:AUTO 1".format(source))

    # One must configure the measurement before the source to avoid potential range issues
    def conf_meas_res(self, NPLC=1, res_range=1000.0, auto_range=True):
        self.interface.write(":sens:func \"res\";:sens:res:mode man;:sens:res:nplc {:f};:form:elem res;".format(NPLC))
        if auto_range:
            self.interface.write(":sens:res:rang:auto 1;")
        else:
            self.interface.write(":sens:res:rang:auto 0;:sens:res:rang {:g}".format(res_range))
