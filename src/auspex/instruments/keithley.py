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
import numpy as np
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

class Keithley2400(SCPIInstrument):
    """Keithley2400 Sourcemeter"""

    SOUR_VALS  = ['VOLT','CURR']
    SENSE_VALS = ['VOLT','CURR','RES']
    MODE_VALS = ['SWE','LIST','FIX']
    SWEEP_RANG = ['BEST','AUTO','FIX']
    SWEEP_SPACE = ['LIN','LOG']
    SWEEP_DIR = ['UP','DOWN']
    SWEEP_ABOR = ['NEV','EARL','LATE']

    source          = StringCommand(scpi_string=":SOUR:FUNC",allowed_values=SOUR_VALS)
    sweep_range     = StringCommand(scpi_string=":SOUR:SWE:RANG",allowed_values=SWEEP_RANG)
    sweep_space     = StringCommand(scpi_string=":SOUR:SWE:SPAC",allowed_values=SWEEP_SPACE)
    sweep_direction = StringCommand(scpi_string=":SOUR:SWE:DIRE",allowed_values=SWEEP_DIR)
    sweep_abort     = StringCommand(scpi_string=":SOUR:SWE:CAB",allowed_values=SWEEP_ABOR)
    sweep_points    = IntCommand(scpi_string=":SOUR:SWE:POIN")
    output          = StringCommand(scpi_string=":OUTP",value_map={'ON': '1', 'OFF': '0'})

    sense           = StringCommand(get_string=":SENS:FUNC?",set_string=":SENS:FUNC '{:s}'",allowed_values=SENSE_VALS)
    concurrent      = StringCommand(scpi_string=":SENS:FUNC:CONC",value_map={'ON': '1', 'OFF': '0'})
    current         = FloatCommand(get_string=":MEAS:CURR?")
    voltage         = FloatCommand(get_string=":MEAS:VOLT?")
    resistance      = FloatCommand(get_string=":MEAS:RES?")


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

# Run Measurement Sweep
    def sweep(self):

        #Construct list of variables to report from sense and source lists
        var_list = self.sense.split(",")
        var_list = list(map(lambda x: x.replace(':DC','').replace('"',''),var_list))
        var_list.append(self.source)

        #Format Sweep Output
        self.interface.write((":FORM:ELEM "+','.join(['{:s}']*len(var_list))).format(*var_list))
        nvar = len(self.interface.query(":FORM:ELEM?").split(","))

        #Setup Trigger
        sweep_points = self.sweep_points
        self.interface.write(":TRIG:COUN {:d}".format(sweep_points))

        #Set long timeout for read
        tmout = self.interface._resource.timeout
        self.interface._resource.timeout = 12. * 60. * 60. * 1000. #milliseconds

        #Run Sweep
        if self.output == 'OFF':
            self.output = 'ON'

        sweep = np.array(self.interface.query(":READ?").split(","), dtype=np.float32).reshape(sweep_points,nvar)
        self.output = 'OFF'

        self.interface._resource.timeout = tmout
        self.beep(261.6,0.25)

        return sweep



#Level of Source

    @property
    def level(self):
        return self.interface.query(":SOUR:{:s}:LEV?".format(self.source))

    @level.setter
    def level(self, level):
        self.interface.write(":SOUR:{:s}:LEV {:g}".format(self.source,level))

#Mode of Source

    @property
    def mode(self):
        return self.interface.query(":SOUR:{:s}:MODE?".format(self.source))

    @mode.setter
    def mode(self, mode):
        if mode not in self.MODE_VALS:
            raise ValueError(("Mode must be "+'|'.join(['{}']*len(self.MODE_VALS))).format(*self.MODE_VALS))
        self.interface.write(":SOUR:{:s}:MODE {:s}".format(self.source,mode))

#Range of Source

    @property
    def source_range(self):
        auto = int(self.interface.query(":SOUR:{:s}:RANG:AUTO?".format(self.source)))
        if auto == 1:
            return "AUTO"
        else:
            return self.interface.query(":SOUR:{:s}:RANG?".format(self.source))

    @source_range.setter
    def source_range(self, range):
        source = self.source
        if range != "AUTO":
            self.interface.write(":SOUR:{:s}:RANG:AUTO 0;:SOUR:{:s}:RANG {:g}".format(source,source,range))
        else:
            self.interface.write(":SOUR:{:s}:RANG:AUTO 1".format(source))

#Sweep Start

    @property
    def sweep_start(self):
        return self.interface.query(":SOUR:{:s}:STAR?".format(self.source))

    @sweep_start.setter
    def sweep_start(self, start):
        self.interface.write(":SOUR:{:s}:STAR {:g}".format(self.source,start))

#Sweep Stop

    @property
    def sweep_stop(self):
        return self.interface.query(":SOUR:{:s}:STOP?".format(self.source))

    @sweep_stop.setter
    def sweep_stop(self, stop):
        self.interface.write(":SOUR:{:s}:STOP {:g}".format(self.source,stop))

#Sweep Step

    @property
    def sweep_step(self):
        return self.interface.query(":SOUR:{:s}:STEP?".format(self.source))

    @sweep_step.setter
    def sweep_step(self, step):
        self.interface.write(":SOUR:{:s}:STEP {:g}".format(self.source,step))

#Compliance of Sense

    @property
    def compliance(self):
        return self.interface.query(":SENS:{:s}:PROT?".format(self.sense.replace('"','')))

    @compliance.setter
    def compliance(self, comp):
        self.interface.write(":SENS:{:s}:PROT {:g}".format(self.sense.replace('"',''),comp))

#Range of Sense

    @property
    def sense_range(self):
        auto = int(self.interface.query(":SENS:{:s}:RANG:AUTO?".format(self.sense.replace('"',''))))
        if auto == 1:
            return "AUTO"
        else:
            return self.interface.query(":SENS:{:s}:RANG?".format(self.sense.replace('"','')))

    @sense_range.setter
    def sense_range(self, range):
        sense = self.sense.replace('"','')
        if range != "AUTO":
            self.interface.write(":SENS:{:s}:RANG:AUTO 0;:SENS:{:s}:RANG {:g}".format(sense,sense,range))
        else:
            self.interface.write(":SENS:{:s}:RANG:AUTO 1".format(sense))

    # One must configure the measurement before the source to avoid potential range issues
    def conf_meas_res(self, NPLC=1, res_range=1000.0, auto_range=True):
        self.interface.write(":sens:func \"res\";:sens:res:mode man;:sens:res:nplc {:f};:form:elem res;".format(NPLC))
        if auto_range:
            self.interface.write(":sens:res:rang:auto 1;")
        else:
            self.interface.write(":sens:res:rang:auto 0;:sens:res:rang {:g}".format(res_range))
