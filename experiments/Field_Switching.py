# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430

from pycontrol.experiment import Parameter, FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.debug import Print
from pycontrol.filters.io import WriteToHDF5

import itertools
import numpy as np
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt

from analysis.h5shell import h5shell

import logging
logger = logging.getLogger('pycontrol')
logging.basicConfig(format='%(name)s-%(levelname)s: \t%(message)s')
logger.setLevel(logging.DEBUG)

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

class FieldSwitchingExperiment(Experiment):
    """ Field Switching Experiment
    Measure pillar resistance while sweeping bias field
    """

    # Description
    sample = "CSHE2-C4R1"
    comment = "Field Switching"

    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    measure_current= FloatParameter(default=3e-6, unit="A")

    # Instrument resources
    mag   = AMI430("192.168.5.109")
    keith = Keithley2400("GPIB0::25::INSTR")

    # Things coming back
    resistance = OutputConnector()

    def init_instruments(self):

        # ===================
        #    Setup the Keithley
        # ===================

        self.keith.triad()
        self.keith.conf_meas_res(res_range=1e5)
        self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        self.keith.current = self.measure_current.value
        self.mag.ramp()

        # Assign methods
        self.field.assign_method(self.mag.set_field)

        # Create hooks for relevant delays
        self.field.add_post_push_hook(lambda: time.sleep(0.1))

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        self.resistance.set_descriptor(descrip)

    async def run(self):
        """This is run for each step in a sweep."""
        res = self.keith.resistance
        await self.resistance.push(res)
        # Seemingly we need to give the filters some time to catch up here...
        logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken,
                                                            self.resistance.num_points() ))
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)

    def shutdown_instruments(self):
        self.keith.current = 0.0e-5
        self.mag.zero()

def extract_data(fname):
    f = h5shell(fname, 'r')
    axes = f['axes']
    fields = axes[f.grep('field','axes')[0]].value
    resistances = f[f.grep('data')[0]].value
    f.close()
    return fields, resistances

if __name__ == '__main__':
    exp = FieldSwitchingExperiment()
    exp.sample = "CSHE-2 C4R7"
    exp.comment = "Field Sweep"
    wr = WriteToHDF5("data\CSHE-Switching\CSHE-Die2-C4R7\CSHE2-C4R7-FieldSwitch_2016-06-24.h5")
    edges = [(exp.resistance, wr.data)]
    exp.set_graph(edges)
    exp.init_instruments()

    fields = np.linspace(0,-0.02,20)
    fields = np.append(fields, np.flipud(fields))
    main_sweep = exp.add_sweep(exp.field,fields)
    exp.run_sweeps()
    exp.shutdown_instruments()
