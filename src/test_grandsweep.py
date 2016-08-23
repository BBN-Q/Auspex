# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.experiment import Parameter, FloatParameter, IntParameter, Experiment, SweepAxis
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.io import WriteToHDF5
from pycontrol.filters.debug import Print

import itertools
import numpy as np
import asyncio
import time, sys
import h5py
import matplotlib.pyplot as plt

from pycontrol.filters.filter import Filter
from pycontrol.stream import InputConnector
from tqdm import tqdm
import time

#from analysis.h5shell import h5shell

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

class SweepExperiment(Experiment):

    # Description
    sample = "CSHE2-C4R1"
    comment = "Field Switching"

    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    duration       = FloatParameter(default=0.0, unit="s")
    measure_current= FloatParameter(default=3e-6, unit="A")
    voltage        = FloatParameter(default=0.0, unit="V")
    iteration      = IntParameter(default=1)

    attempts = 10
    # Things coming back
    resistance = OutputConnector()

    def init_instruments(self):
        print("Initialize instruments...")

        def set_voltage(volt):
            print("Set voltage = {}".format(volt))
            time.sleep(0.02)

        def set_field(field):
            print("Set field = {}".format(field))
            time.sleep(0.02)

        def set_duration(duration):
            print("Set duration = {}".format(duration))
            time.sleep(0.02)

        def set_iteration(iteration):
            print("Set iteration = {}".format(iteration))
            time.sleep(0.02)

        # Assign method
        self.field.assign_method(set_field)
        self.voltage.assign_method(set_voltage)
        self.duration.assign_method(set_duration)
        self.iteration.assign_method(set_iteration)
        print("Done initializing.")

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("attempts", range(self.attempts)))
        self.resistance.set_descriptor(descrip)

    async def run(self):
        """This is run for each step in a sweep."""
        res = np.random.random(self.attempts)
        await self.resistance.push(res)
        logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken,
                                                            self.resistance.num_points() ))
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.2)

    def shutdown_instruments(self):
        print("Shutted down.")

     

if __name__ == '__main__':
    exp = SweepExperiment()

    def extend(sweep):
        print("Extend the value of points of Axis {}".format(sweep.name))
        sweep.points = 2*np.array(sweep.points)
        print("New points of Axis {}: {}".format(sweep.name, sweep.points))


    exp.sample = "Test ProgressBar"
    exp.comment = "Test"
    printer = Print()
    edges = [(exp.resistance,printer.data)]
    exp.set_graph(edges)
    exp.init_instruments()

    sweep_voltage = SweepAxis(exp.voltage, [1,2])
    sweep_duration = SweepAxis(exp.duration, [0.1, 0.2])
    sweep_iteration = SweepAxis(exp.iteration, [1,2], func=extend, params=sweep_voltage)
    sweep_field = SweepAxis(exp.field, [100,200])

    exp.add_sweep_axis(sweep_voltage)
    exp.add_sweep_axis(sweep_duration)
    exp.add_sweep_axis(sweep_iteration)
    exp.add_sweep_axis(sweep_field)
    exp.run_sweeps()
    exp.shutdown_instruments()
