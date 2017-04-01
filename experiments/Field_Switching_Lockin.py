# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import SR865
from auspex.instruments import AMI430
from auspex.experiment import Experiment
from auspex.parameter import FloatParameter
from auspex.stream import OutputConnector
from auspex.filters import WriteToHDF5, Plotter
from auspex.log import logger

import numpy as np
import asyncio
import time
import datetime

class FieldSwitchingLockinExperiment(Experiment):
    """ Field Switching Experimen: measure resistance on Keithley while sweeping AMI430 field
    """
    field           = FloatParameter(default=0.0, unit="T")
    # measure_current = FloatParameter(default=3e-6, unit="A")
    resistance      = OutputConnector(unit="Ohm")

    res_reference = 1e6
    vout = 10e-3
    mag   = AMI430("192.168.5.109")
    # keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    def init_instruments(self):
        # self.keith.triad()
        # self.keith.conf_meas_res(NPLC=10, res_range=1e5)
        # self.keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
        # self.keith.current = self.measure_current.value

        # Initialize lockin
        # self.lock.this = that
        self.lock.amp = self.vout
        self.mag.ramp()

        # self.measure_current.assign_method(self.keith.set_current)
        self.field.assign_method(self.mag.set_field)
        self.field.add_post_push_hook(lambda: time.sleep(10)) # Field set delay

    async def run(self):
        """This is run for each step in a sweep."""
        # await self.resistance.push(self.keith.resistance)
        # vout = self.lock.x
        await self.resistance.push(self.lock.mag*self.res_reference/self.lock.amp)
        logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken,
                                                                self.resistance.num_points() ))
        await asyncio.sleep(10) # Give the filters some time to catch up?

    def shutdown_instruments(self):
        # self.keith.current = 0.0e-5
        self.mag.zero()

if __name__ == '__main__':
    sample_name = "Hypress_3_2_COSTM3"
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    file_path = "data\CRAM01\{samp:}\{samp:}-FieldSwitch_{date:}.h5".format(samp=sample_name, date=date)

    exp = FieldSwitchingLockinExperiment()
    wr  = WriteToHDF5(file_path)
    plt = Plotter()

    edges = [(exp.resistance, wr.sink),
             (exp.resistance, plt.sink)]
    exp.set_graph(edges)

    fields = np.linspace(-0.4,0.4,200)
    fields = np.append(fields, np.flipud(fields))
    main_sweep = exp.add_sweep(exp.field,fields)
    exp.run_sweeps()
