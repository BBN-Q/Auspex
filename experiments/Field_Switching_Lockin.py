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
    resistance      = OutputConnector(unit="Ohm")

    res_reference = 1e3
    vsource  = 10e-3
    mag   = AMI430("192.168.5.109")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    def init_instruments(self):
        # Initialize lockin
        self.lock.amp = self.vsource
        self.lock.tc = 3
        self.mag.ramp()
        self.delay = self.lock.measure_delay()
        self.field.assign_method(self.mag.set_field)
        self.field.add_post_push_hook(lambda: time.sleep(0.1)) # Field set delay
        time.sleep(self.delay)

    async def run(self):
        """This is run for each step in a sweep."""

        await asyncio.sleep(self.delay)
        await self.resistance.push(self.res_reference/((self.lock.amp/self.lock.mag)-1.0))

    def shutdown_instruments(self):
        self.mag.zero()
        self.lock.amp = 0

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
