from pycontrol.experiment import Parameter, FloatParameter, IntParameter, Experiment
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.io import WriteToHDF5

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
logger.setLevel(logging.INFO)

# Experimental Topology
# lockin AO 2 -> Analog Attenuator Vdd
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# AWG Sync Marker Out -> DAQmx PFI0
# AWG Samp. Marker Out -> PSPL Trigger

class ProgressBarExperiment(Experiment):

    # Description
    sample = "CSHE2-C4R1"
    comment = "Field Switching"

    # Parameters
    field          = FloatParameter(default=0.0, unit="T")
    measure_current= FloatParameter(default=3e-6, unit="A")
    voltage        = FloatParameter(default=0.0, unit="V")

    # Things coming back
    resistance = OutputConnector()

    def init_instruments(self):
        pass

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        self.resistance.set_descriptor(descrip)

    async def run(self):
        """This is run for each step in a sweep."""
        res = np.random.random(1)
        await self.resistance.push(res)
        logger.debug("Stream has filled {} of {} points".format(self.resistance.points_taken,
                                                            self.resistance.num_points() ))
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.2)

    def shutdown_instruments(self):
        print("Shutted down.")

class ProgressBar(Filter):
    """ Display progress bar(s) on the terminal.

    n: number of progress bars to be display, \
    corresponding to the number of axes (counting from outer most)
    """
    data = InputConnector()
    def __init__(self, num=1):
        super(ProgressBar,self).__init__()
        self.num    = num
        self.bars   = []
        self.w_id   = 0

    async def run(self):
        self.stream = self.data.input_streams[0]
        axes = self.stream.descriptor.axes
        num_axes = len(axes)
        totals = [self.stream.descriptor.num_points_through_axis(axis) for axis in range(num_axes)]
        chunk_sizes = [max(1,self.stream.descriptor.num_points_through_axis(axis+1)) for axis in range(num_axes)]
        self.num = min(self.num, num_axes)

        for i in range(self.num):
            self.bars.append(tqdm(total=totals[i]/chunk_sizes[i]))

        while True:
            if self.stream.done() and self.w_id==self.stream.num_points():
                break

            new_data = np.array(await self.stream.queue.get()).flatten()
            while self.stream.queue.qsize() > 0:
                new_data = np.append(new_data, np.array(self.stream.queue.get_nowait()).flatten())
            self.w_id += new_data.size
            num_data = self.stream.points_taken
            for i in range(self.num):
                if num_data == 0:
                    # Reset the progress bar with a new one
                    self.bars[i].close()
                    self.bars[i] = tqdm(total=totals[i]/chunk_sizes[i])
                pos = int(10*num_data / chunk_sizes[i])/10.0 # One decimal is good enough
                if pos > self.bars[i].n:
                    self.bars[i].update(pos - self.bars[i].n)
                num_data = num_data % chunk_sizes[i]
            

if __name__ == '__main__':
    exp = ProgressBarExperiment()
    exp.sample = "Test ProgressBar"
    exp.comment = "Test"
    progbar = ProgressBar(num=3)
    edges = [(exp.resistance,progbar.data)]
    exp.set_graph(edges)
    exp.init_instruments()
    main_sweep = exp.add_sweep(exp.field,np.linspace(0,-0.02,6))
    exp.add_sweep(exp.measure_current,np.linspace(0,6,5))
    exp.add_sweep(exp.voltage,np.linspace(1,5,3))
    exp.run_sweeps()
    exp.shutdown_instruments()
