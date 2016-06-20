import asyncio
import os
import numpy as np

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter, Quantity
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.plot import Plotter
from pycontrol.filters.average import Average


# from __future__ import print_function, division
# import logging
# import time
# from functools import partial

# logger = logging.getLogger('pycontrol')
# logging.basicConfig(format='%(name)s - %(levelname)s: \t%(asctime)s: \t%(message)s')
# logger.setLevel(logging.INFO)

# import numpy as np
# # import scipy as sp
# # import pandas as pd

# from pycontrol.instruments.instrument import Instrument, StringCommand
# from pycontrol.instruments.picosecond import Picosecond10070A
# from pycontrol.sweep import Sweep
# from pycontrol.experiment import FloatParameter, Quantity, Experiment

# class Magnet(Instrument):
#     field = StringCommand(get_string=":field?", set_string=":field %g Oe;")

# class Keithley(Instrument):
#     resistance = StringCommand(get_string=":res?", set_string=":res %g Oe;")
#     testing = StringCommand(get_string=":test?", set_string=":test %g Oe;")

# class TestExperiment(Experiment):

#     # Create instances of instruments
#     mag    = Magnet("FAKE::RESOURCE::NAME")
#     keith1 = Keithley("FAKE::RESOURCE::NAME")
#     keith2 = Keithley("FAKE::RESOURCE::NAME")

#     # Parameters
#     field_x = FloatParameter(name="Field X", unit="G")
#     field_y = FloatParameter(name="Field Y", unit="G")

#     # Quantities
#     resistance_trans = Quantity(name="Transverse Resistance", unit="Ohm")
#     resistance_long = Quantity(name="Longitudinal Resistance", unit="Ohm")

#     def init_instruments(self):
#         self.field_x.assign_method(lambda x: time.sleep(0.01))
#         self.field_y.assign_method(lambda x: time.sleep(0.01))
#         self.resistance_trans.assign_method(lambda: self.field_x.value - self.field_y.value + 20*np.random.random())
#         self.resistance_long.assign_method(lambda: self.field_x.value - self.field_y.value + 40*np.random.random())

#     def run(self):
#         for quant in self._quantities:
#             self._quantities[quant].measure()
#         logger.info("R_t = {}".format(self.resistance_trans.value))

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Create instances of instruments
    fake_instr_1 = TestInstrument1("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")

    # DataStreams
    voltage = OutputConnector()

    # Constants
    samples = 5
    time_val = 0

    def init_instruments(self):
        self.field.assign_method(lambda x: pass)
        self.freq.assign_method(lambda x: pass)

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        logger.debug("Data taker running (inner loop)")
        time_step = 0.1
        await asyncio.sleep(0.002)
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(5) + 0.1*np.random.random(5)
        self.time_val += time_step
        await self.voltage.push(data_row)
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

if __name__ == '__main__':

    exp = TestExperiment()
    # pri = Print(name="Printer")
    avg = Averager(name="Collapse Sample")
    pl1 = Plotter(name="Scope")
    pl2 = Plotter(name="Accumulate")

    avg.axis = 'samples'
    pl1.

    edges = [(exp.voltage, avg.data)]
    exp.set_graph(edges)

    exp.init_instruments()
    exp.add_sweep(exp.field, np.linspace(0,100.0,11))
    exp.add_sweep(exp.freq, np.linspace(0,10.0,3))
    exp.run_sweeps()

    # # Create an instance of the procedure
    # proc = TestExperiment()

    # # Define a sweep over prarameters
    # sweep1 = Sweep(proc)
    # field_x = sweep1.add_parameter(proc.field_y, np.arange(-100, 101, 10))
    # field_y = sweep1.add_parameter(proc.field_x, np.arange(-100, 101, 10))

    # plot1 = sweep1.add_plotter("ResistanceL Vs Field", proc.field_x, proc.resistance_long, color="firebrick", line_width=2)
    # plot2 = sweep1.add_multiplotter("Resistances Vs Field", [proc.field_x, proc.field_x], [proc.resistance_trans, proc.resistance_long],
    #                                 line_color=["firebrick","navy"], line_width=2)

    # # Have to pass sweep parmaters here in order that the plotter knows the x,y grid
    # plot3 = sweep1.add_plotter2d("A Whole New Dimension", field_x, field_y, proc.resistance_trans, palette="Spectral11")

    # # Hooks for clearing the plots at the end of a sub-sweep
    # proc.field_y.add_post_push_hook(plot1.clear)
    # proc.field_y.add_post_push_hook(plot2.clear)

    # # Make sure we update the data at the end of a trace
    # proc.field_y.add_pre_push_hook(partial(plot1.update, force=True))
    # proc.field_y.add_pre_push_hook(partial(plot2.update, force=True))

    # sweep1.run()
