import asyncio
import os
import numpy as np
import sys

from pycontrol.instruments.instrument import Instrument, StringCommand, FloatCommand, IntCommand
from pycontrol.experiment import Experiment, FloatParameter
from pycontrol.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from pycontrol.filters.plot import Plotter
from pycontrol.filters.average import Average
from pycontrol.filters.debug import Print

from pycontrol.logging import logger, logging
logger.setLevel(logging.DEBUG)

class TestInstrument(Instrument):
    frequency = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(name="enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Create instances of instruments
    fake_instr_1 = TestInstrument("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq  = FloatParameter(unit="Hz")

    # DataStreams
    voltage = OutputConnector()

    # Constants
    samples    = 10
    num_trials = 5
    time_val   = 0
    time_step  = 0.1

    def init_instruments(self):
        pass

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        descrip.add_axis(DataAxis("trials", list(range(self.num_trials))))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        data_row = np.sin(2*np.pi*self.time_val)*np.ones(self.samples) 
        
        self.time_val += self.time_step
        for i in range(self.num_trials):
            await asyncio.sleep(0.05)
            await self.voltage.push(data_row + 0.05*np.random.random(self.samples) )
        
        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

if __name__ == '__main__':

    exp = TestExperiment()
    avg1 = Average(name="Collapse Samples")
    avg2 = Average(name="Collapse Trials")
    # pl1 = Plotter(name="Scope")
    # pl2 = Plotter(name="Accumulate")
    pl3 = Plotter(name="Sinusoid")
    # pri = Print("wtf")

    avg1.axis = 'samples'
    avg2.axis = 'trials'

    # sys.exit()
    # pl1.axes = []

    edges = [
             # (exp.voltage, pl1.data),
             (exp.voltage, avg1.data),
             # (avg1.partial_average, pl2.data),
             (avg1.final_average, avg2.data),
             (avg2.final_average, pl3.data)
             ]

    # edges = [(exp.voltage, avg.data),
             # (avg.partial_average, pri.data)]
             # (avg.partial_average, pri.data)]
             # (avg.partial_average, pl1.data),]
             # (avg.final_average, pl2.data)]
    exp.set_graph(edges)

    exp.init_instruments()
    # exp.add_sweep(exp.field, np.linspace(0,100.0,11))
    exp.add_sweep(exp.freq, np.linspace(0,2,30))
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
