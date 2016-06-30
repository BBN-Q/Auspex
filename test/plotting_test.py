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
    num_delays = 80

    def init_instruments(self):
        pass

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", range(self.samples)))
        descrip.add_axis(DataAxis("delay", list(range(self.num_delays))))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        time_vals = np.linspace(0, 1, self.num_delays)
        for tv in time_vals:
            await asyncio.sleep(0.1)
            data_row = np.sin(2*np.pi*self.freq.value*tv)*np.ones(self.samples)
            await self.voltage.push(data_row + 0.01*np.random.random(self.samples) )

        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

if __name__ == '__main__':

    exp = TestExperiment()
    avg1 = Average('samples', name="Collapse Samples")
    avg2 = Average('delay', name="Collapse Delay")
    pl1 = Plotter(name="Scope", color="firebrick", line_width=2)
    pl2 = Plotter(name="Sample Average", color="navy", line_width=2)
    # pl3 = Plotter(name="Sinusoid Final", color="firebrick", line_width=2)
    pl4 = Plotter(name="2D", plot_dims=2, palette="Spectral11")

    edges = [
             (exp.voltage, avg1.data),
             (exp.voltage, pl1.data),
             (avg1.final_average, avg2.data),
             (avg1.final_average, pl4.data),
             (avg1.final_average, pl2.data),
            #  (avg2.final_average, pl3.data)
             ]

    exp.set_graph(edges)

    exp.init_instruments()
    exp.add_sweep(exp.freq, np.linspace(1,2,2))
    exp.run_sweeps()
