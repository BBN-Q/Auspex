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
from pycontrol.filters.channelizer import Channelizer
from pycontrol.filters.integrator import KernelIntegrator

from pycontrol.logging import logger, logging
logger.setLevel(logging.INFO)

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
    num_samples     = 1024
    delays          = 1e-9*np.arange(100, 10001,100)
    sampling_period = 2e-9
    T2              = 5e-6

    def init_instruments(self):
        pass

    def init_streams(self):
        # Add a "base" data axis: say we are averaging 5 samples per trigger
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("samples", 2e-9*np.arange(self.num_samples)))
        descrip.add_axis(DataAxis("delay", self.delays))
        self.voltage.set_descriptor(descrip)

    def __repr__(self):
        return "<SweptTestExperiment>"

    async def run(self):
        pulse_start = 250
        pulse_width = 700

        #fake the response for a Ramsey frequency experiment with a gaussian excitation profile
        for delay in self.delays:
            await asyncio.sleep(0.05)
            record = np.zeros(self.num_samples)
            record[pulse_start:pulse_start+pulse_width] = np.exp(-0.5*(self.freq.value/2e6)**2) * \
                                                          np.exp(-delay/self.T2) * \
                                                          np.sin(2*np.pi * 10e6 * self.sampling_period*np.arange(pulse_width) \
                                                          + np.cos(2*np.pi * self.freq.value * delay))

            #add noise
            record += 0.1*np.random.randn(self.num_samples)

            await self.voltage.push(record)

        logger.debug("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

if __name__ == '__main__':

    exp = TestExperiment()
    channelizer = Channelizer(10e6, 32, name="Demod")
    ki = KernelIntegrator(np.ones(32), name="KI")
    pl1 = Plotter(name="2D Scope", plot_dims=2, palette="Spectral11")
    pl2 = Plotter(name="Demod", plot_dims=2, palette="Spectral11")
    pl3 = Plotter(name="KI", plot_dims=1)
    pl4 = Plotter(name="KI", plot_dims=2, palette="Spectral11")

    edges = [
            (exp.voltage, pl1.data),
            (exp.voltage, channelizer.sink),
            (channelizer.source, pl2.data),
            (channelizer.source, ki.sink),
            (ki.source, pl3.data),
            (ki.source, pl4.data)
            ]

    exp.set_graph(edges)

    exp.init_instruments()
    exp.add_sweep(exp.freq, 1e6*np.linspace(-4,4,41))
    exp.run_sweeps()
