# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
'''
Test partial and final averager plots
'''
import time
import numpy as np

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataAxis, OutputConnector
from auspex.filters.plot import Plotter
from auspex.filters.average import Averager

from auspex.log import logger, logging
logger.setLevel(logging.INFO)

class TestInstrument(SCPIInstrument):
    '''A fake SCPII instrument with frequency property'''
    frequency = FloatCommand(get_string="frequency?", set_string="frequency \
                                {:g}", value_range=(0.1, 10))
    serial_number = IntCommand(get_string="serial?")
    mode = StringCommand(name="enumerated mode", scpi_string=":mode", \
                            allowed_values=["A", "B", "C"])

class TestExperiment(Experiment):
    """Here the run loop merely spews data until it fills up the stream. """

    # Create instances of instruments
    fake_instr_1 = TestInstrument("FAKE::RESOURE::NAME")

    # Parameters
    field = FloatParameter(unit="Oe")
    freq = FloatParameter(unit="Hz")

    # DataStreams
    voltage = OutputConnector(unit="V")

    def init_instruments(self):
        pass

    def init_streams(self):
        self.voltage.add_axis(DataAxis("xs", np.arange(100)))
        self.voltage.add_axis(DataAxis("ys", np.arange(100)))
        self.voltage.add_axis(DataAxis("repeats", np.arange(500)))

    def __repr__(self):
        return "<SweptTestExperiment>"

    def run(self):

        for _ in range(500):
            time.sleep(0.01)
            data = np.zeros((100, 100))
            data[25:75, 25:75] = 1.0
            data = data + 25*np.random.random((100, 100))
            self.voltage.push(data.flatten())

if __name__ == '__main__':

    EXP = TestExperiment()
    AVG = Averager("repeats", name="Averager")
    PL1 = Plotter(name="Partial", plot_dims=2, plot_mode="real", \
                    palette="Spectral11")
    PL2 = Plotter(name="Final", plot_dims=2, plot_mode="real", \
                    palette="Spectral11")

    EDGES = [
        (EXP.voltage, AVG.sink),
        (AVG.partial_average, PL1.sink),
        (AVG.final_average, PL2.sink)
        ]

    AVG.update_interval = 0.2
    PL1.update_interval = 0.2
    PL2.update_interval = 0.2

    EXP.set_graph(EDGES)
    EXP.init_instruments()
    EXP.run_sweeps()
