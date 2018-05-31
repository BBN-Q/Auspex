# Copyright 2017 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from auspex.instruments import AlazarATS9870, AlazarChannel
from auspex.experiment import FloatParameter, IntParameter, Experiment
from auspex.stream import DataStream, DataAxis, DataStreamDescriptor, OutputConnector
from auspex.filters import Channelizer
from auspex.log import logger

import numpy as np
import asyncio
import time, sys, datetime

class nTronSwitchingExperiment(Experiment):

    # Constants (set with attribute access if you want to change these!)
    attempts           = 1 << 8
    samples            = 384 #1024 + 16*20
    measure_amplitude  = 0.1
    measure_duration   = 250.0e-9
    measure_frequency  = 100e6

    # arb
    sample_rate = 12e9
    repeat_time = 4*2.4e-6 # Picked very carefully for 100ns alignment

    # Things coming back
    voltage     = OutputConnector()

    alz = AlazarATS9870("1")

    def __init__(self):
        super(nTronSwitchingExperiment, self).__init__()

    def init_instruments(self):
        self.ch = AlazarChannel({'channel': 1})
        self.alz.add_channel(self.ch)
        alz_cfg = {
            'acquire_mode': 'digitizer',
            'bandwidth': 'Full',
            'clock_type': 'ref',
            'delay': 850e-9,
            'enabled': True,
            'label': "Alazar",
            'record_length': self.samples,
            'nbr_segments': 32,
            'nbr_waveforms': 1,
            'nbr_round_robins': self.attempts,
            'sampling_rate': 1e9,
            'trigger_coupling': 'DC',
            'trigger_level': 125,
            'trigger_slope': 'rising',
            'trigger_source': 'Ext',
            'vertical_coupling': 'AC',
            'vertical_offset': 0.0,
            'vertical_scale': 0.1,
        }
        self.alz.set_all(alz_cfg)
        self.loop.add_reader(self.alz.get_socket(self.ch), self.alz.receive_data, self.ch, self.voltage)

    def init_streams(self):
        # Baked in data axes
        descrip = DataStreamDescriptor()
        descrip.add_axis(DataAxis("time", 1e-9*np.arange(self.samples)))
        descrip.add_axis(DataAxis("gate_pulse_duration", np.arange(32)))
        descrip.add_axis(DataAxis("attempt", range(self.attempts)))

        self.voltage.set_descriptor(descrip)

    async def run(self):
        self.alz.acquire()
        await self.alz.wait_for_acquisition(5.0)
        self.alz.stop()
        # Seemingly we need to give the filters some time to catch up here...
        await asyncio.sleep(0.02)
        logger.info("Stream has filled {} of {} points".format(self.voltage.points_taken, self.voltage.num_points() ))

    def shutdown_instruments(self):
        print("Shutting down")
        self.alz.stop()
        self.loop.remove_reader(self.alz.get_socket(self.ch))
        self.alz.disconnect()

if __name__ == '__main__':
    exp = nTronSwitchingExperiment()
    demod = Channelizer(frequency=exp.measure_frequency, decimation_factor=4, bandwidth=20e6)
    edges = [(exp.voltage, demod.sink),]
    exp.set_graph(edges)
    exp.run_sweeps()

    exp = nTronSwitchingExperiment()
    demod = Channelizer(frequency=exp.measure_frequency, decimation_factor=4, bandwidth=20e6)
    edges = [(exp.voltage, demod.sink),]
    exp.set_graph(edges)
    exp.run_sweeps()
