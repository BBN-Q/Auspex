# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import Instrument, VisaInterface
from types import MethodType


class DigitalAttenuator(Instrument):
    """BBN 3 Channel Instrument"""

    NUM_CHANNELS = 3

    def __init__(self, resource_name):
        super(DigitalAttenuator, self).__init__(resource_name, interface_type="VISA")
        self.name = "BBN Digital Attenuator"
        self.interface._resource.baud_rate = 115200
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\n"

        #Override query to look for ``end``
        def query(self, query_string):
            val = self._resource.query(query_string)
            assert self.read() == "END"
            return val

        self.interface.query = MethodType(query, self.interface, VisaInterface)

    @classmethod
    def channel_check(cls, chan):
        """ Assert the channel requested is feasbile """
        assert chan > 0 and chan <= cls.NUM_CHANNELS, "Invalid channel requested: channel ({:d}) must be between 1 and {:d}".format(chan, cls.NUM_CHANNELS)

    def get_attenuation(self, chan):
        DigitalAttenuator.channel_check(chan)
        return float(self.interface.query("GET {:d}".format(chan)))

    def set_attenuation(self, chan, val):
        DigitalAttenuator.channel_check(chan)
        self.interface.write("SET {:d} {:.1f}".format(chan, val))
        assert self.interface.read() == "Setting channel {:d} to {:.2f}".format(chan, val)
        assert self.interface.read() == "END"

    @property
    def ch1_attenuation(self):
        return self.get_attenuation(1)
    @ch1Attenuation.setter
    def ch1_attenuation(self, value):
        self.set_attenuation(1, value)

    @property
    def ch2_attenuation(self):
        return self.get_attenuation(2)
    @ch2Attenuation.setter
    def ch2_attenuation(self, value):
        self.set_attenuation(2, value)

    @property
    def ch3_attenuation(self):
        return self.get_attenuation(3)
    @ch3Attenuation.setter
    def ch3_attenuation(self, value):
        self.set_attenuation(3, value)