# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.instrument import Instrument, SCPIInstrument, VisaInterface, MetaInstrument
from types import MethodType
from pycontrol.log import logger
# import aps2

class DigitalAttenuator(SCPIInstrument):
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
    @ch1_attenuation.setter
    def ch1_attenuation(self, value):
        self.set_attenuation(1, value)

    @property
    def ch2_attenuation(self):
        return self.get_attenuation(2)
    @ch2_attenuation.setter
    def ch2_attenuation(self, value):
        self.set_attenuation(2, value)

    @property
    def ch3_attenuation(self):
        return self.get_attenuation(3)
    @ch3_attenuation.setter
    def ch3_attenuation(self, value):
        self.set_attenuation(3, value)


class MakeSettersGetters(MetaInstrument):
    def __init__(self, name, bases, dct):
        super(MakeSettersGetters, self).__init__(name, bases, dct)

        for k,v in dct.items():
            if isinstance(v, property):
                logger.debug("Adding '%s' command to APS", k)
                setattr(self, 'set_'+k, v.fset)
                setattr(self, 'get_'+k, v.fget)

class APS2(Instrument, metaclass=MakeSettersGetters):
    """BBN APS2"""
    instrument_type = "AWG"

    def __init__(self, resource_name):
        self.name = "BBN APS2"
        self.resource_name = resource_name
        self._lib = aps2.APS2()
        self._lib.connect(resource_name)

        self.set_amplitude = self._lib.set_amplitude
        self.set_offset    = self._lib.set_offset
        self.set_enabled   = self._lib.set_enabled

    def __del__(self):
        self._lib.disconnect()

    def set_all(self, settings_dict):
        # Pop the channel settings
        ch1_settings = settings_dict.pop('chan_1')
        ch2_settings = settings_dict.pop('chan_2')

        # Call the non-channel commands
        super(APS2, self).set_all(settings_dict)

        for name, value in ch1_settings.items():
            if hasattr(self, name):
                getattr(self, name)(0, value)
        for name, value in ch2_settings.items():
            if hasattr(self, name):
                getattr(self, name)(1, value)

    @property
    def seq_file(self):
        return None
    @seq_file.setter
    def seq_file(self, filename):
        self._lib.load_sequence_file(filename)

    @property
    def trigger_source(self):
        return self._lib.get_trigger_source()
    @trigger_source.setter
    def trigger_source(self, source):
        if source in ["Internal", "External", "Software", "System"]:
            self._lib.set_trigger(getattr(aps2,source.upper()))
        else:
            raise ValueError("Invalid trigger source specification.")
            
    @property
    def trigger_interval(self):
        return self._lib.get_trigger_interval()
    @trigger_interval.setter
    def trigger_interval(self, value):
        self._lib.set_trigger_interval(value)

    @property
    def sampling_rate(self):
        return self._lib.get_sampling_rate()
    @sampling_rate.setter
    def sampling_rate(self, value):
        self._lib.set_sampling_rate(value)
