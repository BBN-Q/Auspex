# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS2', 'DigitalAttenuator']

from .instrument import Instrument, SCPIInstrument, VisaInterface, MetaInstrument
from auspex.log import logger

from types import MethodType
from unittest.mock import MagicMock
import auspex.globals

# Dirty trick to avoid loading libraries when scraping
# This code using quince.
if auspex.globals.auspex_dummy_mode:
    fake_aps2 = True
else:
    try:
        import aps2
        fake_aps2 = False
    except:
        logger.warning("Could not find APS2 python driver.")
        fake_aps2 = True

class DigitalAttenuator(SCPIInstrument):
    """BBN 3 Channel Instrument"""

    NUM_CHANNELS = 3

    def __init__(self, resource_name=None, name='Unlabeled Digital Attenuator'):
        super(DigitalAttenuator, self).__init__(resource_name=resource_name,
            name=name)

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        super(DigitalAttenuator, self).connect(resource_name=self.resource_name,
            interface_type=interface_type)
        self.interface._resource.baud_rate = 115200
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\n"
        #Override query to look for ``end``
        def query(self, query_string):
            val = self._resource.query(query_string)
            assert self.read() == "END"
            return val
        self.interface.query = MethodType(query, self.interface)

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

class SpectrumAnaylzer(SCPIInstrument):
    """BBN USB Spectrum Analyzer"""

    IF_FREQ = 0.0107 # 10.7 MHz IF notch filter

    def __init__(self, resource_name=None, *args, **kwargs):
        super(SpectrumAnaylzer, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type=None):
        super(SpectrumAnaylzer, self).connect(resource_name, interface_type)
        self.interface._resource.timeout = 0.1
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\n"

    def get_voltage(self):
        volt = None
        for ct in range(10):
            try:
                volt = float(self.interface.query("READ "))
            except ValueError:
                pass
        if volt is None:
            logger.warning("Failed to get data from BBN Spectrum Analyzer "+
                " at {}.".format(self.resource_name))
        return volt

    @property
    def voltage(self):
        return self.get_voltage()

    def peak_amplitude(self):
        volt = self.get_voltage()
        if volt is None:
            return None
        else:
            interp = -100. + (volt - 75.) * (8/45)
            return interp



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

    yaml_template = """
        APS2-Name:
          type: APS2             # Used by QGL and Auspex. QGL assumes XXXPattern for the pattern generator
          enabled: true          # true or false, optional
          master: true           # true or false
          slave_trig:            # name of marker below, optional, i.e. 12m4. Used by QGL.
          address:               # IP address or hostname should be fine
          trigger_interval: 0.0  # (s)
          trigger: External      # Internal, External, Software, or System
          delay: 0.0
          seq_file: test.h5      # optional sequence file
          tx_channels:           # All transmit channels
            '12':                # Quadrature channel name (string)
              phase_skew: 0.0    # (deg) - Used by QGL
              amp_factor: 1.0    # Used by QGL
              '1':
                offset: 0.0
                amplitude: 1.0
              '2':
                offset: 0.0
                amplitude: 1.0
          markers:
            12m1:
              delay: 0.0         # (s)
            12m2:
              delay: 0.0
            12m3:
              delay: 0.0
            12m4:
              delay: 0.0
    """

    def __init__(self, resource_name=None, name="Unlabeled APS2"):
        self.name = name
        self.resource_name = resource_name

        if fake_aps2:
            self.wrapper = MagicMock()
        else:
            self.wrapper = aps2.APS2()

        self.set_amplitude = self.wrapper.set_channel_scale
        self.set_offset    = self.wrapper.set_channel_offset
        self.set_enabled   = self.wrapper.set_channel_enabled

        self.get_amplitude = self.wrapper.get_channel_scale
        self.get_offset    = self.wrapper.get_channel_offset
        self.get_enabled   = self.wrapper.get_channel_enabled

        self.run           = self.wrapper.run
        self.stop          = self.wrapper.stop
        self.connected     = False

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to 'connect' if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name

        self.wrapper.connect(self.resource_name)
        self.connected = True

    def disconnect(self):
        if self.resource_name and self.connected:
            self.stop()
            self.wrapper.disconnect()
            self.connected = False

    def set_all(self, settings_dict, prefix=""):
        # Pop the channel settings
        settings = settings_dict.copy()
        quad_channels = settings.pop('tx_channels')

        # Call the non-channel commands
        super(APS2, self).set_all(settings)

        # Mandatory arguments
        for key in ['address', 'seq_file', 'trigger_interval', 'trigger_source', 'master']:
            if key not in settings.keys():
                raise ValueError("Instrument {} configuration lacks mandatory key {}".format(self, key))

        # We expect a dictionary of channel names and their properties
        main_quad_dict = quad_channels.pop('12', None)
        if not main_quad_dict:
            raise ValueError("APS2 {} expected to receive quad channel '12'".format(self))

        # Set the properties of individual hardware channels (offset, amplitude)
        for chan_num, chan_name in enumerate(['1', '2']):
            chan_dict = main_quad_dict.pop(chan_name, None)
            if not chan_dict:
                raise ValueError("Could not find channel {} in quadrature channel 12 in settings for {}".format(chan_name, self))
            for chan_attr, value in chan_dict.items():
                if hasattr(self, 'set_' + chan_attr):
                    getattr(self, 'set_' + chan_attr)(chan_num, value)

    @property
    def seq_file(self):
        return None
    @seq_file.setter
    def seq_file(self, filename):
        self.wrapper.load_sequence_file(filename)

    @property
    def trigger_source(self):
        return self.wrapper.get_trigger_source()
    @trigger_source.setter
    def trigger_source(self, source):
        if source in ["Internal", "External", "Software", "System"]:
            self.wrapper.set_trigger_source(getattr(aps2,source.upper()))
        else:
            raise ValueError("Invalid trigger source specification.")

    @property
    def trigger_interval(self):
        return self.wrapper.get_trigger_interval()
    @trigger_interval.setter
    def trigger_interval(self, value):
        self.wrapper.set_trigger_interval(value)

    @property
    def sampling_rate(self):
        return self.wrapper.get_sampling_rate()
    @sampling_rate.setter
    def sampling_rate(self, value):
        self.wrapper.set_sampling_rate(value)
