# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS', 'APS2', 'DigitalAttenuator', 'SpectrumAnalyzer']

from .instrument import Instrument, SCPIInstrument, VisaInterface, MetaInstrument
from auspex.log import logger
import auspex.config as config
from types import MethodType
from unittest.mock import MagicMock
from time import sleep
from visa import VisaIOError
import numpy as np
from copy import deepcopy

# Dirty trick to avoid loading libraries when scraping
# This code using quince.
aps2_missing = False
if config.auspex_dummy_mode:
    fake_aps2 = True
    aps2 = MagicMock()
else:
    try:
        import aps2
        fake_aps2 = False
    except:
        fake_aps2 = True
        aps2_missing = True
        aps2 = MagicMock()

aps1_missing = False
if config.auspex_dummy_mode:
    fake_aps1 = True
    aps1 = MagicMock()
else:
    try:
        import APS as libaps
        fake_aps1 = False
    except:
        fake_aps1 = True
        aps1_missing = True
        aps1 = MagicMock()

class DigitalAttenuator(SCPIInstrument):
    """BBN 3 Channel Instrument"""
    instrument_type = "Digital attenuator"
    NUM_CHANNELS = 3
    instrument_type = 'Attenuator'

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
        self.interface._resource.timeout = 1000
        #Override query to look for ``end``
        def query(self, query_string):
            val = self._resource.query(query_string)
            assert self.read() == "END"
            return val
        self.interface.query = MethodType(query, self.interface)
        sleep(2) #!!! Why is the digital attenuator so slow?

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

class SpectrumAnalyzer(SCPIInstrument):
    """BBN USB Spectrum Analyzer"""
    instrument_type = "Spectrum analyzer"
    IF_FREQ = 0.0107e9 # 10.7 MHz IF notch filter

    def __init__(self, resource_name=None, *args, **kwargs):
        super(SpectrumAnalyzer, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type=None):
        super(SpectrumAnalyzer, self).connect(resource_name, interface_type)
        self.interface._resource.timeout = 100
        self.interface._resource.baud_rate = 115200
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\n"

    def get_voltage(self):
        volt = None
        for ct in range(10):
            try:
                volt = float(self.interface._resource.query("READ "))
            except (ValueError, VisaIOError):
                sleep(0.01)
            else:
                break
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

class APS(Instrument, metaclass=MakeSettersGetters):
    """BBN APSI or DACII"""

    instrument_type = "AWG"
    yaml_template = """
    APS-Name:
      type: APS               # Used by QGL and Auspex. QGL assumes XXXPattern for the pattern generator
      enabled: true            # true or false, optional
      master: true             # true or false
      slave_trig:              # name of marker below, optional, i.e. 12m4. Used by QGL.
      address:                 # APS serial number
      trigger_interval: 0.0    # (s)
      trigger_source: External # Internal or External
      seq_file: test.h5        # optional sequence file
      tx_channels:             # All transmit channels
        '12':                  # Quadrature channel name (string)
          phase_skew: 0.0      # (deg) - Used by QGL
          amp_factor: 1.0      # Used by QGL
          delay: 0.0           # (s) - Used by QGL
          '1':
            enabled: true
            offset: 0.0
            amplitude: 1.0
          '2':
            enabled: true
            offset: 0.0
            amplitude: 1.0
         '34':                  # Quadrature channel name (string)
           phase_skew: 0.0      # (deg) - Used by QGL
           amp_factor: 1.0      # Used by QGL
           delay: 0.0           # (s) - Used by QGL
           '1':
             enabled: true
             offset: 0.0
             amplitude: 1.0
           '2':
             enabled: true
             offset: 0.0
             amplitude: 1.0
      markers:
        1m1:
          delay: 0.0         # (s)
        2m1:
          delay: 0.0
        3m1:
          delay: 0.0
        4m1:
          delay: 0.0
                    """

    def __init__(self, resource_name=None, name="Unlabled APS"):
        self.name = name
        self.resource_name = resource_name

        if aps1_missing:
            logger.warning("Could not load aps1 library")

        if fake_aps2:
            self.wrapper = MagicMock()
        else:
            self.wrapper = libaps.APS()

        self.run           = self.wrapper.run
        self.stop          = self.wrapper.stop
        self.connected     = False

        self.read_register = self.wrapper.read_register

        self._sequence_filename = None
        self._run_mode = "RUN_SEQUENCE"
        self._repeat_mode = "TRIGGERED"

        self._run_mode_dict = {1: 'RUN_SEQUENCE', 0:'RUN_WAVEFORM'}
        self._run_mode_inv_dict = {v: k for k, v in self._run_mode_dict.items()}

        self._repeat_mode_dict = {1: "CONTINUOUS", 0: "TRIGGERED"}
        self._repeat_mode_inv_dict = {v: k for k, v in self._repeat_mode_dict.items()}

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to 'connect' if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name
        self.wrapper.connect(bytes(self.resource_name, 'ascii') if type(self.resource_name) == str else self.resource_name)
        self.connected = True

    def disconnect(self):
        if self.resource_name and self.connected:
            self.stop()
            self.wrapper.disconnect()
            self.connected = False

    def set_amplitude(self, chs, value):
        if isinstance(chs, int) or len(chs)==1:
            self.wrapper.set_amplitude(int(chs), value)
        else:
            self.wrapper.set_amplitude(int(chs[0]), value)
            self.wrapper.set_amplitude(int(chs[2]), value)
            self.wrapper.set_amplitude(int(chs[3]), value)
            self.wrapper.set_amplitude(int(chs[1]), value)

    def set_offset(self, chs, value):
        if isinstance(chs, int) or len(chs)==1:
            self.wrapper.set_amplitude(int(chs), value)
        else:
            self.wrapper.set_amplitude(int(chs[0]), value)
            self.wrapper.set_amplitude(int(chs[1]), value)
            self.wrapper.set_amplitude(int(chs[2]), value)
            self.wrapper.set_amplitude(int(chs[3]), value)

    def set_all(self, settings_dict, prefix=""):
        # Pop the channel settings
        settings = deepcopy(settings_dict)
        quad_channels = settings.pop('tx_channels')
        # Call the non-channel commands
        super(APS, self).set_all(settings)

        # Mandatory arguments
        for key in ['address', 'seq_file', 'trigger_interval', 'trigger_source', 'master']:
            if key not in settings.keys():
                raise ValueError("Instrument {} configuration lacks mandatory key {}".format(self, key))

        # Set the properties of individual hardware channels (offset, amplitude)
        for chan_group in ('12', '34'):
            quad_dict = quad_channels.pop(chan_group, None)
            if not quad_dict:
                raise ValueError("APS {} expected to receive quad channel '{}'".format(self, chan_group))
            for chan_num, chan_name in enumerate(list(chan_group)):
                chan_dict = quad_dict.pop(chan_name, None)
                if not chan_dict:
                    raise ValueError("Could not find channel {} in quadrature channel '{}' in settings for {}".format(chan_name, chan_group, self))
                for chan_attr, value in chan_dict.items():
                    try:
                        getattr(self, 'set_' + chan_attr)(chan_num, value)
                    except AttributeError:
                        pass

    def load_waveform(self, channel, data):
        if channel not in (1, 2, 3, 4):
            raise ValueError("Cannot load APS waveform {} on {} -- must be 1-4.".format(channel, self.name))
        try:
            self.wrapper.loadWaveform(channel, waveform)
        except AttributeError as ex:
            raise ValueError("Channel waveform data must be a numpy array.") from ex
        except NameError as ex:
            raise NameError("Channel data in incompatible type.") from ex

    def load_waveform_from_file(self, channel, data):
        if channel not in (1, 2, 3, 4):
            raise ValueError("Cannot load APS waveform {} on {} -- must be 1-4.".format(channel, self.name))
        #Warning: This is the one place in APS.py that does not subtract 1 from
        #the channel number. I am doing it here, but it should probably be fixed
        #in APS.py of libaps.
        self.wrapper.load_waveform_from_file(channel-1, filename)


    def trigger(self):
        raise NotImplementedError("Software trigger not present on APSI/DACII")

    @property
    def waveform_frequency(self):
        raise NotImplementedError("Hardware NCO not present on APSI/DACII")

    @property
    def mixer_correction_matrix(self):
        raise NotImplementedError("Harware mixer correction not present on APSI/DACII")

    @property
    def run_mode(self):
        return self._run_mode
    @run_mode.setter
    def run_mode(self, mode):
        mode = mode.upper()
        if mode not in self._run_mode_dict.values():
            raise ValueError("Unknown run mode {} for APS {}. Run mode must be one of {}.".format(mode, self.name, list(self._run_mode_dict.values())))
        else:
            self.wrapper.setRunMode(self._run_mode_inv_dict[mode])
            self._mode = mode

    @property
    def repeat_mode(self):
        return self._repeat_mode
    @repeat_mode.setter
    def repeat_mode(self, mode):
        mode = mode.upper()
        if mode not in self._repeat_mode_dict.values():
            raise ValueError("Unknown repeat mode {} for APS {}. Repeat mode must be one of {}.".format(mode, self.name, list(self._repeat_mode_dict.values())))
        else:
            self.wrapper.setRepeatMode(self._repeat_mode_inv_dict[mode])
            self._mode = mode

    @property
    def trigger_source(self):
        return self.wrapper.triggerSource
    @trigger_source.setter
    def trigger_source(self, source):
        source = source.lower()
        if source in ["internal", "external"]:
            self.wrapper.triggerSource = source
        else:
            raise ValueError("Invalid trigger source specification.")

    @property
    def trigger_interval(self):
        return self.wrapper.triggerInterval
    @trigger_interval.setter
    def trigger_interval(self, value):
        self.wrapper.triggerInterval = value

    @property
    def seq_file(self):
        return self._sequence_filename
    @seq_file.setter
    def seq_file(self, filename):
        self.wrapper.load_config(filename)
        self._sequence_filename = filename

    @property
    def sampling_rate(self):
        return self.wrapper.samplingRate
    @sampling_rate.setter
    def sampling_rate(self, value):
        self.wrapper.samplingRate = freq


class APS2(Instrument, metaclass=MakeSettersGetters):
    """BBN APS2"""
    instrument_type = "AWG"

    yaml_template = """
        APS2-Name:
          type: APS2               # Used by QGL and Auspex. QGL assumes XXXPattern for the pattern generator
          enabled: true            # true or false, optional
          master: true             # true or false
          slave_trig:              # name of marker below, optional, i.e. 12m4. Used by QGL.
          address:                 # IP address or hostname should be fine
          trigger_interval: 0.0    # (s)
          trigger_source: External # Internal, External, Software, or System
          seq_file: test.h5        # optional sequence file
          tx_channels:             # All transmit channels
            '12':                  # Quadrature channel name (string)
              phase_skew: 0.0      # (deg) - Used by QGL
              amp_factor: 1.0      # Used by QGL
              delay: 0.0           # (s) - Used by QGL
              '1':
                enabled: true
                offset: 0.0
                amplitude: 1.0
              '2':
                enabled: true
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

        if aps2_missing:
            logger.warning("Could not load aps2 library")

        if fake_aps2:
            self.wrapper = MagicMock()
        else:
            self.wrapper = APS.APS()

        self.set_enabled   = self.wrapper.set_channel_enabled
        self.set_mixer_phase_skew = self.wrapper.set_mixer_phase_skew
        self.set_mixer_amplitude_imbalance = self.wrapper.set_mixer_amplitude_imbalance

        self.get_amplitude = self.wrapper.get_channel_scale
        self.get_offset    = self.wrapper.get_channel_offset
        self.get_enabled   = self.wrapper.get_channel_enabled
        self.get_mixer_phase_skew = self.wrapper.get_mixer_phase_skew
        self.get_mixer_amplitude_imbalance = self.wrapper.get_mixer_amplitude_imbalance

        self.run           = self.wrapper.run
        self.stop          = self.wrapper.stop
        self.connected     = False

        self._sequence_filename = None
        self._mode = "RUN_SEQUENCE"

        if not fake_aps2:
            self._mode_dict = aps2.run_mode_dict
            self._mode_inv_dict = {v: k for k, v in aps2.run_mode_dict.items()}

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

    def set_amplitude(self, chs, value):
        if isinstance(chs, int) or len(chs)==1:
            self.wrapper.set_channel_scale(int(chs), value)
        else:
            self.wrapper.set_channel_scale(int(chs[0])-1, value)
            self.wrapper.set_channel_scale(int(chs[1])-1, value)

    def set_offset(self, chs, value):
        if isinstance(chs, int) or len(chs)==1:
            self.wrapper.set_channel_offset(int(chs), value)
        else:
            self.wrapper.set_channel_offset(int(chs[0])-1, value)
            self.wrapper.set_channel_offset(int(chs[1])-1, value)

    def set_all(self, settings_dict, prefix=""):
        # Pop the channel settings
        settings = deepcopy(settings_dict)
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
                try:
                    getattr(self, 'set_' + chan_attr)(chan_num, value)
                except AttributeError:
                    pass

    def load_waveform(self, channel, data):
        if channel not in (1, 2):
            raise ValueError("Cannot load APS waveform data to channel {} on {} -- must be 1 or 2.".format(channel, self.name))
        try:
            if data.dtype == np.int:
                self.wrapper.set_waveform_int(channel-1, data.astype(np.int16))
            elif data.dtype == np.float:
                self.wrapper.set_waveform_float(channel-1, data.astype(np.float32))
            else:
                raise ValueError("Channel waveform data must be either int or float. Unknown type {}.".format(data.dtype))
        except AttributeError as ex:
            raise ValueError("Channel waveform data must be a numpy array.") from ex

    def trigger(self):
        self.wrapper.trigger()

    @property
    def waveform_frequency(self):
        return self.wrapper.get_waveform_frequency()
    @waveform_frequency.setter
    def waveform_frequency(self, freq):
        self.wrapper.set_waveform_frequency(freq)

    @property
    def mixer_correction_matrix(self):
        return self.wrapper.get_mixer_correction_matrix()
    @mixer_correction_matrix.setter
    def mixer_correction_matrix(self, matrix):
        try:
            if matrix.shape != (2,2):
                raise ValueError("Mixer correction matrix must be 2 x 2. Got {} instead.".format(matrix.shape))
        except AttributeError as ex:
            raise ValueError("Mixer correction matrix must be a 2 x 2 numpy matrix. Got {} instead".format(matrix))
        self.wrapper.set_mixer_correction_matrix(matrix)

    @property
    def run_mode(self):
        return self._mode
    @run_mode.setter
    def run_mode(self, mode):
        mode = mode.upper()
        if mode not in self._mode_dict.values():
            raise ValueError("Unknown run mode {} for APS2 {}. Run mode must be one of {}.".format(mode, self.name, list(self._mode_dict.values())))
        else:
            self.wrapper.set_run_mode(self._mode_inv_dict[mode])
            self._mode = mode

    @property
    def seq_file(self):
        return self._sequence_filename
    @seq_file.setter
    def seq_file(self, filename):
        self.wrapper.load_sequence_file(filename)
        self._sequence_filename = filename

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

    @property
    def fpga_temperature(self):
        return self.wrapper.get_fpga_temperature()
