# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['APS', 'APS2', 'TDM', 'DigitalAttenuator', 'SpectrumAnalyzer']

from .instrument import Instrument, SCPIInstrument, VisaInterface, MetaInstrument
from auspex.log import logger
import auspex.config as config
from types import MethodType
from unittest.mock import MagicMock
from time import sleep
from visa import VisaIOError
import numpy as np
from copy import deepcopy
import os.path

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
        import aps as libaps
        if libaps.APS_PY_WRAPPER_VERSION < 1.4:
            raise ImportError("Old version of libaps found. Please update.")
        fake_aps1 = False
    except:
        fake_aps1 = True
        aps1_missing = True
        libaps = MagicMock()

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

    def configure_with_proxy(self, proxy):
        for i, c in enumerate(proxy.channels):
            self.set_attenuation(i+1, c.attenuation)
        super(DigitalAttenuator, self).configure_with_proxy(proxy)

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
                try:
                    setattr(getattr(self, 'set_'+k), "__doc__", v.__doc__)
                    setattr(getattr(self, 'get_'+k), "__doc__", v.__doc__)
                except:
                    pass

class APS(Instrument, metaclass=MakeSettersGetters):
    """BBN APSI or DACII"""

    instrument_type = "AWG"

    def __init__(self, resource_name=None, name="Unlabled APS"):
        self.name = name
        self.resource_name = resource_name

        if aps1_missing:
            logger.warning("Could not load aps1 library!")

        if fake_aps1:
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
        self._sampling_rate = 1200

        self._run_mode_dict = {1: 'RUN_SEQUENCE', 0:'RUN_WAVEFORM'}
        self._run_mode_inv_dict = {v: k for k, v in self._run_mode_dict.items()}

        self._repeat_mode_dict = {1: "CONTINUOUS", 0: "TRIGGERED"}
        self._repeat_mode_inv_dict = {v: k for k, v in self._repeat_mode_dict.items()}

    def _initialize(self):
        if self.connected:
            self.wrapper.init(force=False)
            self.run_mode = self._run_mode
            self.repeat_mode = self._repeat_mode
            self.sampling_rate = self._sampling_rate
            for i in range(1,5):
                self.wrapper.set_enabled(i, True)
        else:
            raise IOError('Cannot initialize an unconnected APS!')

    def connect(self, resource_name=None):
        if resource_name is None and self.resource_name is None:
            raise Exception("Must supply a resource name to 'connect' if the instrument was initialized without one.")
        elif resource_name is not None:
            self.resource_name = resource_name
        self.wrapper.connect(self.resource_name)
        self.connected = True
        self._initialize()

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
            self.wrapper.set_amplitude(int(chs[1]), value)

    def set_offset(self, chs, value):
        if isinstance(chs, int) or len(chs)==1:
            self.wrapper.set_offset(int(chs), value)
        else:
            self.wrapper.set_offset(int(chs[0]), value)
            self.wrapper.set_offset(int(chs[1]), value)

    def configure_with_proxy(self, proxy_obj):
        super(APS, self).configure_with_proxy(proxy_obj)
        self.wrapper.set_offset(1, proxy_obj.ch("12").I_channel_offset)
        self.wrapper.set_offset(2, proxy_obj.ch("12").Q_channel_offset)
        self.wrapper.set_offset(3, proxy_obj.ch("34").I_channel_offset)
        self.wrapper.set_offset(4, proxy_obj.ch("34").Q_channel_offset)
        self.wrapper.set_amplitude(1, proxy_obj.ch("12").I_channel_amp_factor)
        self.wrapper.set_amplitude(2, proxy_obj.ch("12").Q_channel_amp_factor)
        self.wrapper.set_amplitude(3, proxy_obj.ch("34").I_channel_amp_factor)
        self.wrapper.set_amplitude(4, proxy_obj.ch("34").Q_channel_amp_factor)

    def load_waveform(self, channel, data):
        if channel not in (1, 2, 3, 4):
            raise ValueError("Cannot load APS waveform {} on {} -- must be 1-4.".format(channel, self.name))
        try:
            self.wrapper.load_waveform(channel, data)
        except AttributeError as ex:
            raise ValueError("Channel waveform data must be a numpy array.") from ex
        except NameError as ex:
            raise NameError("Channel data in incompatible type.") from ex

    def load_waveform_from_file(self, channel, filename):
        #NOT IN USE
        if channel not in (1, 2, 3, 4):
            raise ValueError("Cannot load APS waveform {} on {} -- must be 1-4.".format(channel, self.name))
        self.wrapper.load_waveform_from_file(channel-1, filename)


    def trigger(self):

        raise NotImplementedError("Software trigger not present on APSI/DACII")

    # utility functions for mixer calibration.
    def set_mixer_amplitude_imbalance(self, chs, amp):
        self.wrapper.set_amplitude(int(chs[0]), amp)

    def set_mixer_phase_skew(self, chs, phase, SSB = 0.0):
        qwf = -1 * np.sin(2*np.pi*SSB*np.arange(1200,dtype=np.float64)*1e-6/self.sampling_rate + phase)
        self.wrapper.load_waveform(int(chs[1]), qwf)

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
            for ch in (1,2,3,4):
                self.wrapper.set_run_mode(ch, self._run_mode_inv_dict[mode])
            self._run_mode = mode

    @property
    def repeat_mode(self):
        return self._repeat_mode
    @repeat_mode.setter
    def repeat_mode(self, mode):
        mode = mode.upper()
        if mode not in self._repeat_mode_dict.values():
            raise ValueError("Unknown repeat mode {} for APS {}. Repeat mode must be one of {}.".format(mode, self.name, list(self._repeat_mode_dict.values())))
        else:
            for ch in (1,2,3,4):
                self.wrapper.set_repeat_mode(ch, self._repeat_mode_inv_dict[mode])
            self._repeat_mode = mode

    @property
    def trigger_source(self):
        return self.wrapper.trigger_source
    @trigger_source.setter
    def trigger_source(self, source):
        source = source.lower()
        self.wrapper.trigger_source = source

    @property
    def trigger_interval(self):
        return self.wrapper.trigger_interval
    @trigger_interval.setter
    def trigger_interval(self, value):
        self.wrapper.trigger_interval = value

    @property
    def sequence_file(self):
        return self._sequence_filename
    @sequence_file.setter
    def sequence_file(self, filename):
        assert os.path.exists(filename), f"Sequence file {filename} for APS {self} does not exist."
        self.wrapper.load_config(filename)
        self._sequence_filename = filename

    @property
    def sampling_rate(self):
        return self.wrapper.sampling_rate
    @sampling_rate.setter
    def sampling_rate(self, freq):
        self.wrapper.sampling_rate = freq


class APS2(Instrument, metaclass=MakeSettersGetters):
    """BBN APS2"""
    instrument_type = "AWG"

    def __init__(self, resource_name=None, name="Unlabeled APS2"):
        self.name = name
        self.resource_name = resource_name

        if aps2_missing:
            logger.warning("Could not load aps2 library")

        if fake_aps2:
            self.wrapper = MagicMock()
        else:
            self.wrapper = aps2.APS2()

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

    def configure_with_proxy(self, proxy_obj):
        super(APS2, self).configure_with_proxy(proxy_obj)
        self.set_offset(0, proxy_obj[1].I_channel_offset)
        self.set_offset(1, proxy_obj[1].Q_channel_offset)
        self.set_amplitude(0, proxy_obj[1].I_channel_amp_factor)
        self.set_amplitude(1, proxy_obj[1].Q_channel_amp_factor)

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
    def sequence_file(self):
        return self._sequence_filename
    @sequence_file.setter
    def sequence_file(self, filename):
        if filename:
            self.wrapper.load_sequence_file(filename)
        self._sequence_filename = filename

    @property
    def trigger_source(self):
        return self.wrapper.get_trigger_source()
    @trigger_source.setter
    def trigger_source(self, source):
        if source.lower() in ["internal", "external", "software", "system"]:
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

    @property
    def amp_factor(self):
        return self.wrapper.get_mixer_amplitude_imbalance()
    @amp_factor.setter
    def amp_factor(self, amp):
        self.wrapper.set_mixer_amplitude_imbalance(amp)

    @property
    def phase_skew(self):
        return self.wrapper.get_mixer_phase_skew()
    @phase_skew.setter
    def phase_skew(self, skew):
        self.wrapper.set_mixer_phase_skew(skew)

class TDM(APS2):
    """BBN TDM"""
    instrument_type = "AWG"

    def configure_with_proxy(self, proxy_obj):
        super(APS2, self).configure_with_proxy(proxy_obj)
    #
    # def set_all(self, settings_dict):
    #     super(APS2, self).set_all(settings_dict)
    #     self.master = False # only for APS2. To make the TDM the master, set trigger_source: Internal for TDM and System for all the APS2
