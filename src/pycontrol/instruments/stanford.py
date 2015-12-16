from .instrument import Instrument, Command, FloatCommand, IntCommand
import numpy as np
import time

def indexed_map(values):
    return {v: '%d' % i for i, v in enumerate(values)}

def indexed_map_chan(values):
    return {v: '%d,0' % i for i, v in enumerate(values)}

class SR830(Instrument):
    """The SR830 lock-in amplifier."""
    SAMPLE_RATE_VALUES = [62.5e-3, 125e-3, 250e-3, 500e-3, 1, 2, 4, 8, 16,
                                32, 64, 128, 256, 512, "Trigger"]
    TIME_CONSTANT_VALUES = [10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 10e-3,
                            30e-3, 100e-3, 300e-3, 1, 3, 10, 3, 100, 300, 1e3,
                            3e3, 10e3, 30e3]
    SENSITIVITY_VALUES = [2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9,
                          500e-9, 1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6,
                          200e-6, 500e-6, 1e-3, 2e-3, 5e-3, 10e-3, 20e-3,
                          50e-3, 100e-3, 200e-3, 500e-3, 1]

    EXPANSION_VALUES = [0, 10, 100]
    FILTER_SLOPE_VALUES = [6, 12, 18, 24]
    RESERVE_VALUES = ['High Reserve', 'Normal', 'Low Noise']
    CHANNEL1_VALUES = ['X', 'R', 'X Noise', 'Aux In 1', 'Aux In 2']
    CHANNEL2_VALUES = ['Y', 'Theta', 'Y Noise', 'Aux In 3', 'Aux In 4']

    SAMPLE_RATE_MAP = indexed_map(SAMPLE_RATE_VALUES)
    TIME_CONSTANT_MAP = indexed_map(TIME_CONSTANT_VALUES)
    SENSITIVITY_MAP = indexed_map(SENSITIVITY_VALUES)
    EXPANSION_MAP = indexed_map(EXPANSION_VALUES)
    FILTER_SLOPE_MAP = indexed_map(FILTER_SLOPE_VALUES)
    RESERVE_MAP = indexed_map(RESERVE_VALUES)
    CHANNEL1_MAP = indexed_map_chan(CHANNEL1_VALUES)
    CHANNEL2_MAP = indexed_map_chan(CHANNEL2_VALUES)

    amplitude = FloatCommand("amplitude", scpi_string="SLVL")
    frequency = FloatCommand("frequency", scpi_string="FREQ", aliases=['freq'])
    phase = FloatCommand("phase", scpi_string="PHAS")

    x = FloatCommand("x", get_string="OUTP?1;")
    y = FloatCommand("y", get_string="OUTP?2;")
    channel_1 = FloatCommand("Channel 1", get_string="OUTR?1;", aliases=["ch1"])
    channel_2 = FloatCommand("Channel 2", get_string="OUTR?2;", aliases=["ch2"])

    magnitude = FloatCommand("magnitude", get_string="OUTP?3;", aliases=['r', 'mag'])
    theta = FloatCommand("theta", get_string="OUTP?4;")

    aux_in_1 = FloatCommand("Auxiliary Input 1", get_string="OAUX?1;", aliases=["ai1"])
    aux_in_2 = FloatCommand("Auxiliary Input 2", get_string="OAUX?2;", aliases=["ai2"])
    aux_in_3 = FloatCommand("Auxiliary Input 3", get_string="OAUX?3;", aliases=["ai3"])
    aux_in_4 = FloatCommand("Auxiliary Input 4", get_string="OAUX?4;", aliases=["ai4"])

    aux_out_1 = FloatCommand("Auxiliary Output 1", get_string="AUXV?1;", set_string="AUXV1,{:f}", aliases=["ao1"])
    aux_out_2 = FloatCommand("Auxiliary Output 2", get_string="AUXV?2;", set_string="AUXV2,{:f}", aliases=["ao2"])
    aux_out_3 = FloatCommand("Auxiliary Output 3", get_string="AUXV?3;", set_string="AUXV3,{:f}", aliases=["ao3"])
    aux_out_4 = FloatCommand("Auxiliary Output 4", get_string="AUXV?4;", set_string="AUXV4,{:f}", aliases=["ao4"])

    channel_1_type = Command("Channel 1", get_string="DDEF?1;", set_string="DDEF1,{:s}", value_map=CHANNEL1_MAP)
    channel_2_type = Command("Channel 2", get_string="DDEF?2;", set_string="DDEF2,{:s}", value_map=CHANNEL2_MAP)
    sensitivity    = Command("Sensitivity", get_string="SENS?;", set_string="SENS{:s}", value_map=SENSITIVITY_MAP)
    time_constant  = Command("Time Constant", get_string="OFLT?;", set_string="OFLT{:s}", value_map=TIME_CONSTANT_MAP, aliases=['tc', 'TC'])
    filter_slope   = Command("Filter Slope", get_string="OFSL?;", set_string="OFSL{:s}", value_map=FILTER_SLOPE_MAP)
    reserve_mode   = Command("Reserve Mode", get_string="RMOD?;", set_string="RMOD{:s}", value_map=RESERVE_MAP)

    sample_rate    = Command("Sample Rate", get_string="SRAT?;", set_string="SRAT{:s}", value_map=SAMPLE_RATE_MAP)
    buffer_mode    = Command("Buffer Mode", get_string="SEND?;", set_string="SEND{:s}", value_map={"SHOT": "0", "LOOP": "1"})
    buffer_trigger_mode = Command("Buffer Trigger Mode", get_string="TSTR?;", set_string="TSTR{:s}",
                                  value_map={True: "1", False: "0"})
    buffer_points  = IntCommand("Buffer Points", get_string="SPTS?;")

    def get_buffer(self, channel):
        stored_points = self.buffer_points
        self.interface.write("TRCB?{:d},0,{:d}".format(channel, stored_points))
        buf = self.interface.read_raw()
        return np.frombuffer(buf, dtype=np.float32)

    def buffer_start(self):
        self.interface.write("STRT;")
        # Inconsistent behavior missing first trigger/data point,
        # pause seems to address this issue.
        time.sleep(0.1)
    def buffer_pause(self):
        self.interface.write("PAUS;")
    def buffer_reset(self):
        self.interface.write("REST;")
    def trigger(self):
        self.interface.write("TRIG;")

    def __init__(self, name, resource_name, mode='current', **kwargs):
        super(SR830, self).__init__(name, resource_name, **kwargs)
        self.interface._resource.read_termination = u"\n"

    def measure_delay(self):
        """Return how long we must wait for the values to have settled, based on the filter slope."""
        fs = self.filter_slope
        tc = self.time_constant
        if fs <= 7: # 6dB/oct
            return 5*tc
        elif fs <= 13: # 12dB/oct
            return 7*tc
        elif fs <= 19: # 18dB/oct
            return 9*tc
        elif fs <= 25: # 24dB/oct
            return 10*tc
        else:
            raise Exception("Unknown delay for unknown filter slope {:f}".format(fs))

class SR865(Instrument):
    """The SR865 lock-in amplifier."""
    TIME_CONSTANT_VALUES = [1e-6, 3e-9, 10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 10e-3,
                            30e-3, 100e-3, 300e-3, 1, 3, 10, 3, 100, 300, 1e3,
                            3e3, 10e3, 30e3]
    SENSITIVITY_VALUES = [ 1, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3,
                           5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6,
                           5e-7, 2e-7, 1e-7, 5e-8, 2e-8, 1e-8, 5e-9, 2e-9, 1e-9]
    FILTER_SLOPE_VALUES = [6, 12, 18, 24]
    CHANNEL1_VALUES = ['X', 'R', 'X Noise', 'Aux In 1', 'Aux In 2']
    CHANNEL2_VALUES = ['Y', 'Theta', 'Y Noise', 'Aux In 3', 'Aux In 4']

    TIME_CONSTANT_MAP = indexed_map(TIME_CONSTANT_VALUES)
    SENSITIVITY_MAP = indexed_map(SENSITIVITY_VALUES)
    FILTER_SLOPE_MAP = indexed_map(FILTER_SLOPE_VALUES)
    CHANNEL1_MAP = indexed_map_chan(CHANNEL1_VALUES)
    CHANNEL2_MAP = indexed_map_chan(CHANNEL2_VALUES)

    amplitude = FloatCommand("amplitude", get_string="SLVL?", set_string="SLVL {:f}")
    frequency = FloatCommand("frequency", get_string="FREQ?", set_string="FREQ {:f}", aliases=['freq'])
    phase = FloatCommand("phase", get_string="PHAS?", set_string="PHAS {:g}")
    offset = FloatCommand("DC offset", get_string="SOFF?", set_string="SOFF {:f}", aliases=['dc', 'DC'])

    x = FloatCommand("x", get_string="OUTP? 0;", aliases=["ch1"])
    y = FloatCommand("y", get_string="OUTP? 1;", aliases=["ch2"])
    magnitude = FloatCommand("magnitude", get_string="OUTP? 2;", aliases=['r', 'mag'])
    theta = FloatCommand("theta", get_string="OUTP? 3;")

    channel_1_type = Command("Channel 1", get_string="DDEF?1;", set_string="DDEF1,{:s}", value_map=CHANNEL1_MAP)
    channel_2_type = Command("Channel 2", get_string="DDEF?2;", set_string="DDEF2,{:s}", value_map=CHANNEL2_MAP)
    # sample_frequency = Command("Sample Frequency", get_string="SRAT?;", set_string="SRAT{:s}", value_map=SAMPLE_RATE_MAP)
    sensitivity = Command("Sensitivity", get_string="SCAL?;", set_string="SCAL {:s}", value_map=SENSITIVITY_MAP)
    time_constant = Command("Time Constant", get_string="OFLT?;", set_string="OFLT{:s}", value_map=TIME_CONSTANT_MAP, aliases=['tc', 'TC'])
    filter_slope = Command("Filter Slope", get_string="OFSL?;", set_string="OFSL{:s}", value_map=FILTER_SLOPE_MAP)

    capture_quants = Command("Buffer quantities", scpi_string="CAPTURECFG", value_map={"X": "0", "XY": "1", "RT": "2", "XYRT": "3"})
    max_capture_rate = Command("Maximum capture rate", get_string="CAPTURERATEMAX?")
    capture_rate = IntCommand("Capture rate", scpi_string="CAPTURERATE")

    ao1 = FloatCommand("Analog output 1", set_string="AUXV 0, {:g};", get_string="AUXV? 0;")
    ao2 = FloatCommand("Analog output 2", set_string="AUXV 1, {:g};", get_string="AUXV? 1;")
    ao3 = FloatCommand("Analog output 3", set_string="AUXV 2, {:g};", get_string="AUXV? 2;")
    ao4 = FloatCommand("Analog output 4", set_string="AUXV 3, {:g};", get_string="AUXV? 3;")

    ai1 = FloatCommand("Analog output 1", get_string="OAUX? 0;")
    ai2 = FloatCommand("Analog output 2", get_string="OAUX? 1;")
    ai3 = FloatCommand("Analog output 3", get_string="OAUX? 2;")
    ai4 = FloatCommand("Analog output 4", get_string="OAUX? 3;")

    def __init__(self, name, resource_name, mode='current', **kwargs):
        super(SR865, self).__init__(name, resource_name, **kwargs)
        self.interface._resource.read_termination = u"\n"

    @property
    def capture_length(self):
        quants = self.capture_quants
        num_vars = len(quants) # Length of the string HAPPENS to correspond to number of quantities
        return int(int(self.interface.query("CAPTURELEN?"))*1024/(4*num_vars))
    @capture_length.setter
    def capture_length(self, num_points):
        quants = self.capture_quants
        num_vars = len(quants) # Length of the string HAPPENS to correspond to number of quantities
        kb = int(4*num_vars*num_points/1024)
        self.interface.write("CAPTURELEN {:d}".format(kb))

    @property
    def capture_rate(self):
        return float(self.interface.query("CAPTURERATE?"))
    @capture_rate.setter
    def capture_rate(self, value):
        allowed_values = 1.25e6/np.power(2, np.arange(0,21,1))
        if value not in allowed_values:
            raise ValueError("Capture rate must be the base clock 1.25 MHz / 2^n, where 0 <= n <= 20.")
        else:
            divs_of_base_rate = int(np.log2(1.25e6/value))
            self.interface.write("CAPTURERATE {:d}".format(divs_of_base_rate))

    def get_capture(self, channel):
        kb = int(self.interface.query("CAPTURELEN?"))
        return self.interface.query_binary_values("CAPTUREGET? {:d},{:d}".format(0, kb), datatype='f')

    def capture_done(self):
        bits = int(self.interface.query("CAPTURESTAT?"))
        return 0x1 & (bits >> 2)

    def capture_start(self, mode="ONE", hw_trigger=False):
        if mode not in ["ONE", "CONT"]:
            raise ValueError("mode must be either ONE or CONT")
        trig = "ON" if hw_trigger else "OFF"
        self.interface.write("CAPTURESTART {:s},{:s};".format(mode, trig))

    def capture_stop(self):
        self.interface.write("CAPTURESTOP;")
