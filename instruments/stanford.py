from .instrument import instrument

def indexed_map(values):
	return {v: i for i, v in enumerate(values)}

class SR830(Instrument):
	"""The SR830 lock-in amplifier."""
	SAMPLE_FREQUENCY_VALUES = [62.5e-3, 125e-3, 250e-3, 500e-3, 1, 2, 4, 8, 16,
                                32, 64, 128, 256, 512]
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

    SAMPLE_FREQUENCY_MAP = indexed_map(SAMPLE_FREQUENCY_VALUES)
    TIME_CONSTANT_MAP = indexed_map(TIME_CONSTANT_VALUES)
    SENSITIVITY_MAP = indexed_map(SENSITIVITY_VALUES)
    EXPANSION_MAP = indexed_map(EXPANSION_VALUES)
	FILTER_SLOPE_MAP = indexed_map(FILTER_SLOPE_VALUES)
    RESERVE_MAP = indexed_map(RESERVE_MAP)
    CHANNEL1_MAP = indexed_map(CHANNEL1_MAP)
    CHANNEL2_MAP = indexed_map(CHANNEL2_MAP)

    amplitude = Control("amplitude", get_string="SLVL?", set_string="SLVL%0.3f")
    frequency = Control("frequency", get_string="FREQ?", set_string="FREQ%0.3e")
    phase = Control("phase", get_string="PHAS?", set_string="PHAS%0.2f")
    
    x = Control("x", get_string="OUTP?1")
    y = Control("y", get_string="OUTP?2")
    magnitude = Control("magnitude", get_string="OUTP?3")
    theta = Control("theta", get_string="OUTP?4")

    channel_1 = Control("Channel 1", get_string="DDEF?1;", set_string="DDEF1,%d,0", allowed_values=CHANNEL1_VALUES, value_map=CHANNEL1_MAP)
    channel_2 = Control("Channel 2", get_string="DDEF?2;", set_string="DDEF2,%d,0", allowed_values=CHANNEL2_VALUES, value_map=CHANNEL2_MAP)
    sample_frequency = Control("Sample Frequency", get_string="SRAT?;", set_string="SRAT%f", allowed_values=SAMPLE_FREQUENCY_VALUES, value_map=SAMPLE_FREQUENCY_MAP)
    sensitivity = Control("Sensitivity", get_string="SENS?;", set_string="SENS%f", allowed_values=SENSITIVITY_VALUES, value_map=SENSITIVITY_MAP)
    time_constant = Control("Time Constant", get_string="OFLT?;", set_string="OFLT%f", allowed_values=TIME_CONSTANT_VALUES, value_map=TIME_CONSTANT_MAP)
    filter_slope = Control("Filter Slope", get_string="OFSL?;", set_string="OFSL%f", allowed_values=FILTER_SLOPE_VALUES, value_map=FILTER_SLOPE_MAP)
    reserve_mode = Control("Reserve Mode", get_string="RMOD?;", set_string="RMOD%f", allowed_values=RESERVE_VALUES, value_map=RESERVE_MAP)
