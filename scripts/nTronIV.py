# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.keysight import *


from PyDAQmx import *

import numpy as np
import time
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd

if __name__ == '__main__':

    analog_input = Task()
    read = int32()
    # DAQmx Configure Code
    num_samples = int(1e5)
    sample_rate = int(2e4)
    frequency = 20.0
    samples_per_period = int(sample_rate/frequency)
    data = np.zeros((2*num_samples), dtype=numpy.float64)

    analog_input.CreateAIVoltageChan("Dev1/ai2", "", DAQmx_Val_RSE, -4.1, 4.1, DAQmx_Val_Volts, None)
    analog_input.CreateAIVoltageChan("Dev1/ai3", "", DAQmx_Val_RSE, -4, 4, DAQmx_Val_Volts, None)

    analog_input.CfgSampClkTiming("", sample_rate, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, num_samples)
    # analog_input.CfgInputBuffer(num_samples)
    analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
    # analog_input.SetStartTrigRetriggerable(0)

    # DAQmx Start Code
    analog_input.StartTask()
    print("Trigger me")


    analog_input.ReadAnalogF64(num_samples, -1, DAQmx_Val_GroupByChannel, data, 2*num_samples, byref(read), None)
    # try:
    analog_input.StopTask()
    analog_input.ClearTask()
    # except Exception as e:
     #    print("Warning failed to stop task.")
    #        pass

    volt_high = data[:len(data)/2]
    volt_low = data[len(data)/2:]

    volt_reshaped = np.reshape(volt_high, (samples_per_period,-1), order='F')
    volt_avg_high = np.mean(volt_reshaped, axis=1)
    volt_reshaped = np.reshape(volt_low, (samples_per_period,-1), order='F')
    volt_avg_low = np.mean(volt_reshaped, axis=1)
    volt_avg_low = volt_avg_low - np.mean(volt_avg_low)

    r_bias = 100e3
    gain = 20
    volt_avg_low = volt_avg_low / gain
    r = r_bias * volt_low / (volt_high - volt_low)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.plot(volt_avg_high, 'r-')
    plt.plot(volt_avg_low)
    plt.figure(2)
    plt.plot(1e6*(volt_avg_high-volt_avg_low)/r_bias, 1e3*volt_avg_low)
    plt.xlabel('I [$\mu$A]')
    plt.ylabel('V [mV]')
    plt.show()
