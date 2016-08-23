# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from pycontrol.instruments.picosecond import Picosecond10070A
from pycontrol.instruments.stanford import SR865
from pycontrol.instruments.keithley import Keithley2400
from pycontrol.instruments.ami import AMI430

from PyDAQmx import *

import numpy as np
import time
from tqdm import tqdm
from scipy.interpolate import interp1d
import pandas as pd

# Experimental Topology
# lockin AO 3 -> Analog Attenuator Vc (Control Voltages)
# Keithley Output -> Voltage divider with 1 MOhm, DAQmx AI1
# PSPL Trigger -> DAQmx PFI0

if __name__ == '__main__':
    pspl  = Picosecond10070A("GPIB0::24::INSTR")
    mag   = AMI430("192.168.5.109")
    keith = Keithley2400("GPIB0::25::INSTR")
    lock  = SR865("USB0::0xB506::0x2000::002638::INSTR")

    APtoP = False
    polarity = 1 if APtoP else -1

    keith.triad()
    keith.conf_meas_res(res_range=1e6)
    keith.conf_src_curr(comp_voltage=0.5, curr_range=1.0e-5)
    keith.current = 0e-6
    mag.ramp()

    # Setup picosecond
    pspl.duration  = 5e-9
    pspl_attenuation = 12
    pspl.amplitude = polarity*7.5*np.power(10, -pspl_attenuation/20)
    pspl.trigger_source = "GPIB"
    pspl.output = True
    pspl.trigger_level = 0.1

    # Ramp to the switching field
    mag.set_field(-0.0) # -130G

    # Variable attenuator
    df = pd.read_csv("calibration/RFSA2113SB.tsv", sep="\t")
    attenuator_interp = interp1d(df["Attenuation"], df["Control Voltage"])
    attenuator_lookup = lambda x : float(attenuator_interp(x))

    analog_input = Task()
    read = int32()

    # DAQmx Configure Code
    samps_per_trig = 5
    analog_input.CreateAIVoltageChan("Dev1/ai1", "", DAQmx_Val_Diff, 0, 1.0, DAQmx_Val_Volts, None)
    analog_input.CfgSampClkTiming("", 1e6, DAQmx_Val_Rising, DAQmx_Val_FiniteSamps, samps_per_trig)
    analog_input.CfgInputBuffer(samps_per_trig)
    analog_input.CfgDigEdgeStartTrig("/Dev1/PFI0", DAQmx_Val_Rising)
    analog_input.SetStartTrigRetriggerable(1)

    # DAQmx Start Code
    analog_input.StartTask()

    attens = np.arange(-12.01, -6, 1)
    durations = 1e-9*np.arange(0.1, 5.01, 1)

    def execute(pol=polarity, direction=1):
        id_dur = 0
        pspl.amplitude = pol*7.5*np.power(10, -pspl_attenuation/20)
        attenss = attens
        if direction==-1: # amplitude large to small
            attenss = np.flipud(attens)

        volts = 7.5*np.power(10, (-pspl_attenuation+attenss)/20)
        buffers = np.zeros((len(durations), len(attens), samps_per_trig))
        for dur in tqdm(durations, leave=True):
            id_atten = 0
            pspl.duration = dur
            time.sleep(0.1) # Allow the PSPL to settle
            for atten in tqdm(attenss, nested=True, leave=False):
                lock.ao3 = attenuator_lookup(atten)
                time.sleep(0.02) # Make sure attenuation is set
                # trigger out
                pspl.trigger()
                analog_input.ReadAnalogF64(samps_per_trig, -1, DAQmx_Val_GroupByChannel,
                                           buffers[id_dur, id_atten], samps_per_trig, byref(read), None)

                id_atten += 1
            id_dur += 1
        return pol*volts, buffers
    # Execute
    volts1, buffers1 = execute(-1,-1)
    volts2, buffers2 = execute(1, 1)
    volts3, buffers3 = execute(1,-1)
    volts4, buffers4 = execute(-1,1)

    try:
        analog_input.StopTask()
    except Exception as e:
        print("Warning failed to stop task.")
        pass

    # Shutting down
    keith.current = 0.0
    mag.zero()
    pspl.output = False

    # Do some polishment
    volts_tot = np.concatenate((volts1, volts2, volts3, volts4), axis=0)
    buffers_tot = np.concatenate((buffers1, buffers2, buffers3, buffers4), axis=1)
    buffers_mean = np.mean(buffers_tot, axis=2) # Average over samps_per_trig

    # Plot
    import matplotlib.pyplot as plt
    for i, dur in enumerate(durations):
        plt.figure(i)
        plt.plot(volts_tot, 1e-3*buffers_mean[i]/3e-6, '-o')
        plt.xlabel("Output V", size=14)
        plt.ylabel("R (kOhm)", size=14)
        plt.title("Duration = {0} ns".format(dur*1e+9))
    plt.show()
