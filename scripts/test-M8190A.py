from pycontrol.instruments.keysight import M8190A
import numpy as np

def waveform(time, delay=1.5e-9, rise_time=150e-12, fall_time=2.0e-9):
    if time<=delay:
        return np.exp(-(time-delay)**2/(2*rise_time**2))
    if time>delay:
        return np.exp(-(time-delay)/fall_time)

if __name__ == '__main__':
    arb = M8190A("Test Arb", "128.33.89.22")
    print(arb.interface.query("*IDN?"))

    arb.set_output(True, channel=1)
    arb.set_output(False, channel=2)

    arb.sample_freq = 12.0e9
    arb.waveform_output_mode = "WSPEED"

    times = np.arange(0, 42.6e-9, 1/12e9)
    volts = [waveform(t, rise_time=0.10e-9, fall_time=1.0e-9) for t in times]

    sync_mkr = np.zeros(len(volts), dtype=np.int16)
    samp_mkr = np.zeros(len(volts), dtype=np.int16)
    samp_mkr[0:128] = 1
    sync_mkr[:128] = 1

    segment_id = 2
    wf = arb.create_binary_wf_data(np.array(volts), sync_mkr=sync_mkr, samp_mkr=samp_mkr)
    arb.use_waveform(wf, segment_id)
