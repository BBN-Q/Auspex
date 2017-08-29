from scipy.optimize import curve_fit
import numpy as np
from numpy.fft import fft
from enum import Enum

import matplotlib.pyplot as plt

def fit_rabi(xdata, ydata):
    """Analyze Rabi amplitude data to find pi-pulse amplitude and phase offset.
        Arguments:
            xdata: ndarray of calibration amplitudes. length should be even.
            ydata: measurement amplitudes
        Returns:
            pi_amp: Fitted amplitude of pi pulsed
            offset: Fitted mixer offset
            fit_pts: Fitted points."""

    def rabi_model(x, *p):
        return p[0] - p[1]*np.cos(2*np.pi*p[2]*(x - p[3]))
        
    #seed Rabi frequency from largest FFT component
    N = len(ydata)
    yfft = fft(ydata)
    f_max_ind = np.argmax(np.abs(yfft[1:N//2]))
    f_0 = 0.5 * max([1, f_max_ind]) / xdata[-1]
    amp_0 = 0.5*(ydata.max() - ydata.min())
    offset_0 = np.mean(ydata)
    phase_0 = 0
    if ydata[N//2 - 1] > offset_0:
        amp_0 = -amp_0
    popt, _ = curve_fit(rabi_model, xdata, ydata, [offset_0, amp_0, f_0, phase_0])
    f_rabi = np.abs(popt[2])
    pi_amp = 0.5/f_rabi
    offset = popt[3]
    return pi_amp, offset, rabi_model(xdata, *popt)

def fit_ramsey(xdata, ydata, two_freqs = False):
    #initial estimate
    #TODO: KT estimate
    if two_freqs:
        p0 = [1e5, 1e5, 0.5, 0.5, 10e-6, 10e-6, 0, 0, 0]
        popt, pcov = curve_fit(ramsey_2f, xdata, ydata, p0 = p0)
        fopt = [popt[0], popt[1]]
    else:
        p0 = [1e5, 1, 10e-6, 0, 0]
        popt, pcov = curve_fit(ramsey_1f, xdata, ydata, p0 = p0)
        fopt = [popt[0]]
    perr = np.sqrt(np.diag(pcov))
    fopt = popt[:two_freqs+1]
    ferr = perr[:two_freqs+1]
    return fopt, ferr, popt

def ramsey_1f(x, f, A, tau, phi, y0):
    return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

def ramsey_2f(x, f1, f2, A1, A2, tau1, tau2, phi1, phi2, y0):
    return ramsey_1f(x, f1, A1, tau1, phi1, y0) + ramsey_1f(x, f2, A2, tau2, phi2, y0)

def fit_drag(data, DRAG_vec, pulse_vec):
    """Fit calibration curves vs DRAG parameter, for variable number of pulses"""
    num_DRAG = len(DRAG_vec)
    num_seqs = len(pulse_vec)
    xopt_vec = np.zeros(num_seqs)
    data = norm_data(data).reshape(num_DRAG, len(data)/num_DRAG)
    #first fit sine to lowest n, for the full range
    data_n = data[:, 1]
    T0 = 2*(DRAG_vec[np.argmax(data_n)] - DRAG_vec[np.argmin(data_n)]) #rough estimate of period

    p0 = [0, 1, T0, 0]
    popt, pcov = curve_fit(sinf, DRAG_vec, data_n, p0 = p0)
    x_fine = np.linspace(min(DRAG_vec), max(DRAG_vec), 1001)
    xopt_vec[0] = x_fine[np.argmin(sinf(x_fine, *popt))]

    for ct in range(1, len(pulse_vec)):
        #quadratic fit for subsequent steps, narrower range
        data_n = data[:, ct]
        p0 = [1, xopt_vec[ct-1], 0]
        #recenter for next fit
        closest_ind =np.argmin(abs(DRAG_vec - x0))
        fit_range = np.round(0.5*num_DRAG*pulse_vec[0]/pulse_vec[ct])
        curr_DRAG_vec = DRAG_vec[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
        reduced_data_n = data_n[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
        #quadratic fit
        popt, pcov = curve_fit(quadf, curr_DRAG_vec, reduced_data_n, p0 = p0)
        perr = np.sqrt(np.diag(pcov))
        x_fine = np.linspace(min(curr_DRAG_vec), max(curr_DRAG_vec), 1001)
        x0 = x_fine[np.argmin(quadf(x_fine, *popt))]
    return xopt_vec[-1], perr[1]

def sinf(x, f, A, phi, y0):
    return A*np.sin(2*np.pi*f*x + phi) + y0

def quadf(x, A, x0, b):
    return A*(x-x0)**2+b

class CR_cal_type(Enum):
    LENGTH = 1
    PHASE = 2
    AMPLITUDE = 3

def fit_CR(xpoints, data, cal_type):
    """Fit CR calibration curves for variable pulse length, phase, or amplitude"""
    data0 = data[:len(data)/2]
    data1 = data[len(data)/2:]
    x_fine = np.linspace(min(xpoints), max(xpoints), 1001)
    if cal_type == CR_cal_type.LENGTH:
        p0 = [1, 2*xpoints[-1], -np.pi/2, 0]
        popt0, _ = curve_fit(sinf, xpoints, data0, p0 = p0)
        popt1, _ = curve_fit(sinf, xpoints, data1, p0 = p0)
        #find the first zero crossing
        yfit0 = sinf(x_fine[:round(abs(popt0[1])/2/(xpoints[1]-xpoints[0]))], *popt0)
        yfit1 = sinf(x_fine[:round(abs(popt1[1])/2/(xpoints[1]-xpoints[0]))], *popt1)
        #average between the two qc states, rounded to 10 ns
        xopt = round((x_fine[np.argmin(abs(yfit0))] + x_fine[np.argmin(abs(yfit1))])/2/10e-9)*10e-9
        print('CR length = %f ns'%xopt*1e9)
    elif cal_type == CR_cal_type.PHASE:
        p0 = [1, xpoints[-1], np.pi, 0]
        popt0, _ = curve_fit(sinf, x_fine, data0, p0 = p0)
        popt1, _ = curve_fit(sinf, x_fine, data1, p0 = p0)
        #find the phase for maximum contrast
        contrast = (sinf(x_fine, *popt0) - sinf(x_fine, *popt1))/2
        print('CR contrast = %f'%max(contrast))
        xopt = x_fine[np.argmax(contrast)] - np.pi
    elif cal_type == CR_cal_type.AMPLITUDE:
        popt0 = np.polyfit(xpoints, data0, 1)
        popt1 = np.polyfit(xpoints, data1, 1)
        yfit0 = popt0[0]*x_fine+popt0[1]
        yfit1 = popt1[0]*x_fine+popt1[1]
        #average between optimum amplitudes
        xopt = -(popt0[1]/popt0[0] + popt1[1]/popt1[0])/2
        print('CR amplitude = %f'%xopt)
    return xopt, popt0, popt1
