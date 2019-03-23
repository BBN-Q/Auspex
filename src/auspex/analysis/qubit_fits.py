# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from auspex.log import logger
from copy import copy
import matplotlib.pyplot as plt
from .fits import AuspexFit
from .signal_analysis import KT_estimation

"""
fit_rabi_amp
fit_rabi_width
fit_t1
fit_single_qubit_rb
fit_ramsey
"""

class RabiAmpFit(AuspexFit):

    xlabel = "Amplitude"
    ylabel = r"<$\sigma_z$>"
    title = "Rabi Amp Fit"

    @staticmethod
    def _model(x, *p):
        return p[0] - p[1]*np.cos(2*np.pi*p[2]*(x - p[3]))

    def _initial_guess(self):
        #seed Rabi frequency from largest FFT component
        N = len(self.ypts)
        yfft = np.fft.fft(self.ypts)
        f_max_ind = np.argmax(np.abs(yfft[1:N//2]))
        f_0 = 0.5 * max([1, f_max_ind]) / self.xpts[-1]
        amp_0 = 0.5*(self.ypts.max() - self.ypts.min())
        offset_0 = np.mean(self.ypts)
        phase_0 = 0
        if self.ypts[N//2 - 1] > offset_0:
            amp_0 = -amp_0
        return [offset_0, amp_0, f_0, phase_0]

    def _fit_dict(self, p):

        return {"y0": p[0], 
                "Api": p[1],
                "f": p[2],
                "phi": p[3]}

    def __str__(self):
        return "y0 - Api*cos(2*pi*f*(t - phi))"

    @property
    def pi_amp(self):
        return self.fit_params["Api"]

    def annotation(self):
        return r"$A_\pi$ = {0:.2e} {1} {2:.2e}".format(self.fit_params["Api"], chr(177), self.fit_errors["Api"])

class RabiWidthFit(AuspexFit):

    xlabel = "Delay"
    ylabel = r"<$\sigma_z$>"
    title = "Rabi Width Fit"

    @staticmethod
    def _model(x, *p):
        return p[0] + p[1]*np.exp(-x/p[2])*np.cos(2*np.pi*p[3]*(x - p[4]))

    def _initial_guess(self):
        frabi, Tcs, amps = KT_estimation(self.ypts-np.mean(self.ypts), self.xpts, 1)
        offset = np.average(self.xpts)
        amp = np.max(self.ypts)
        trabi = self.xpts[np.size(self.ypts) // 3]# assume Trabi is 1/3 of the scan
        phase = 90.0
        return [offset, amp, trabi, frabi[0], phase]

    def _fit_dict(self, p):

        return {"y0": p[0], 
                "A": p[1],
                'T': p[2],
                "f": p[3],
                "phi": p[4]}

    def __str__(self):
        return "y0 + A*exp(-x/T)*cos(2*pi*f*(t - phi))"

    @property
    def t_rabi(self):
        return self.fit_params["T"]

    def annotation(self):
        return r"$T_\pi$ = {0:.2e} {1} {2:.2e}".format(self.fit_params["T"], chr(177), self.fit_errors["T"])  

class T1Fit(AuspexFit):

    xlabel = "Delay"
    ylabel = r"<$\sigma_z$>"
    title = r"$T_1$ Fit"

    @staticmethod
    def _model(x, *p):
        return p[0]*np.exp(-x/p[1]) + p[2]

    def _initial_guess(self):
        amp = np.max(self.ypts)
        offset = self.ypts[-1]
        t1 = self.xpts[np.size(self.ypts) // 3]
        return [amp, t1, offset]

    def _fit_dict(self, p):
        return {"A": p[0], "T1": p[1], "A0": p[2]}

    def __str__(self):
        return "A0 + A*exp(-t/T1)"

    @property
    def T1(self):
        return self.fit_params["T1"]

    def annotation(self):
        return r"$T_1$ = {0:.2e} {1} {2:.2e}".format(self.fit_params["T1"], chr(177), self.fit_errors["T1"])  

class RamseyFit(AuspexFit):

    xlabel = "Delay"
    ylabel = r"<$\sigma_z$>"
    title = "Ramsey Fit"

    def __init__(self, xpts, ypts, two_freqs=False, AIC=True, make_plots=False, force=False):
        self.AIC = AIC
        self.two_freqs = two_freqs 
        self.force = force 
        self.plots = make_plots

        assert len(xpts) == len(ypts), "Length of X and Y points must match!"
        self.xpts = xpts 
        self.ypts = ypts 
        self._do_fit()


    def _initial_guess_1f(self):
        freqs, Tcs, amps = KT_estimation(self.ypts-np.mean(self.ypts), self.xpts, 1)
        return [freqs[0], abs(amps[0]), Tcs[0], np.angle(amps[0]), np.mean(self.ypts)]

    def _initial_guess_2f(self):
        freqs, Tcs, amps = KT_estimation(self.ypts-np.mean(self.ypts), xdata, 2)
        p0 = [*freqs, *abs(amps), *Tcs, *np.angle(amps), np.mean(self.ypts)]

    @staticmethod
    def _ramsey_1f(x, f, A, tau, phi, y0):
        return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

    @staticmethod
    def _model_2f(x, *p):
            return (RamseyFit._ramsey_1f(x, p[0], p[2], p[4], p[6], p[8]) + 
                    RamseyFit._ramsey_1f(x, p[1], p[3], p[5], p[7], p[9]))

    @staticmethod
    def _model_1f(x, *p):
            return RamseyFit._ramsey_1f(x, p[0], p[1], p[2], p[3], p[4])

    def _aicc(self, e, k, n):
        return 2*k+e+(k+1)*(k+1)/(n-k-2)

    def _do_fit(self):
        if self.two_freqs:
            self._initial_guess = self._initial_guess_2f
            self._model = self._model_2f
            try:
                super()._do_fit()
                if not self.AIC:
                    if self.plots:
                        self.make_plots()
                    return
                two_freq_chi2 = self.sq_error
            except:
                logger.info("Two-frequency fit failed. Trying single-frequency fit.")

        self._initial_guess = self._initial_guess_1f
        self._model = self._model_1f
        self.two_freqs = False
        super()._do_fit()
        one_freq_chi2 = self.sq_error


        if self.two_freqs and self.AIC:
            #Compare the one and two frequency fits
            aic = self._aicc(two_freq_chi2, 9, len(self.xpts)) - self._aicc(one_freq_chi2, 5, len(self.xpts))

            if aic > 0 and not self.force:
                self.two_freqs = False
                logger.info(f"Selecting one-frequency fit with AIC = {aic}")
                if self.plots:
                    self.make_plots()
            else:
                self.two_freqs = True
                self._initial_guess = self._initial_guess_2f
                self._model = self._model_2f
                super()._do_fit()
        else:
            if self.plots:
                self.make_plots()

    def annotation(self):
        #TODO: fixme
        return "RamseyFit"

    @property
    def T2(self):
        return self.fit_params["tau"]

    @property
    def ramsey_freq(self):
        return self.fit_params["f"]

    def _fit_dict(self, p):
        if self.two_freqs:
            return {"f": (p[0], p[1]),
                    "A": (p[2], p[3]), 
                    "tau": (p[4], p[5]),
                    "phi": (p[6], p[7]),
                    "y0": (p[8], p[9])}
        else:
            return {"f": p[0],
                    "A": p[1],
                    "tau": p[2],
                    "phi": p[3],
                    "y0": p[4]}

class SingleQubitRBFit(AuspexFit):

    xlabel = r"$log_2$ Clifford Number"
    ylabel = r"<$\sigma_z$>"
    title = "Single Qubit RB Fit"

    def __init__(self, lengths, data, make_plots=False):

        repeats = len(data) // len(lengths)
        xpts = np.repeat(lengths[:], repeats)
        ypts = np.mean(np.reshape(data,(len(lengths),repeats)),1)

        self.data = data
        self.lengths = lengths
        self.data_points = np.reshape(data,(len(lengths),repeats))
        self.errors = np.std(self.data_points, 1)

        super().__init__(xpts, ypts, make_plots=make_plots)

    @staticmethod
    def _model(x, *p):
        return p[0] * (1-p[1])**x + p[2]

    def _initial_guess(self):
        return [0.5, 0.01, 0.5]

    def __str__(self):
        return "A*(1 - r)^N + B"

    def _fit_dict(self, p):
        return {"A": p[0], "r": p[1]/2, "B": p[2]}

    def annotation(self):
        return r'avg. error rate r = {:.2e}  {} {:.2e}'.format(self.fit_params["r"]/2, chr(177), self.fit_errors["r"]/2)

    def make_plots(self):

        plt.figure()
        plt.plot(self.xpts, self.data,'.',markersize=15, label='data')
        plt.errorbar(self.lengths, self.ypts, yerr=self.errors/np.sqrt(len(self.lengths)),
                        fmt='*', elinewidth=2.0, capsize=4.0, label='mean')
        plt.plot(range(int(self.lengths[-1])), self.model(range(self.lengths[-1])), label='fit')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.annotate(self.annotation(), xy=(0.4, 0.10), 
                     xycoords='axes fraction', size=12)

class PhotonNumberFit(AuspexFit):
    ''' Fit number of measurement photons before a Ramsey. See McClure et al., Phys. Rev. App. 2016
    input params:
    1 - cavity decay rate kappa (MHz)
    2 - detuning Delta (MHz)
    3 - dispersive shift 2Chi (MHz)
    4 - Ramsey decay time T2* (us)
    5 - exp(-t_meas/T1) (us), only if starting from |1> (to include relaxation during the 1st msm't)
    6 - initial qubit state (0/1)
    '''
    def __init__(self, xpts, ypts, params, make_plots=False):
        self.params = params 

    def _initial_guess(self):
        return [0, 1]

    @staticmethod
    def _model(x, *p):
        pa = p[0]
        pb = p[1]
        params = self.params 

        if params[5] == 1:
            return params[4]*model_0(t, pa, pb) + (1-params[4])*model_0(t, pa+np.pi, pb)
        else:
            return (-np.imag(np.exp(-(1/params[3]+params[1]*1j)*t + (pa-pb*params[2]*(1-np.exp(-((params[0] + params[2]*1j)*t)))/(params[0]+params[2]*1j))*1j)))

    def _fit_dict(self, p):
        return {"Pa": p[0], "Pb": p[1]}

#### OLD STYLE FITS THAT NEED TO BE CONVERTED

def fit_drag(data, DRAG_vec, pulse_vec):
    """Fit calibration curves vs DRAG parameter, for variable number of pulses"""
    num_DRAG = len(DRAG_vec)
    num_seqs = len(pulse_vec)
    xopt_vec = np.zeros(num_seqs)
    perr_vec = np.zeros(num_seqs)
    popt_mat = np.zeros((3, num_seqs))
    data = data.reshape(len(data)//num_DRAG, num_DRAG)
    popt_mat[:, 0] = [1, DRAG_vec[data[0,:].argmin()], 0]
    for ct in range(len(pulse_vec)):
        #quadratic fit with increasingly narrower range
        data_n = data[ct, :]
        p0 = popt_mat[:, max(0,ct-1)]
        if ct > 0:
            #recenter for next fit
            closest_ind =np.argmin(abs(DRAG_vec - p0[1]))
            fit_range = int(np.round(0.5*num_DRAG*pulse_vec[0]/pulse_vec[ct]))
            curr_DRAG_vec = DRAG_vec[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
            reduced_data_n = data_n[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
        else:
            curr_DRAG_vec = DRAG_vec
            reduced_data_n = data_n
        #quadratic fit
        popt, pcov = curve_fit(quadf, curr_DRAG_vec, reduced_data_n, p0 = p0)
        perr_vec[ct] = np.sqrt(np.diag(pcov))[0]
        x_fine = np.linspace(min(curr_DRAG_vec), max(curr_DRAG_vec), 1001)
        xopt_vec[ct] = x_fine[np.argmin(quadf(x_fine, *popt))] #why not x0?
        popt_mat[:,ct] = popt
    return xopt_vec, perr_vec, popt_mat

def sinf(x, f, A, phi, y0):
    return A*np.sin(2*np.pi*f*x + phi) + y0

def quadf(x, A, x0, b):
    return A*(x-x0)**2+b

def fit_quad(xdata, ydata):
    popt, pcov = curve_fit(quadf, xdata, ydata, p0 = [1, min(ydata), 0])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

