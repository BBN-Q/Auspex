# Copyright 2019 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from scipy.optimize import curve_fit
from auspex.log import logger
from copy import copy
import matplotlib.pyplot as plt
from .fits import AuspexFit
from .signal_analysis import KT_estimation

class RabiAmpFit(AuspexFit):
    """A fit to a Rabi amplitude curve, assuming a cosine model.
    """

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
        """Returns the pi-pulse amplitude of the fit.
        """
        return 0.5/self.fit_params["f"]

    def annotation(self):
        return r"$A_\pi$ = {0:.2e} {1} {2:.2e}".format(self.pi_amp, chr(177), self.fit_errors["Api"])

class RabiWidthFit(AuspexFit):
    """Fit to a single-frequency decaying cosine for fitting Rabi-vs-time experiments
    """
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
    """Fit to a decaying exponential for T1 measurement experiments.
    """
    xlabel = "Delay"
    ylabel = r"<$\sigma_z$>"
    title = r"$T_1$ Fit"

    @staticmethod
    def _model(x, *p):
        return p[0]*np.exp(-x/p[1]) + p[2]

    def _initial_guess(self):
        ## Initial guess using method of linear regression via integral equations
        ## https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
        N = len(self.xpts)
        S = np.zeros(N)
        for j in range(2, N):
            S[j] = S[j-1] + 0.5*((self.ypts[j] + self.ypts[j-1]) *
                                    (self.xpts[j] - self.xpts[j-1]))
        xs = self.xpts - self.xpts[0]
        ys = self.ypts - self.ypts[0]
        M = np.array([[np.sum(xs**2), np.sum(xs * S)], [np.sum(xs * S), np.sum(S**2)]])
        B1 = (np.linalg.inv(M) @ np.array([np.sum(ys * xs), np.sum(ys * S)]).T)[1]
        theta = np.exp(B1 * self.xpts)
        M2 = np.array([[N, np.sum(theta)], [np.sum(theta), np.sum(theta**2)]])
        A = np.linalg.inv(M2) @ np.array([np.sum(self.ypts), np.sum(self.ypts * theta)]).T

        return [A[1], -1.0/B1, A[0]]


    def _fit_dict(self, p):
        return {"A": p[0], "T1": p[1], "A0": p[2]}

    def __str__(self):
        return "A0 + A*exp(-t/T1)"

    @property
    def T1(self):
        """Return the measured T1 (i.e. decay constant of exponential).
        """
        return self.fit_params["T1"]


    def make_plots(self):
        """Create plot on both linear and semilog scale
        """
        logger.info("Semilog plot of |1> state probability requires calibrated data.")
        plt.figure(figsize=(2*6.4, 4.8))
        plt.subplot(121)
        plt.plot(self.xpts, self.ypts, ".", markersize=15, label="Data")
        plt.plot(self.xpts, self.model(self.xpts), "-", linewidth=3, label="Fit")
        plt.xlabel(self.xlabel, fontsize=14)
        plt.ylabel(self.ylabel, fontsize=14)
        plt.annotate(self.annotation(), xy=(0.4, 0.10), xycoords='axes fraction', size=12)
        plt.subplot(122)
        plt.semilogy(self.xpts, -1/2*(self.ypts - self.fit_params["A0"]), ".", markersize=15, label="Data")
        plt.semilogy(self.xpts, -1/2*(self.model(self.xpts) - self.fit_params["A0"]), "-", linewidth=3, label="Fit")
        plt.xlabel(self.xlabel, fontsize=14)
        plt.ylabel('|1> probability', fontsize=14)
        plt.suptitle(self.title, fontsize=14)

    def annotation(self):
        return r"$T_1$ = {0:.2e} {1} {2:.2e}".format(self.fit_params["T1"], chr(177), self.fit_errors["T1"])

class RamseyFit(AuspexFit):

    """Fit to a Ramsey experiment using either a one or two frequency decaying
        sine model.
    """

    xlabel = "Delay"
    ylabel = r"<$\sigma_z$>"
    title = "Ramsey Fit"

    def __init__(self, xpts, ypts, two_freqs=True, AIC=True, make_plots=False, force=False, ax=None):
        """One or two frequency Ramsey experiment fit. If a two-frequency fit is selected
            by the user or by comparing AIC scores, fit parameters are returned as tuples instead
            of single numbers.

        Args:
            xpts (numpy.array): Time data points.
            ypts (numpy.array): Qubit measurements.
            two_freqs (Bool): If true, attempt a two-frequency fit of the data.
            AIC (Bool): Decide between one and two frequency fits using  the Akaike
                information criterion.
            make_plots (Bool): Display a plot of data and fit result.
            ax (Axes, optional): Axes on which to draw plot. If None, new figure is created
            force (Bool): Force the selection of a two-frequency fit regardless of AIC score.
        """

        self.AIC = AIC
        self.dict_option = two_freqs
        self.two_freqs = two_freqs
        self.force = force
        self.plots = make_plots
        self.ax = ax

        assert len(xpts) == len(ypts), "Length of X and Y points must match!"
        self.xpts = xpts
        self.ypts = ypts
        self._do_fit()


    def _initial_guess_1f(self):
        freqs, Tcs, amps = KT_estimation(self.ypts-np.mean(self.ypts), self.xpts, 1)
        return [freqs[0], abs(amps[0]), Tcs[0], np.angle(amps[0]), np.mean(self.ypts)]

    def _initial_guess_2f(self):
        freqs, Tcs, amps = KT_estimation(self.ypts-np.mean(self.ypts), self.xpts, 2)
        return [*freqs, *abs(amps), *Tcs, *np.angle(amps), np.mean(self.ypts)]

    @staticmethod
    def _ramsey_1f(x, f, A, tau, phi, y0):
        return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

    @staticmethod
    def _model_2f(x, *p):
        return (RamseyFit._ramsey_1f(x, p[0], p[2], p[4], p[6], p[8]) + RamseyFit._ramsey_1f(x, p[1], p[3], p[5], p[7], p[8]))

    @staticmethod
    def _model_1f(x, *p):
        return RamseyFit._ramsey_1f(x, p[0], p[1], p[2], p[3], p[4])

    def _aicc(self, e, k, n):
        return 2*k+e+(2*k*(k+1))/(n-k-1)

    def _do_fit(self):
        if self.two_freqs:

            self.dict_option = True
            self._initial_guess = self._initial_guess_2f
            self._model = self._model_2f

            try:
                super()._do_fit()
                two_freq_chi2 = self.sq_error
            except:
                self.two_freqs = False
                logger.info("Two-frequency fit failed. Trying single-frequency fit.")

            if self.two_freqs and self.AIC:
                #Compare the one and two frequency fits
                self.dict_option = False
                self._initial_guess = self._initial_guess_1f
                self._model = self._model_1f
                super()._do_fit()
                one_freq_chi2 = self.sq_error

                aic = self._aicc(two_freq_chi2, 9, len(self.xpts)) - self._aicc(one_freq_chi2, 5, len(self.xpts))

                if aic > 0 and not self.force:
                    self.two_freqs = False
                    rl = 100*np.exp(-aic/2)
                    logger.info(f"Selecting one-frequency fit with relative likelihood = {rl:.2f}%")
                    if rl>33:
                        logger.info("Relative likelihood of 2nd frequency high, take more averages or set force = True.")

                else:
                    self.dict_option = True
                    self._initial_guess = self._initial_guess_2f
                    self._model = self._model_2f
                    super()._do_fit()

        if not self.two_freqs:
            self.dict_option = False
            self._initial_guess = self._initial_guess_1f
            self._model = self._model_1f
            super()._do_fit()

        if self.plots:
            self.make_plots()

    def annotation(self):
        if self.two_freqs:
            return r"$T_2$ = {0:.2e} {1} {2:.2e} "'\n'"$T_2$ = {3:.2e} {4} {5:.2e}".format(self.fit_params["tau1"], chr(177), self.fit_errors["tau1"], self.fit_params["tau2"], chr(177), self.fit_errors["tau2"])
        else:
            return r"$T_2$ = {0:.2e} {1} {2:.2e}".format(self.fit_params["tau"], chr(177), self.fit_errors["tau"])

    @property
    def T2(self):
        if self.two_freqs:
            return self.fit_params["tau1"], self.fit_params["tau2"]
        else:
            return self.fit_params["tau"]

    @property
    def ramsey_freq(self):
        if self.two_freqs:
            return self.fit_params["f1"], self.fit_params["f2"]
        else:
            return self.fit_params["f"]

    def _fit_dict(self, p):
        if self.dict_option:
            return {"f1": p[0],
                    "A1": p[2],
                    "tau1": p[4],
                    "phi1": p[6],
                    "f2": p[1],
                    "A2": p[3],
                    "tau2": p[5],
                    "phi2": p[7],
                    "y0": p[8]}
        else:
            return {"f": p[0],
                    "A": p[1],
                    "tau": p[2],
                    "phi": p[3],
                    "y0": p[4]}

class SingleQubitRBFit(AuspexFit):
    """Fit to an RB decay curve using the model A*(r^n) + B
    """


    ylabel = r"<$\sigma_z$>"
    title = "Single Qubit RB Fit"

    def __init__(self, lengths, data, make_plots=False, log_scale_x=True, smart_guess=True, bounded_fit=True, ax=None):

        self.lengths = sorted(list(set(lengths)))

        repeats = len(data) // len(self.lengths)
        xpts = np.array(self.lengths)
        ypts = np.mean(np.reshape(data,(len(self.lengths),repeats)),1)

        self.data = data
        self.data_points = np.reshape(data,(len(self.lengths),repeats))
        self.errors = np.std(self.data_points, 1)
        self.log_scale_x = log_scale_x
        self.ax = ax
        self.smart_guess = smart_guess

        if log_scale_x:
            self.xlabel = r"$log_2$ Clifford Number"
        else:
            self.xlabel = "Clifford Number"

        if bounded_fit:
            self.bounds = ((0, -np.inf, 0), (1, np.inf, 1))

        super().__init__(xpts, ypts, make_plots=make_plots, ax=ax)

    @staticmethod
    def _model(x, *p):
        return p[0] * (1-p[1])**x + p[2]

    def _initial_guess(self):
        if self.smart_guess:
            ## Initial guess using method of linear regression via integral equations
            ## https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
            N = len(self.xpts)
            S = np.zeros(N)
            for j in range(2, N):
                S[j] = S[j-1] + 0.5*((self.ypts[j] + self.ypts[j-1]) *
                                        (self.xpts[j] - self.xpts[j-1]))
            xs = self.xpts - self.xpts[0]
            ys = self.ypts - self.ypts[0]
            M = np.array([[np.sum(xs**2), np.sum(xs * S)],
                          [np.sum(xs * S), np.sum(S**2)]])
            B1 = (np.linalg.inv(M) @ np.array([np.sum(ys * xs), np.sum(ys * S)]).T)[1]
            theta = np.exp(B1 * self.xpts)
            M2 = np.array([[N, np.sum(theta)], [np.sum(theta), np.sum(theta**2)]])
            A = np.linalg.inv(M2) @ np.array([np.sum(self.ypts), np.sum(self.ypts * theta)]).T

            return [A[1], 1-np.exp(B1), A[0]]

        return [1, 0, 0.5]

    def __str__(self):
        return "A*(1 - r)^N + B"

    def _fit_dict(self, p):
        return {"A": p[0], "r": p[1]/2, "B": p[2]}

    def annotation(self):
        return r'avg. error rate r = {:.2e}  {} {:.2e}'.format(self.fit_params["r"], chr(177), self.fit_errors["r"])

    def make_plots(self):
        if self.ax is None:
            plt.figure()
            #plt.plot(self.xpts, self.data,'.',markersize=15, label='data')
            plt.errorbar(self.lengths, self.ypts, yerr=self.errors/np.sqrt(len(self.lengths)),
                            fmt='*', elinewidth=2.0, capsize=4.0, label='mean')
            plt.plot(range(int(self.lengths[-1])), self.model(range(int(self.lengths[-1]))), label='fit')
            if self.log_scale_x:
                plt.xscale('log')

            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.legend()
            plt.annotate(self.annotation(), xy=(0.4, 0.10),
                         xycoords='axes fraction', size=12)
        else:
            self.ax.errorbar(self.lengths, self.ypts, yerr=self.errors/np.sqrt(len(self.lengths)),
                            fmt='*', elinewidth=2.0, capsize=4.0, label='mean')
            self.ax.plot(range(int(self.lengths[-1])), self.model(range(int(self.lengths[-1]))), label='fit')
            if self.log_scale_x:
                self.ax.set_xscale('log')

            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
            self.ax.legend()
            self.ax.annotate(self.annotation(), xy=(0.4, 0.10),
                         xycoords='axes fraction', size=12)

class SingleQubitLeakageRBFit(SingleQubitRBFit):

    def __init__(self, lengths, data, make_plots=False, log_scale_x=True, bounded_fit=True, ax=None, leakage=True):

        # Compute populations from the tomography data
            a = data[-3]
            b = data[-2]
            c = data[-1]

            pop_mat = np.linalg.inv([[a,b,c],[b,a,c],[1,1,1]])

            points = []

            for i in range(len(data[:-3]) // 2):
                v = data[2*i]
                vp = data[2*i+1]
                points.append(np.matmul(pop_mat, np.array([v, vp, 1])))

            self.pop0, self.pop1, self.pop2 = zip(*points)

            pop_comp = [(self.pop0[i] + self.pop1[i]) if leakage else self.pop0[i] for i in range(len(self.pop0))]

            super().__init__(lengths[:-3][::2], pop_comp, make_plots, log_scale_x, bounded_fit, ax)

    def leakage(self):
        leak = (1 - self.fit_params['B'])*self.fit_params['r']
        err = leak*np.sqrt((self.fit_errors['B']/self.fit_params['B'])**2 + ((self.fit_errors['r']/self.fit_params['r'])**2))
        return leak, err

    def get_pops(self):
        repeats = len(self.pop0) // len(set(self.lengths))
        pop0 = np.mean(np.reshape(self.pop0,(len(self.lengths),repeats)),1)
        pop0_err = np.std(np.reshape(self.pop0,(len(self.lengths),repeats)),1)
        pop1 = np.mean(np.reshape(self.pop1,(len(self.lengths),repeats)),1)
        pop1_err = np.std(np.reshape(self.pop1,(len(self.lengths),repeats)),1)
        pop2 = np.mean(np.reshape(self.pop2,(len(self.lengths),repeats)),1)
        pop2_err = np.std(np.reshape(self.pop2,(len(self.lengths),repeats)),1)
        return pop0, pop0_err, pop1, pop1_err, pop2, pop2_err



class PhotonNumberFit(AuspexFit):
    """Fit number of measurement photons before a Ramsey. See McClure et al., Phys. Rev. App. 2016
    input params:
    1 - cavity decay rate kappa (MHz)
    2 - detuning Delta (MHz)
    3 - dispersive shift 2Chi (MHz)
    4 - Ramsey decay time T2* (us)
    5 - exp(-t_meas/T1) (us), only if starting from |1> (to include relaxation during the 1st msm't)
    6 - initial qubit state (0/1)
    """
    def __init__(self, xpts, ypts, T2, delta, kappa, chi, T1factor, init_state, make_plots=False):
        self.gamma2 = 1.0/T2
        self.delta = delta
        self.chi = chi
        self.kappa = kappa
        self.T1factor = T1factor
        self.init_state = init_state
        super().__init__(xpts, ypts, make_plots=make_plots)


    def _initial_guess(self):
        return [0, 1]

    def _model(self, x, *p):
        phi0 = p[0]
        n0 = p[1]

        q = self.kappa + 1j*self.chi
        tau = (1.0 - np.exp(-q*x))/q
        A0 = np.exp(-(self.gamma2 + self.delta*1j)*x + (phi0 - n0*self.chi*tau)*1j)
        B0 = np.exp(-(self.gamma2 + self.delta*1j)*x + (phi0 + np.pi - n0*self.chi*tau)*1j)
        A  = 0.5*(1 - np.imag(A0))
        B = 0.5*(1 - np.imag(B0))

        if self.init_state == 1:
            return 1 - (self.T1factor*A + (1-self.T1factor)*B)
        else:
            return A

    def _fit_dict(self, p):
        return {"phi0": p[0], "n0": p[1]}

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

def sinf(x, f=0, A=0, phi=0, y0=0):
    return A*np.sin(2*np.pi*f*x + phi) + y0

def quadf(x, A, x0, b):
    return A*(x-x0)**2+b

def fit_quad(xdata, ydata):
    popt, pcov = curve_fit(quadf, xdata, ydata, p0 = [1, min(ydata), 0])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
