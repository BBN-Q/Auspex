from scipy.optimize import curve_fit
import numpy as np
from numpy.fft import fft
from scipy.linalg import svd, eig, inv, pinv
from enum import Enum
from auspex.log import logger
from collections.abc import Iterable
import matplotlib.pyplot as plt

from .signal_analysis import hilbert, KT_estimation

plt.style.use('ggplot')

class AuspexFit(object):

    xlabel = "X points"
    ylabel = "Y points"
    title = "Auspex Fit"

    def __init__(self, xpts, ypts, make_plots=False):
        assert len(xpts) == len(ypts), "Length of X and Y points must match!"
        self.xpts = xpts 
        self.ypts = ypts 
        self._do_fit()
        if make_plots:
            self.make_plots()

    def _initial_guess(self):
        raise NotImplementedError("Not implemented!")

    def _model(self, x, *p):
        raise NotImplementedError("Not implemented!")

    def _fit_dict(self, p):
        raise NotImplementedError("Not implemented!")

    def make_plots(self):
        plt.figure()
        plt.plot(self.xpts, self.ypts, ".", markersize=15, label="Data")
        plt.plot(self.xpts, self.model(self.xpts), "-", linewidth=3, label="Fit")
        plt.xlabel(self.xlabel, fontsize=14)
        plt.ylabel(self.ylabel, fontsize=14)
        plt.title(self.title, fontsize=14)
        plt.annotate(self.annotation(), xy=(0.4, 0.10), 
                     xycoords='axes fraction', size=12)

    def annotation(self):
        return str(self)

    def _do_fit(self):
        p0 = self._initial_guess()
        popt, pcov = curve_fit(self._model, self.xpts, self.ypts, p0)
        perr = np.sqrt(np.diag(pcov))
        fit = np.array([self._model(x, *popt) for x in self.xpts])
        self.sq_error = np.sum((fit - self.ypts)**2)
        dof = len(self.xpts) - len(p0)

        # Compute badness of fit:
        # Under the null hypothesis (that the model is valid and that the observations
        # do indeed have Gaussian statistics), the mean squared error is χ² distributed
        # with `dof` degrees of freedom. We can quantify badness-of-fit in terms of how
        # far the observed MSE is from the expected value, in units of σ = 2dof (the expected
        # standard deviation for the χ² distribution)
        self.Nsigma = self.sq_error/np.sqrt(2*dof) - dof/np.sqrt(2*dof) 
        self.fit_function = lambda x: self._model(x, *popt)

        self.fit_params = self._fit_dict(popt)
        self.fit_errors = self._fit_dict(perr)

    def model(self, x):
        if isinstance(x, Iterable): 
            return np.array([self.fit_function(_) for _ in x])
        else:
            return self.fit_function(x)

class LorentzFit(AuspexFit):

    xlabel = "X Data"
    ylabel = "Y Data"
    title = "Lorentzian Fit"

    def _model(self, x, *p):
        """Model for a simple Lorentzian"""
        return p[0]/((x-p[1])**2 + (p[2]/2)**2) + p[3]

    def _initial_guess(self):
        """Initial guess for a Lorentzian fit."""
        y0 = np.median(self.ypts)
        if np.abs(np.max(self.ypts) - y0) <= np.abs(np.min(self.ypts) - y0):
            idx = np.argmin(self.ypts)
            direc = -1
        else:
            idx = np.argmax(self.ypts)
            direc = 1
        f0 = self.xpts[idx]
        half = direc*np.abs(y0 + self.ypts[idx]) / 2.0
        if direc == -1:
            zeros = np.where((self.ypts-half)<0)[0]
        elif direc == 1:
            zeros = np.where((self.ypts-half)>0)[0]
        if len(zeros) >= 2:
            idx_l = zeros[0]
            idx_r = zeros[-1]
        else:
            idx_l = 0
            idx_r = len(self.xpts)-1
        width = np.abs(self.xpts[idx_l] - self.xpts[idx_r])
        amp = direc * width**2 * abs(self.ypts[idx] - y0) / 4
        return [amp, f0, width, y0]

    def _fit_dict(self, p):

        return {"A": p[0], 
                "b": p[1],
                "c": p[2],
                "d": p[3]}

    def __str__(self):
        return "A /((x-b)^2 + (c/2)^2) + d"

class CR_cal_type(Enum):
    LENGTH = 1
    PHASE = 2
    AMP = 3

def fit_CR(xpoints, data, cal_type):
    """Fit CR calibration curves for variable pulse length, phase, or amplitude"""
    data0 = data[:len(data)//2]
    data1 = data[len(data)//2:]
    if cal_type == CR_cal_type.LENGTH:
        xpoints = xpoints[0]
        x_fine = np.linspace(min(xpoints), max(xpoints), 1001)
        p0 = [1/(2*xpoints[-1]), 1, np.pi/2, 0]
        popt0, _ = curve_fit(sinf, xpoints, data0, p0 = p0)
        popt1, _ = curve_fit(sinf, xpoints, data1, p0 = p0)
        #find the first zero crossing
        yfit0 = sinf(x_fine[:int(1/abs(popt0[0])/2/(x_fine[1]-x_fine[0]))], *popt0)
        yfit1 = sinf(x_fine[:int(1/abs(popt1[0])/2/(x_fine[1]-x_fine[0]))], *popt1)
        #average between the two qc states, rounded to 10 ns
        xopt = round((x_fine[np.argmin(abs(yfit0))] + x_fine[np.argmin(abs(yfit1))])/2/10e-9)*10e-9
        logger.info('CR length = {} ns'.format(xopt*1e9))
    elif cal_type == CR_cal_type.PHASE:
        xpoints = xpoints[1]
        x_fine = np.linspace(min(xpoints), max(xpoints), 1001)
        p0 = [1/(xpoints[-1]), 1, np.pi, 0]
        popt0, _ = curve_fit(sinf, xpoints, data0, p0 = p0)
        popt1, _ = curve_fit(sinf, xpoints, data1, p0 = p0)
        #find the phase for maximum contrast
        contrast = (sinf(x_fine, *popt0) - sinf(x_fine, *popt1))/2
        logger.info('CR contrast = {}'.format(max(contrast)))
        xopt = x_fine[np.argmax(contrast)] - np.pi
    elif cal_type == CR_cal_type.AMP:
        xpoints = xpoints[2]
        x_fine = np.linspace(min(xpoints), max(xpoints), 1001)
        popt0 = np.polyfit(xpoints, data0, 1) # tentatively linearize
        popt1 = np.polyfit(xpoints, data1, 1)
        #average between optimum amplitudes
        xopt = -(popt0[1]/popt0[0] + popt1[1]/popt1[0])/2
        logger.info('CR amplitude = {}'.format(xopt))
    return xopt, popt0, popt1

