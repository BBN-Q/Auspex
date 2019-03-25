from scipy.optimize import curve_fit
import numpy as np
from auspex.log import logger
from collections.abc import Iterable
import matplotlib.pyplot as plt

from .signal_analysis import *
from .qubit_fits import *
from .resonator_fits import *

plt.style.use('ggplot')

class AuspexFit(object):
    """A generic fit class wrapping scipy.optimize.curve_fit for convenience.
        Specific fits should inherit this class.

    Attributes:
        xlabel (str): Plot x-axis label.
        ylabel (str): Plot y-axis label.
        title (str): Plot title.
    """


    xlabel = "X points"
    ylabel = "Y points"
    title = "Auspex Fit"

    def __init__(self, xpts, ypts, make_plots=False):
        """Perform a least squares fit of 1-D data.

        Args:
            xpts (numpy.array): Independent fit variable data.
            ypts (numpy.array): Dependent fit variable data.
            make_plots (bool, optional): Generate a plot of the data and fit.
        """


        assert len(xpts) == len(ypts), "Length of X and Y points must match!"
        self.xpts = xpts
        self.ypts = ypts
        self._do_fit()
        if make_plots:
            self.make_plots()

    def _initial_guess(self):
        """Return an initial guess for the fit parameters.
            Should be implemented in child class.
        """
        raise NotImplementedError("Not implemented!")

    @staticmethod
    def _model(x, *p):
        """Fit model function. Implemented as a static method for convenience.
            Should be implemented in child class.

        Args:
            x (numpy.array): Dependent variable for fit.
            *p (list): Fit parameters.
        """
        raise NotImplementedError("Not implemented!")

    def _fit_dict(self, p):
        """Return a dictionary of fit parameters.
            Should be implemented in child class.
        """
        raise NotImplementedError("Not implemented!")

    def make_plots(self):
        """Create a plot of the input data and the fitted model. By default will
            include any annotation defined in the `annotation()` class method.
        """
        plt.figure()
        plt.plot(self.xpts, self.ypts, ".", markersize=15, label="Data")
        plt.plot(self.xpts, self.model(self.xpts), "-", linewidth=3, label="Fit")
        plt.xlabel(self.xlabel, fontsize=14)
        plt.ylabel(self.ylabel, fontsize=14)
        plt.title(self.title, fontsize=14)
        plt.annotate(self.annotation(), xy=(0.4, 0.10),
                     xycoords='axes fraction', size=12)

    def annotation(self):
        """Annotation for the `make_plot()` method. Should return a string
            that is passed to `matplotlib.pyplot.annotate`.
        """
        return str(self)

    def _do_fit(self):
        """Fit the data using `scipy.optimize.curve_fit`. This function will
            also compute the χ^2 and badness of fit of the function.

           Fit parameters and errors on those parameters are placed in dictionaries
           defined by the `_fit_dict` method. Also creates a `fit_function` method
           that is the model function evaluated at the fitted parameters.
         """
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
        """ The fit function evaluated at the parameters found by `curve_fit`.

        Args:
            x: A number or `numpy.array` returned by the model function.
        """
        if isinstance(x, Iterable):
            return np.array([self.fit_function(_) for _ in x])
        else:
            return self.fit_function(x)

class LorentzFit(AuspexFit):
    """A fit to a simple Lorentzian function `A /((x-b)^2 + (c/2)^2) + d`
    """

    xlabel = "X Data"
    ylabel = "Y Data"
    title = "Lorentzian Fit"

    @staticmethod
    def _model(x, *p):
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
