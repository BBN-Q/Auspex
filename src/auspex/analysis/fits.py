from scipy.optimize import curve_fit
import numpy as np
from auspex.log import logger
from collections.abc import Iterable
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from .signal_analysis import *

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
    bounds = None
    ax = None

    def __init__(self, xpts, ypts, make_plots=False, ax=None):
        """Perform a least squares fit of 1-D data.

        Args:
            xpts (numpy.array): Independent fit variable data.
            ypts (numpy.array): Dependent fit variable data.
            make_plots (bool, optional): Generate a plot of the data and fit.
            ax (Axes, optional): Axes on which to draw plot. If None, new figure is created
        """


        assert len(xpts) == len(ypts), "Length of X and Y points must match!"
        self.xpts = xpts
        self.ypts = ypts
        self._do_fit(self.bounds)
        if make_plots:
            self.ax = ax
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
        if self.ax is None:
            plt.figure()
            plt.plot(self.xpts, self.ypts, ".", markersize=15, label="Data")
            plt.plot(self.xpts, self.model(self.xpts), "-", linewidth=3, label="Fit")
            plt.xlabel(self.xlabel, fontsize=14)
            plt.ylabel(self.ylabel, fontsize=14)
            plt.title(self.title, fontsize=14)
            plt.annotate(self.annotation(), xy=(0.4, 0.10),
                         xycoords='axes fraction', size=12)
        else:
            self.ax.plot(self.xpts, self.ypts, ".", markersize=15, label="Data")
            self.ax.plot(self.xpts, self.model(self.xpts), "-", linewidth=3, label="Fit")
            self.ax.set_xlabel(self.xlabel, fontsize=14)
            self.ax.set_ylabel(self.ylabel, fontsize=14)
            self.ax.set_title(self.title, fontsize=14)
            self.ax.annotate(self.annotation(), xy=(0.4, 0.10),
                         xycoords='axes fraction', size=12)
    def annotation(self):
        """Annotation for the `make_plot()` method. Should return a string
            that is passed to `matplotlib.pyplot.annotate`.
        """
        return str(self)

    def _do_fit(self, bounds=None):
        """Fit the data using `scipy.optimize.curve_fit`. This function will
            also compute the χ^2 and badness of fit of the function.

           Fit parameters and errors on those parameters are placed in dictionaries
           defined by the `_fit_dict` method. Also creates a `fit_function` method
           that is the model function evaluated at the fitted parameters.
         """
        p0 = self._initial_guess()
        if not bounds:
            bounds = (-np.inf, np.inf)
        else:
            assert all((len(b) == len(p0) for b in bounds)), 'Number of bounds must equal number of variables!'
        popt, pcov = curve_fit(self._model, self.xpts, self.ypts, p0, bounds=bounds)
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

class GaussianFit(AuspexFit):
    """A fit to a gaussian function"""

    xlabel = "X Data"
    ylabel = "Y Data"
    title  = "Gaussian Fit"

    @staticmethod
    def _model(x, *p):
        return p[0] + p[1]*np.exp(-0.5*((x-p[2])/p[3])**2)

    def _initial_guess(self):
        ## Initial guess using modified Caruana's algorithm
        ## See: H. Guo. "A Simple Algorithm for Fitting a Gaussian Function [DSP Tips and Tricks]"
        ###     IEEE Signal Processing Magazine. September 2011. DOI: 10.1109/MSP.2011.941846
        N = len(self.xpts)

        #use first and last points to 
        B = 0.5*(self.ypts[-1] + self.ypts[0])
        y = self.ypts - B
        mask = y>0
        y = y[mask]
        x = self.xpts[mask]

        M = np.array([[np.sum(y**2),      np.sum(x*y**2),    np.sum(x**2*y**2)],
                      [np.sum(x*y**2),    np.sum(x**2*y**2), np.sum(x**3*y**2)],
                      [np.sum(x**2*y**2), np.sum(x**3*y**2), np.sum(x**4*y**2)]])
        v = np.array([np.sum(y**2*np.log(y)),
                      np.sum(x*y**2*np.log(y)),
                      np.sum(x*y**2*np.log(y))])
        a, b, c = np.linalg.inv(M) @ v.T 

        mu = -b/(2.0*c)
        sigma = np.sqrt(-1/(2.0*c))
        A = np.exp(a - b**2/(4.0*c))

        return (B, A, mu, sigma)

    def _fit_dict(self, p):
        return {"B": p[0],
                "A": p[1],
                "μ": p[2],
                "σ": p[3]}

    def __str__(self):
        return "A exp(-(x-μ)^2/2σ^2) + B"

class MultiGaussianFit(AuspexFit):
    """A fit to a sum of gaussian function. Use with care!"""

    xlabel = "X Data"
    ylabel = "Y Data"
    title  = "Sum of Gaussians Fit"

    def __init__(self, x, y, make_plots=False, n_gaussians=2, n_samples=int(1e5)):
        """Fit data to a sum of Gaussians.

        Args:
            n_gaussians: Expected number of Gaussian peaks.
            n_samples: Number of random samples to generate for GMM estimation. (see `MultiGaussianFit._initial_guess`)
        """
        self.n_gaussians = n_gaussians 
        self.n_samples = n_samples
        super().__init__(x, y, make_plots=make_plots)

    @staticmethod
    def _model(x, *p):

        ngauss = int((len(p)-1)/3)

        assert ngauss > 1, "For a single Gaussian fit, use the `GaussianFit` class!"

        def one_gaussian(x, *p):
            return p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)

        out = p[0]
        for j in range(ngauss):
            out += one_gaussian(x, *p[3*j+1:3*j+4])
        return out

    def _initial_guess(self):
        ## Initial guess for the multi-gaussian fit
        ## The idea is to draw random samples using the data as a PDF, then run a 
        ## Gaussian mixture model (ie. clustering) to get a good initial guess for the gaussians. 
        ## Note that this is pretty slow...
        
        B = 0.5*(self.ypts[-1] + self.ypts[0])
        #normalize and center
        y = self.ypts - B
        mask = y>0
        x0 = self.xpts[np.argmax(y)]
        xn = self.xpts[mask] - x0
        yn = y[mask] / np.sum(y[mask])
        #Generate random samples
        samples = np.random.choice(a=xn, size=self.n_samples, p=yn)
        gmm = GaussianMixture(n_components=self.n_gaussians)
        gmm.fit(samples.reshape(len(samples),1))
        means = gmm.means_.flatten() + x0 
        sigmas = np.sqrt(gmm.covariances_.flatten())
        amps = gmm.weights_.flatten() * (2*np.pi)
        output = np.zeros(1+3*self.n_gaussians)
        output[0] = B
        for j in range(self.n_gaussians):
            output[3*j+1] = amps[j]
            output[3*j+2] = means[j]
            output[3*j+3] = sigmas[j]

        return output

    def _fit_dict(self, p):
        fdict = {"B": p[0]}
        for j in range(self.n_gaussians):
            fdict[f"A{j}"] = p[3*j+1]
            fdict[f"μ{j}"] = p[3*j+2]
            fdict[f"σ{j}"] = p[3*j+3]
        return fdict

    def __str__(self):
        return f"Sum of Gaussians with N={self.n_gaussians}"



