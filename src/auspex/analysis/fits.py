from scipy.optimize import curve_fit
import numpy as np

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
    return fopt, ferr

def ramsey_1f(x, f, A, tau, phi, y0):
    return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

def ramsey_2f(x, f1, f2, A1, A2, tau1, tau2, phi1, phi2, y0):
    return ramsey_1f(x, f1, A1, tau1, phi1) + ramsey_1f(x, f2, A2, tau2, phi2)

def fit_drag(data):
    pass
