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
    return fopt, ferr, popt

def ramsey_1f(x, f, A, tau, phi, y0):
    return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

def ramsey_2f(x, f1, f2, A1, A2, tau1, tau2, phi1, phi2, y0):
    return ramsey_1f(x, f1, A1, tau1, phi1) + ramsey_1f(x, f2, A2, tau2, phi2)

def fit_drag(data, DRAG_vec, pulse_vec):
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
def fit_photon_number(xdata, ydata, params):
    ''' Fit number of measurement photons before a Ramsey. See McClure et al., Phys. Rev. App. 2016
    input params:
	1 - cavity decay rate kappa (MHz)
	2 - detuning Delta (MHz)
	3 - dispersive shift 2Chi (MHz)
	4 - Ramsey decay time T2* (us)
	5 - exp(-t_meas/T1) (us), only if starting from |1> (to include relaxation during the 1st msm't)
	6 - initial qubit state (0/1)
    '''
	params[:2]*=2*pi # convert to angular frequencies
	model_0(t, p) = (-imag(exp(-(1/params[3]+params[1]*1j).*t + (p[0]-p[1]*params[2]*(1-exp(-((params[0] + params[2]*1j).*t)))/(params[0]+params[2]*1j))*1j)))
	if params[5] == 1:
	       model(t, p) = params[4]*model_0(t, p) + (1-params[4])*model_0(t, [p[0]+np.pi; p[1:]]) if params[5] == 1  else model_0(t, p)
	popt, pcov = curve_fit(model, xdata, ydata, p0 = [0, 1])
    perr = np.sqrt(np.diag(pcov))
	return popt[1], perr[1]
