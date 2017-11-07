from scipy.optimize import curve_fit
import numpy as np
from numpy.fft import fft
from scipy.linalg import svd, eig, inv, pinv
from enum import Enum
from auspex.log import logger


def hilbert(signal):
    # construct the Hilbert transform of the signal via the FFT
    # in essense, we just want to set negative frequency components to zero
    spectrum = np.fft.fft(signal)
    n = len(signal)
    midpoint = int(np.ceil(n/2))

    kernel = np.zeros(n)
    kernel[0] = 1
    if n%2 == 0:
        kernel[midpoint] = 1
    kernel[1:midpoint] = 2
    return np.fft.ifft(kernel * spectrum)

def KT_estimation(data, times, order):
    # Find the hilbert transform
    analytic_signal = hilbert(data)
    time_step = times[1]-times[0]

    # Create the Hankel matrix
    N = len(analytic_signal)
    K = order
    M = (N//2)-1
    L = N-M+1
    H = np.zeros((L, M), dtype=np.complex128)
    for ct in range(M):
        H[:,ct] = analytic_signal[ct:ct+L]

    #Try and seperate the signal and noise subspace via the svd
    U,S,V = svd(H, False) # V is not transposed/conjugated in numpy svd

    #Reconstruct the approximate Hankel matrix with the first K singular values
    #Here we can iterate and modify the singular values
    S_k = np.diag(S[:K])

    #Estimate the variance from the rest of the singular values
    varEst = (1/((M-K)*L)) * np.sum(S[K:]**2)
    Sfilt = np.matmul(S_k**2 - L*varEst*np.eye(K), inv(S_k))
    Hbar = np.matmul(np.matmul(U[:,:K], Sfilt), V[:K,:])

    #Reconstruct the data from the averaged anti-diagonals
    cleanedData = np.zeros(N, dtype=np.complex128)
    tmpMat = np.flip(Hbar,1)
    idx = -L+1
    for ct in range(N-1,-1,-1):
        cleanedData[ct] = np.mean(np.diag(tmpMat,idx))
        idx += 1

    #Create a cleaned Hankel matrix
    cleanedH = np.empty_like(H)
    cleanedAnalyticSig = hilbert(cleanedData)
    for ct in range(M):
        cleanedH[:,ct] = cleanedAnalyticSig[ct:ct+L]

    #Compute Q with total least squares
    #U_K1*Q = U_K2
    U = svd(cleanedH, False)[0]
    U_K = U[:,0:K]
    tmpMat = np.hstack((U_K[:-1,:],U_K[1:,:]))
    V = svd(tmpMat, False)[2].T.conj()
    n = np.size(U_K,1)
    V_AB = V[:n,n:]
    V_BB = V[n:,n:]
    Q = np.linalg.lstsq(V_BB.conj().T, -V_AB.conj().T)[0].conj().T

    #Now poles are eigenvalues of Q
    poles, _ = eig(Q)

    #Take the log and return the decay constant and frequency
    freqs = np.zeros(K)
    Tcs   = np.zeros(K)
    for ct in range(K):
        sk = np.log(poles[ct])
        freqs[ct] = np.imag(sk)/(2*np.pi*time_step)
        Tcs[ct] = -1.0/np.real(sk)*time_step

    #Refit the data to get the amplitude
    A = np.zeros((N, K), dtype=np.complex128)
    for ct in range(K):
        A[:,ct] = np.power(poles[ct], range(0,N))

    amps = np.linalg.lstsq(A, cleanedData)[0]

    return freqs, Tcs, amps

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
    if two_freqs:
        # Initial KT estimation
        freqs, Tcs, amps = KT_estimation(ydata, xdata, 2)
        p0 = [*freqs, *abs(amps), *Tcs, *np.angle(amps), np.mean(ydata)]
        popt, pcov = curve_fit(ramsey_2f, xdata, ydata, p0 = p0)
        fopt = [popt[0], popt[1]]
    else:
        # Initial KT estimation
        freqs, Tcs, amps = KT_estimation(ydata, xdata, 1)
        p0 = [freqs[0], abs(amps[0]), Tcs[0], np.angle(amps[0]), np.mean(ydata)]
        popt, pcov = curve_fit(ramsey_1f, xdata, ydata, p0 = p0)
        fopt = [popt[0]]
    perr = np.sqrt(np.diag(pcov))
    fopt = popt[:two_freqs+1]
    ferr = perr[:two_freqs+1]
    return fopt, ferr, popt

def ramsey_1f(x, f, A, tau, phi, y0):
    return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

def ramsey_2f(x, f1, f2, A1, A2, tau1, tau2, phi1, phi2, y0):
    return ramsey_1f(x, f1, A1, tau1, phi1, y0/2) + ramsey_1f(x, f2, A2, tau2, phi2, y0/2)

def fit_drag(data, DRAG_vec, pulse_vec):
    """Fit calibration curves vs DRAG parameter, for variable number of pulses"""
    num_DRAG = len(DRAG_vec)
    num_seqs = len(pulse_vec)
    xopt_vec = np.zeros(num_seqs)
    perr_vec = np.zeros(num_seqs)
    popt_mat = np.zeros((4, num_seqs))
    data = data.reshape(len(data)//num_DRAG, num_DRAG)
    #first fit sine to lowest n, for the full range
    data_n = data[1, :]
    T0 = 2*(DRAG_vec[np.argmax(data_n)] - DRAG_vec[np.argmin(data_n)]) #rough estimate of period

    p0 = [0, 1, T0, 0]
    popt, pcov = curve_fit(sinf, DRAG_vec, data_n, p0 = p0)
    perr_vec[0] = np.sqrt(np.diag(pcov))[0]
    x_fine = np.linspace(min(DRAG_vec), max(DRAG_vec), 1001)
    xopt_vec[0] = x_fine[np.argmin(sinf(x_fine, *popt))]
    popt_mat[:,0] = popt
    for ct in range(1, len(pulse_vec)):
        #quadratic fit for subsequent steps, narrower range
        data_n = data[ct, :]
        p0 = [1, xopt_vec[ct-1], 0]
        #recenter for next fit
        closest_ind =np.argmin(abs(DRAG_vec - xopt_vec[ct-1]))
        fit_range = int(np.round(0.5*num_DRAG*pulse_vec[0]/pulse_vec[ct]))
        curr_DRAG_vec = DRAG_vec[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
        reduced_data_n = data_n[max(0, closest_ind - fit_range) : min(num_DRAG-1, closest_ind + fit_range)]
        #quadratic fit
        popt, pcov = curve_fit(quadf, curr_DRAG_vec, reduced_data_n, p0 = p0)
        perr_vec[ct] = np.sqrt(np.diag(pcov))[0]
        x_fine = np.linspace(min(curr_DRAG_vec), max(curr_DRAG_vec), 1001)
        xopt_vec[ct] = x_fine[np.argmin(quadf(x_fine, *popt))] #why not x0?
        popt_mat[:3,ct] = popt
    return xopt_vec, perr_vec, popt_mat

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
    params = [2*np.pi*p for p in params[:3]] + params[3:] # convert to angular frequencies
    def model_0(t, pa, pb):
        return (-np.imag(np.exp(-(1/params[3]+params[1]*1j)*t + (pa-pb*params[2]*(1-np.exp(-((params[0] + params[2]*1j)*t)))/(params[0]+params[2]*1j))*1j)))
    def model(t, pa, pb):
        return  params[4]*model_0(t, pa, pb) + (1-params[4])*model_0(t, pa+np.pi, pb) if params[5] == 1  else model_0(t, pa, pb)
    popt, pcov = curve_fit(model, xdata, ydata, p0 = [0, 1])
    perr = np.sqrt(np.diag(pcov))
    finer_delays = np.linspace(np.min(xdata), np.max(xdata), 4*len(xdata))
    fit_curve = model(finer_delays, *popt)
    return popt[1], perr[1], (finer_delays, fit_curve)

def fit_quad(xdata, ydata):
    popt, pcov = curve_fit(quadf, xdata, ydata, p0 = [1, min(ydata), 0])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

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
