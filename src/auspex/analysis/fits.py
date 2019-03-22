from scipy.optimize import curve_fit
import numpy as np
from numpy.fft import fft
from scipy.linalg import svd, eig, inv, pinv
from enum import Enum
from auspex.log import logger
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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
    Q = np.linalg.lstsq(V_BB.conj().T, -V_AB.conj().T, rcond=-1)[0].conj().T

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

    amps = np.linalg.lstsq(A, cleanedData, rcond=-1)[0]

    return freqs, Tcs, amps

def fit_rabi_amp(xdata, ydata, showPlot=False):
    """
    Analyze Rabi amplitude data to find pi-pulse amplitude and phase offset.
        Arguments:
            xdata: ndarray of calibration amplitudes. length should be even.
            ydata: measurement amplitudes
        Returns:
            pi_amp: Fitted amplitude of pi pulsed
            offset: Fitted mixer offset
            fit_pts: Fitted points.
    """

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
    popt, _ = curve_fit(rabi_amp_model, xdata, ydata, \
        [offset_0, amp_0, f_0, phase_0])
    f_rabi = np.abs(popt[2])
    pi_amp = 0.5/f_rabi
    offset = popt[3]
    return pi_amp, offset, popt

def rabi_amp_model(x, *p):
    return p[0] - p[1]*np.cos(2*np.pi*p[2]*(x - p[3]))

def fit_rabi_width(xdata, ydata, showPlot=False):
    """
    Fit a simple Rabi oscillation experiment.

    Parameters
    ----------
    xdata : time points (array like)
    ydata : y-points (array like)
    showPlot : plot the result (boolean)

    Returns
    -------
    popt : fit parameters for the Rabi oscillation model \
            p0 + p1*np.exp(-x/p2)*np.cos(2*np.pi*p[3]*(x - p[4])) (array like)
    perr : sqrt of the popt covariance matrix diagonal  (array like)
    """

    frabi, Tcs, amps = KT_estimation(ydata-np.mean(ydata), xdata, 1)
    offset = np.average(xdata)
    amp = np.max(ydata)
    trabi = xdata[np.size(ydata) // 3]# assume Trabi is 1/3 of the scan
    phase = 90.0

    popt, pcov = curve_fit(rabi_width_model, xdata, ydata, \
                [offset, amp, trabi, frabi, phase])
    perr = np.sqrt(np.diag(pcov))

    trabi_fit = popt[2]
    trabi_fit_error = perr[2]

    if showPlot:
        xpts = np.linspace(xdata[0],xdata[-1],num=1000)

        plt.plot(xdata,ydata,'.',markersize=1.0, label='data')
        plt.plot(xpts, rabi_width_model(xpts, *popt), label='fit')
        plt.xlabel('time [ns]')
        plt.ylabel(r'<$\sigma_z$>')
        plt.legend()
        plt.annotate(r'$T_1$ = {0:.2e}  {1} {2:.2e} $\mu s$'.format( \
        popt[1]/1e3, chr(177), perr[1]/1e3), xy=(0.4, 0.10), \
                     xycoords='axes fraction', size=12)

    return trabi_fit, trabi_fit_error

def rabi_width_model(x, *p):
    return p[0] + p[1]*np.exp(-x/p[2])*np.cos(2*np.pi*p[3]*(x - p[4]))

def fit_t1(xdata, ydata, showPlot=False):
    """
    Fit simple single qubit T1.

    Parameters
    ----------
    xdata : time points (array like)
    ydata : scaled y-points (array like)
    showPlot : plot the result (boolean)

    Returns
    -------
    popt : fit parameters for the T1 model p0*np.exp(-x/p1) + p2 (array like)
    perr : sqrt of the popt covariance matrix diagonal  (array like)
    """

    amp = np.max(ydata)
    offset = ydata[-1]
    t1 = xdata[np.size(ydata) // 3]# assume T1 is 1/3 of the length of the scan

    popt, pcov = curve_fit(t1_model, xdata, ydata, [amp, t1, offset])
    perr = np.sqrt(np.diag(pcov))

    t1_fit = popt[1]
    t1_fit_error = perr[1]

    if showPlot:
        xpts = np.linspace(xdata[0],xdata[-1],num=1000)

        plt.plot(xdata,ydata,'.',markersize=1.0, label='data')
        plt.plot(xpts, t1_model(xpts, *popt), label='fit')
        plt.xlabel('time [ns]')
        plt.ylabel(r'<$\sigma_z$>')
        plt.legend()
        plt.annotate(r'$T_1$ = {0:.2e}  {1} {2:.2e} $\mu s$'.format( \
        popt[1]/1e3, chr(177), perr[1]/1e3), xy=(0.4, 0.10), \
                     xycoords='axes fraction', size=12)
        plt.show()

    print(r'T1 = {0:.2e}  {1} {2:.2e} us'.format(popt[1]/1e3, \
                    chr(177), perr[1]/1e3))
    return t1_fit, t1_fit_error

def t1_model(x, *p):
    return p[0]*np.exp(-x/p[1]) + p[2]

def fit_ramsey(xdata, ydata, two_freqs = False, AIC = True, showPlot=False, force=False):
    if two_freqs:
        # Initial KT estimation
        freqs, Tcs, amps = KT_estimation(ydata-np.mean(ydata), xdata, 2)
        p0 = [*freqs, *abs(amps), *Tcs, *np.angle(amps), np.mean(ydata)]
        try:
            popt2, pcov2 = curve_fit(ramsey_2f, xdata, ydata, p0 = p0, maxfev=5000)
            fopt2 = [popt2[0], popt2[1]]
            perr2 = np.sqrt(np.diag(pcov2))
            ferr2 = perr2[:2]
            fit_result_2 = (fopt2, ferr2, popt2, perr2)
            fit_model = ramsey_2f

            if not AIC:
                if showPlot:
                    plot_ramsey(xdata, ydata, popt2, perr2, fit_model=fit_model)
                print('Using a two-frequency fit.')
                print('T2 = {0:.3f} {1} {2:.3f} us'.format(popt2[4]*1e6, \
                    chr(177), perr2[4]*1e6))
                return fit_result_2
        except:
            fit_model = ramsey_1f
            logger.info('Two-frequency fit failed. Trying with single frequency.')
        # Initial KT estimation
    freqs, Tcs, amps = KT_estimation(ydata-np.mean(ydata), xdata, 1)
    p0 = [freqs[0], abs(amps[0]), Tcs[0], np.angle(amps[0]), np.mean(ydata)]
    popt, pcov = curve_fit(ramsey_1f, xdata, ydata, p0 = p0)
    fopt = [popt[0]]
    perr = np.sqrt(np.diag(pcov))
    fopt = [popt[0]]
    ferr = [perr[0]]
    fit_result_1 = (fopt, ferr, popt, perr)
    fit_model = ramsey_1f

    if two_freqs and AIC:
        def aicc(e, k, n):
            return 2*k+e+(k+1)*(k+1)/(n-k-2)
        def sq_error(xdata, popt, model):
            return sum((model(xdata, *popt) - ydata)**2)
        try:
            aic = aicc(sq_error(xdata, fit_result_2[2], ramsey_2f), 9, \
                len(xdata)) \
             - aicc(sq_error(xdata, fit_result_1[2], ramsey_1f), 5, len(xdata))
            if aic > 0 and not force:
                if showPlot:
                    plot_ramsey(xdata, ydata, popt, perr, fit_model=fit_model)
                print('Using a one-frequency fit.')
                print('T2 = {0:.3f} {1} {2:.3f} us'.format(popt[2]*1e6, \
                    chr(177), perr[2]*1e6))
                return fit_result_1
            else:
                fit_model = ramsey_2f
                if showPlot:
                    plot_ramsey(xdata, ydata, popt2, perr2, fit_model=fit_model)
                print('Using a two-frequency fit.')
                print('T2 = {0:.3f} {1} {2:.3f}us'.format( \
                    fit_result_2[2,2]/1e3, chr(177), fit_result_2[3,2]/1e3))
                return fit_result_2
        except:
            pass

    if not two_freqs and showPlot:
        plot_ramsey(xdata, ydata, popt, perr, fit_model=fit_model)

        print('Using a one-frequency fit.')
        print('T2 = {0:.3f} {1} {2:.3f} us'.format(popt[2]/1e3, chr(177), \
            perr[2]/1e3))

    if fit_model == ramsey_1f:
        return fit_result_1

def ramsey_1f(x, f, A, tau, phi, y0):
    return A*np.exp(-x/tau)*np.cos(2*np.pi*f*x + phi) + y0

def ramsey_2f(x, f1, f2, A1, A2, tau1, tau2, phi1, phi2, y0):
    return ramsey_1f(x, f1, A1, tau1, phi1, y0/2) + \
        ramsey_1f(x, f2, A2, tau2, phi2, y0/2)

def plot_ramsey(xdata, ydata, popt, perr, fit_model=ramsey_1f):
    xpts = np.linspace(xdata[0],xdata[-4],num=1000)

    plt.plot(xdata,ydata,'.',markersize=3.0, label='data')
    plt.plot(xpts, fit_model(xpts, *popt), label='fit')
    plt.xlabel('time [ns]')
    plt.ylabel(r'<$\sigma_z$>')
    plt.legend()
    if fit_model == ramsey_1f:
        plt.annotate(r'$T_2$ = {:.2e}  {} {:.2e} $\mu s$'.format( \
        popt[2]/1e3, chr(177), perr[2]/1e3), xy=(0.4, 0.15), \
                     xycoords='axes fraction', size=12)
        plt.annotate(r'$f_1$ = {:.2e}  {} {:.2e} MHz'.format( \
        popt[0]*1e3, chr(177), perr[0]*1e3), xy=(0.4, 0.05), \
                     xycoords='axes fraction', size=12)
    else:
        plt.annotate(r'$T^1_2$ = {0:.2e}  {1} {2:.2e} $\mu s$'.format( \
        popt[4]/1e3, chr(177), perr[4]/1e3), xy=(0.4, 0.35), \
                     xycoords='axes fraction', size=10)
        plt.annotate(r'$T^2_2$ = {0:.2e}  {1} {2:.2e} $\mu s$'.format( \
        popt[5]/1e3, chr(177), perr[5]/1e3), xy=(0.4, 0.25), \
                     xycoords='axes fraction', size=10)
        plt.annotate(r'$f_1$ = {0:.2e}  {1} {2:.2e} MHz'.format( \
        popt[0]*1e3, chr(177), perr[0]*1e3), xy=(0.4, 0.15), \
                     xycoords='axes fraction', size=10)
        plt.annotate(r'$f_2$ = {0:.2e}  {1} {2:.2e} MHz'.format( \
        popt[1]*1e3, chr(177), perr[1]*1e3), xy=(0.4, 0.05), \
                     xycoords='axes fraction', size=10)

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

def rb_model(x, *p):
    """Simple one qubit randomized benchmarking model"""
    return p[0] * (1-p[1])**x + p[2]

def fit_single_qubit_rb(data, lengths, showPlot=False):
    """
    Fit simple single qubit RB.  The average error rate will be printed

    Parameters
    ----------
    data : scaled RB data as <z> expectation values (array like)
    lengths : a list of the numbers of cliffords used
            (i.e. [4, 8, 16, ...]) (array like)
    showPlot : plot the result (boolean)

    Returns
    -------
    popt : fit parameters for the RB model p0*(1-p1)^n + p2 (array like)
    pcov : covariance matrix of popt (array like)
    """
    repeats = len(data)//len(lengths)
    xpts = np.repeat(lengths[:],repeats)

    data_points = np.reshape(data,(len(lengths),repeats))
    avg_points = np.mean(np.reshape(data,(len(lengths),repeats)),1)
    errors = np.std(data_points,1)

    fidelity = 0.5 * (1-data_points)
    avg_fidelity = 0.5 * (1-avg_points)

    popt, pcov = curve_fit(rb_model, lengths, avg_points, [0.5, 0.01, 0.5])
    perr = np.sqrt(np.diag(pcov))

    avg_infidelity = popt[1] / 2
    avg_infidelity_err = perr[1] / 2

    if showPlot:
        plt.plot(xpts,data,'.',markersize=0.7, label='data')
        plt.errorbar(lengths, avg_points, yerr=errors/np.sqrt(len(lengths)),\
        fmt='*', elinewidth=2.0, capsize=4.0, label='mean')
        plt.plot(range(lengths[-1]), rb_model(range(lengths[-1]), *popt), \
        label='fit')
        plt.xlabel('Clifford number')
        plt.ylabel(r'<$\sigma_z$>')
        plt.legend()
        plt.annotate(r'avg. error rate r = {:.2e}  {} {:.2e}'.format( \
        popt[1]/2, chr(177), perr[1]/2), xy=(0.05, 0.10), \
                     xycoords='axes fraction', size=12) # hack the pm symbol

    print(r'Average error rate: r = {:.2e} {} {:.2e}'.format( \
    avg_infidelity, chr(177), avg_infidelity_err))
    return avg_infidelity, avg_infidelity_err, popt, pcov

def cal_scale(data):
    """
    Scale the data assuming 4 cal points

    Parameters
    ----------
    data : unscaled data with cal points

    Returns
    -------
    data : scaled data array
    """
    # assume with have 2 cal repeats
    # TO-DO: make this general!!
    numRepeats = 2
    pi_cal = np.mean(data[-1*numRepeats:])
    zero_cal = np.mean(data[-2*numRepeats:-1*numRepeats])

    # negative to convert to <z>
    scale_factor = -(pi_cal - zero_cal) / 2
    data = data[:-2*numRepeats]
    data = (data - zero_cal)/scale_factor + 1

    return data

def cal_data(data, quad=np.real, qubit_name="q1", group_name="main", \
        return_type=np.float32, key=""):
    """
    Rescale data to :math:`\\sigma_z`. expectation value based on calibration sequences.

    Parameters:
        data (numpy array)       
            The data from the writer or buffer, which is a dictionary
            whose keys are typically in the format qubit_name-group_name, e.g.
            ({'q1-main'} : array([(0.0+0.0j, ...), (...), ...]))
        quad (numpy function)       
            This should be the quadrature where most of
            the data can be found.  Options are: np.real, np.imag, np.abs
            and np.angle
        qubit_name (string)  
            Name of the qubit in the data file. Default is 'q1'
        group_name (string)  
            Name of the data group to calibrate. Default is 'main'
        return_type (numpy data type)  
            Type of the returned data. Default is np.float32.
        key (string)         
            In the case where the dictionary keys don't conform
            to the default format a string can be passed in specifying the
            data key to scale.
    Returns:
        numpy array (type ``return_type``)
            Returns the data rescaled to match the calibration results for the :math:`\\sigma_z` expectation value.


    Examples:
        Loading and calibrating data 

        >>> exp = QubitExperiment(T1(q1),averages=500)
        >>> exp.run_sweeps()
        >>> data, desc = exp.writers[0].get_data()

    """
    if key:
        pass
    else:
        key = qubit_name + "-" + group_name

    fields = data[key].dtype.fields.keys()
    meta_field = [f for f in fields if 'metadata' in f][0]
    ind_axis = meta_field.replace("_metadata", "")

    ind0 = np.where(data[key][meta_field] == 0 )[0]
    ind1 = np.where(data[key][meta_field] == 1 )[0]

    dat = quad(data[key]["Data"])
    zero_cal = np.mean(dat[ind0])
    one_cal = np.mean(dat[ind1])

    scale_factor = -(one_cal - zero_cal)/2

    #assumes calibrations at the end only
    y_dat = dat[:-(len(ind0) + len(ind1))]
    x_dat = data[key][ind_axis][:-(len(ind0) + len(ind1))]
    y_dat = (y_dat - zero_cal)/scale_factor + 1
    return y_dat.astype(return_type), x_dat
