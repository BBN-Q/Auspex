import numpy as np
import matplotlib.pyplot as plt
import auspex.config as config
config.auspex_dummy_mode = True
from auspex.filters import SingleShotMeasurement as SSM

def generate_fake_data(alpha, phi, sigma, N = 5000, plot=False):

    N_samples = 256
    data_start = 3
    data_length = 100
    gnd_mean = np.array([alpha*np.cos(phi), alpha*np.sin(phi)])
    ex_mean = np.array([alpha*np.cos(phi + np.pi), alpha*np.sin(phi + np.pi)])
    gndIQ = np.vectorize(complex)(np.random.normal(gnd_mean[0], sigma, N),
                                 np.random.normal(gnd_mean[1], sigma, N))
    exIQ = np.vectorize(complex)(np.random.normal(ex_mean[0], sigma, N),
                                 np.random.normal(ex_mean[1], sigma, N))
    gnd = np.zeros((N_samples, N), dtype=np.complex128)
    ex = np.zeros((N_samples, N), dtype=np.complex128)
    for idx, x in enumerate(zip(gndIQ, exIQ)):
        gnd[data_start:data_start+data_length, idx] = x[0]
        ex[data_start:data_start+data_length, idx] = x[1]

    gnd += sigma/50 * (np.random.randn(N_samples, N) + 1j * np.random.randn(N_samples, N))
    ex += sigma/50 * (np.random.randn(N_samples, N) + 1j * np.random.randn(N_samples, N))

    if plot:
        plt.figure()
        plt.plot(np.real(gndIQ), np.imag(gndIQ), 'b.')
        plt.plot(np.real(exIQ), np.imag(exIQ), 'r.')
        plt.draw()
        plt.show()

        plt.figure()
        plt.plot(np.real(gnd[:,15]), 'b.')
        plt.plot(np.real(ex[:,15]), 'r.')
        plt.draw()
        plt.show()
    return gnd, ex


if __name__ == "__main__":
    gnd, ex = generate_fake_data(3, np.pi/5, 1.6, plot=True)
    ss = SSM(save_kernel=False, optimal_integration_time=False, zero_mean=False,
                set_threshold=True, logistic_regression=True)
    ss.ground_data = gnd
    ss.excited_data = ex
    ss.compute_filter()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ss.pdf_data["I Bins"], ss.pdf_data["Ground I PDF"], "b-")
    plt.plot(ss.pdf_data["I Bins"], ss.pdf_data["Excited I PDF"], "r-")
    plt.plot(ss.pdf_data["I Bins"], ss.pdf_data["Ground I Gaussian PDF"], "b--")
    plt.plot(ss.pdf_data["I Bins"], ss.pdf_data["Excited I Gaussian PDF"], "r--")
    plt.ylabel("PDF")
    plt.subplot(2,1,2)
    plt.semilogy(ss.pdf_data["Q Bins"], ss.pdf_data["Ground Q PDF"], "b-")
    plt.semilogy(ss.pdf_data["Q Bins"], ss.pdf_data["Excited Q PDF"], "r-")
    plt.semilogy(ss.pdf_data["Q Bins"], ss.pdf_data["Ground Q Gaussian PDF"], "b--")
    plt.semilogy(ss.pdf_data["Q Bins"], ss.pdf_data["Excited Q Gaussian PDF"], "r--")
    plt.ylabel("PDF")
    plt.draw()
    plt.show()
