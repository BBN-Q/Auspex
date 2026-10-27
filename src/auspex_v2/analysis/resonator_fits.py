"""Single hanger S21 fits, replacing the experimental V1 circle-fit wrapper.

A circle initializes the fit; results come from unweighted least squares of the
original real and imaginary samples. This assumes constant complex gain and a
linear delay background, not arbitrary calibration, reflection or multiple modes.
"""

import warnings

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from .fits import _real_array

__all__ = ["ResonatorCircleFit", "ResonatorFitWarning", "circle_fit"]


class ResonatorFitWarning(UserWarning):
    """A physical fit exists, but sampling or local uncertainty needs inspection."""


def _circle_seed(data):
    """Solve the centered/scaled algebraic circle without V1's determinant root search."""
    origin = np.mean(data)
    scale = np.sqrt(np.mean(np.abs(data - origin)**2))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Circle data have no finite extent")
    z = (data - origin) / scale
    design = np.column_stack((2 * z.real, 2 * z.imag, np.ones(len(z))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, np.abs(z)**2, rcond=None)
    if rank < 3:
        raise ValueError("Collinear data do not determine a circle")
    x, y, intercept = coefficients
    radius_squared = intercept + x*x + y*y
    if radius_squared <= 0 or not np.isfinite(radius_squared):
        raise ValueError("No finite positive circle radius")
    return origin, scale, z, np.array([x, y, np.sqrt(radius_squared)])


def circle_fit(data):
    """Return (radius, center_real, center_imag) from finite complex circle samples.

    Centered algebraic least squares seeds geometric radial least squares. No
    frequency argument or parameter uncertainties are implied by this helper.
    A partial arc may still be very poorly conditioned despite a numerical fit.
    """
    data = np.asarray(data)
    if data.ndim != 1 or len(data) < 3 or data.dtype.kind not in "biufc" or not np.all(np.isfinite(data)):
        raise ValueError("Circle fitting needs at least three finite numeric samples")
    origin, scale, z, seed = _circle_seed(data)
    result = least_squares(lambda p: np.abs(z - (p[0] + 1j*p[1])) - p[2], seed,
                           bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]), max_nfev=1000)
    if not result.success:
        raise RuntimeError("Circle fit did not converge: " + result.message)
    center = origin + scale * (result.x[0] + 1j*result.x[1])
    return float(scale * result.x[2]), float(center.real), float(center.imag)


def _edge_delay(frequency, data):
    """Estimate delay from separate edge slopes, without bridging resonance phase wraps."""
    count = max(3, len(frequency) // 10)
    numerator = denominator = 0.0
    for indices in (slice(0, count), slice(-count, None)):
        x = frequency[indices]
        phase = np.unwrap(np.angle(data[indices]))
        x = x - np.mean(x)
        numerator += np.dot(x, phase - np.mean(phase))
        denominator += np.dot(x, x)
    return -numerator / denominator / (2 * np.pi)


def _initial_guess(frequency, data, reference):
    """Unwind a local delay, fit circle phase, then recover the off-resonant gain/depth.

    The circle vector rotates as theta - 2 atan(2*(f-fr)/linewidth). Its opposite
    point is the off-resonant gain. Half-phase-turn crossings seed linewidth without
    relying on a Fano-shifted magnitude minimum or differentiating noisy samples.
    """
    span = np.ptp(frequency)
    u = (frequency - reference) / span
    delay = _edge_delay(frequency, data)

    def radial_error(cycles):
        corrected = data * np.exp(2j * np.pi * u * (delay * span + cycles))
        try:
            _, scale, z, seed = _circle_seed(corrected)
            return float(np.mean((np.abs(z - (seed[0] + 1j*seed[1])) - seed[2])**2) * scale**2)
        except ValueError:
            return np.inf

    # Radial error is not unimodal: a large background arc can be another local
    # minimum, especially for weak coupling. Bracket the best sampled basin first.
    grid = np.linspace(-0.5, 0.5, 129)
    best = int(np.argmin([radial_error(cycles) for cycles in grid]))
    bracket = (grid[max(0, best-1)], grid[min(len(grid)-1, best+1)])
    correction = minimize_scalar(radial_error, bounds=bracket, method="bounded",
                                 options={"xatol": 1e-10})
    if not correction.success or not np.isfinite(correction.fun):
        raise ValueError("Cannot initialize cable delay; supply p0")
    delay += correction.x / span
    corrected = data * np.exp(2j * np.pi * (frequency - reference) * delay)
    radius, cx, cy = circle_fit(corrected)
    center = cx + 1j*cy
    phase = np.unwrap(np.angle(corrected - center))
    midpoint = (phase[0] + phase[-1]) / 2
    index = int(np.argmin(np.abs(phase - midpoint)))
    left = int(np.argmin(np.abs(phase - (midpoint + np.pi/2))))
    right = int(np.argmin(np.abs(phase - (midpoint - np.pi/2))))
    width = abs(u[right] - u[left])
    if index in (0, len(u) - 1) or width <= 0:
        raise ValueError("No bracketed resonance for circle initialization; supply p0")
    phase_fit = least_squares(
        lambda p: p[2] - 2*np.arctan(2*(u - p[0])/np.exp(p[1])) - phase,
        [u[index], np.log(width), phase[index]], max_nfev=1000,
        bounds=([-np.inf, np.log(np.finfo(float).eps), -np.inf],
                [np.inf, -np.log(np.finfo(float).eps), np.inf]))
    if not phase_fit.success:
        raise ValueError("Cannot initialize resonance phase; supply p0")
    fr = reference + span * phase_fit.x[0]
    loaded_q = fr / (span * np.exp(phase_fit.x[1]))
    radial_direction = np.exp(1j * (phase_fit.x[2] - np.pi))
    gain = center + radius * radial_direction
    depth = 2 * radius * radial_direction / gain
    return np.array([fr, loaded_q, loaded_q / abs(depth), np.angle(depth),
                     abs(gain), np.angle(gain), delay])


class ResonatorCircleFit:
    """Fit a hanger with explicit units, reference plane and coupling-Q convention.

    S21(f) = a exp(i alpha - 2 pi i (f-f_ref) tau)
             * [1 - (Ql/Qc_mag) exp(i phi) / (1 + 2 i Ql (f-fr)/fr)].
    Frequencies are Hz and tau is seconds. Qc = Qc_mag exp(-i phi),
    Qc_eff = 1/Re(1/Qc), and 1/Qi = 1/Ql - 1/Qc_eff. Negative/undefined
    extracted Q is an error, never repaired with abs(). p0 uses parameter_names
    order and bypasses circle initialization. Signed delay is permitted, including
    for de-embedded data. This is not V1 constructor/return-value compatibility.
    """

    parameter_names = ("fr", "Ql", "Qc_mag", "phi", "a", "alpha", "tau")
    derived_parameter_names = ("Qi", "Qc_eff", "Qc_real", "Qc_imag")

    def __init__(self, frequency_hz, s21_complex, make_plots=False, *, p0=None, maxfev=10000):
        """Own original samples; sort only initialization, preserving residual/model order."""
        self.freqs = _real_array(frequency_hz, "frequency_hz", 1)
        data = np.asarray(s21_complex)
        if (data.ndim != 1 or data.dtype.kind not in "biufc" or not np.all(np.isfinite(data))
                or data.shape != self.freqs.shape or len(data) < 8):
            raise ValueError("Hanger fitting needs at least eight matching finite complex samples")
        if np.any(self.freqs <= 0) or len(np.unique(self.freqs)) != len(self.freqs):
            raise ValueError("Frequencies must be positive distinct Hz coordinates")
        if type(maxfev) is not int or maxfev < 1:
            raise ValueError("maxfev must be a positive integer")
        self.data = np.array(data, dtype=complex, copy=True)
        self.f_ref = float((self.freqs.min() + self.freqs.max()) / 2)
        self.diagnostics = []
        order = np.argsort(self.freqs)
        f, z = self.freqs[order], self.data[order]
        self._reject_background_only(f, z)
        self.p0 = _real_array(_initial_guess(f, z, self.f_ref) if p0 is None else p0, "p0", 1)
        if (self.p0.shape != (7,) or np.any(self.p0[[0, 1, 2, 4]] <= 0)
                or abs(self.p0[3]) >= np.pi/2):
            if p0 is None:
                raise ValueError("Circle initialization did not yield a physical hanger seed; inspect data or supply p0")
            raise ValueError("p0 must have seven parameters, positive fr/Ql/Qc_mag/a and |phi| < pi/2")
        self._fit(maxfev)
        self._check_physical()
        self._results()
        self._diagnostics(f)
        if make_plots:
            self.make_plots()

    @staticmethod
    def _reject_background_only(frequency, data):
        """Reject exact constant-gain/delay traces, not low-SNR data by an arbitrary cutoff."""
        span = np.ptp(frequency)
        u = (frequency - np.mean(frequency)) / span
        phase = np.unwrap(np.angle(data))
        slope, intercept = np.linalg.lstsq(np.column_stack((u, np.ones(len(u)))), phase, rcond=None)[0]
        background = np.mean(np.abs(data)) * np.exp(1j * (slope*u + intercept))
        scale = np.max(np.abs(data))
        tolerance = 100*np.finfo(float).eps * scale * max(1, np.ptp(phase))
        if scale == 0 or np.max(np.abs(data - background)) <= tolerance:
            raise ValueError("No hanger feature beyond a constant-gain/delay background")

    def _model(self, frequency, parameters):
        fr, ql, qc_mag, phi, amplitude, alpha, delay = parameters
        detuning = 2 * ql * (np.asarray(frequency) - fr) / fr
        background = amplitude * np.exp(1j * alpha - 2j*np.pi*(np.asarray(frequency) - self.f_ref)*delay)
        return background * (1 - (ql/qc_mag)*np.exp(1j*phi)/(1 + 1j*detuning))

    def model(self, frequency_hz=None):
        """Evaluate the fitted complex response, defaulting to original sample order."""
        return self._model(self.freqs if frequency_hz is None else frequency_hz, self.popt)

    def _fit(self, maxfev):
        """Refine all seven parameters on original complex residuals in scaled coordinates."""
        span = np.ptp(self.freqs)
        scales = np.array([span, self.p0[1], self.p0[2], 1, self.p0[4], 1, 1/span])
        lower = np.array([-np.inf, 0, 0, -np.pi/2, 0, -np.inf, -np.inf])
        upper = np.array([np.inf, np.inf, np.inf, np.pi/2, np.inf, np.inf, np.inf])

        def residual(q):
            error = (self._model(self.freqs, self.p0 + scales*q) - self.data) / self.p0[4]
            return np.r_[error.real, error.imag]

        result = least_squares(residual, np.zeros(7),
            bounds=((lower-self.p0)/scales, (upper-self.p0)/scales),
            x_scale="jac", max_nfev=maxfev)
        if not result.success:
            raise RuntimeError("Hanger fit did not converge: " + result.message)
        self.popt = self.p0 + scales * result.x
        self.popt[5] = np.angle(np.exp(1j*self.popt[5]))
        self.residuals = self.data - self.model()
        self.sq_error = float(np.sum(np.abs(self.residuals)**2))
        self.dof = 2*len(self.data) - 7
        self.relative_rms = float(np.sqrt(np.mean(np.abs(self.residuals)**2)) / self.popt[4])
        self._active_bounds = bool(np.any(result.active_mask))
        # SVD avoids normal-equation squaring of the condition number. No finite
        # pseudoinverse errors are reported for unidentifiable parameter directions.
        _, singular, vh = np.linalg.svd(result.jac, full_matrices=False)
        threshold = singular[0] * np.finfo(float).eps * max(result.jac.shape)
        self.jacobian_condition = float(singular[0]/singular[-1]) if singular[-1] else np.inf
        if np.count_nonzero(singular > threshold) != 7:
            self.fit_cov = np.full((7, 7), np.nan)
        else:
            cov = (vh.T / singular**2) @ vh
            cov *= np.sum(result.fun**2) / self.dof
            self.fit_cov = cov * np.outer(scales, scales)

    def _check_physical(self):
        """Refuse positive-looking magnitudes when inferred internal dissipation is invalid."""
        fr, ql, qc_mag, phi, amplitude, _, _ = self.popt
        if not np.all(np.isfinite(self.popt)) or min(fr, ql, qc_mag, amplitude) <= 0:
            raise ValueError("Hanger fit returned nonpositive or nonfinite physical parameters")
        if not self.freqs.min() < fr < self.freqs.max():
            raise ValueError("Fitted resonance lies outside the measured frequency span")
        inverse_internal = 1/ql - np.cos(phi)/qc_mag
        if inverse_internal <= 0 or not np.isfinite(1/inverse_internal):
            raise ValueError("Extracted internal Q is nonpositive/nonfinite; inspect normalization, geometry and background")
        if not np.isfinite(qc_mag/np.cos(phi)) or np.cos(phi) <= 0:
            raise ValueError("Extracted dissipative coupling Q is nonpositive/nonfinite")

    def _results(self):
        """Propagate full covariance, including mismatch-phase correlations, to derived Q."""
        _, ql, magnitude, phi, _, _, _ = self.popt
        internal = 1/(1/ql - np.cos(phi)/magnitude)
        effective = magnitude/np.cos(phi)
        coupling = magnitude * np.exp(-1j*phi)
        self.fit_params = dict(zip(self.parameter_names, self.popt))
        self.fit_params.update(Qi=internal, Qc=coupling, Qc_eff=effective,
                               Qc_real=coupling.real, Qc_imag=coupling.imag)
        transform = np.zeros((4, 7))
        transform[0, 1:4] = [internal**2/ql**2, -internal**2*np.cos(phi)/magnitude**2,
                             -internal**2*np.sin(phi)/magnitude]
        transform[1, 2:4] = [1/np.cos(phi), magnitude*np.sin(phi)/np.cos(phi)**2]
        transform[2, 2:4] = [np.cos(phi), -magnitude*np.sin(phi)]
        transform[3, 2:4] = [-np.sin(phi), -magnitude*np.cos(phi)]
        self.derived_cov = transform @ self.fit_cov @ transform.T
        self.Qc_cov = self.derived_cov[2:, 2:].copy()
        self.fit_errors = dict(zip(self.parameter_names, np.sqrt(np.diag(self.fit_cov))))
        self.fit_errors.update(zip(self.derived_parameter_names, np.sqrt(np.diag(self.derived_cov))))
        self.fit_function = self.model

    def _warn(self, message):
        self.diagnostics.append(message)
        warnings.warn(message, ResonatorFitWarning, stacklevel=4)

    def _diagnostics(self, sorted_frequency):
        """Report resolvability heuristics without inventing a noise model or chi-square."""
        p = self.fit_params
        width = p["fr"]/p["Ql"]
        count = np.count_nonzero(np.abs(sorted_frequency-p["fr"]) <= width/2)
        if count < 5:
            self._warn("Fewer than five samples across the fitted linewidth; Q may be undersampled")
        if min(p["fr"]-sorted_frequency[0], sorted_frequency[-1]-p["fr"]) < 2*width:
            self._warn("Less than two linewidths of background on at least one side; inspect gain/delay constraints")
        if (not np.all(np.isfinite(self.fit_cov)) or not np.all(np.isfinite(self.derived_cov))
                or self.jacobian_condition > 1e8):
            self._warn("Covariance is rank-deficient or ill-conditioned; uncertainty estimates are unreliable")
        if self._active_bounds:
            self._warn("Optimizer reached a physical parameter bound; inspect fit assumptions")
        if any(np.isfinite(self.fit_errors[name]) and self.fit_errors[name] >= p[name]
               for name in ("Qi", "Ql", "Qc_mag", "Qc_eff")):
            self._warn("Q uncertainty is comparable to its estimate; the Q extraction is poorly constrained")

    def make_plots(self):
        """Plot original complex data/model and residuals; no backend/style/show side effects."""
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(2, 2, constrained_layout=True)
        order = np.argsort(self.freqs)
        f = self.freqs[order]
        prediction = self.model(f)
        axes[0, 0].plot(self.data.real, self.data.imag, ".", label="Data")
        axes[0, 0].plot(prediction.real, prediction.imag, "-", label="Fit")
        axes[0, 0].set(xlabel="Re S21", ylabel="Im S21", title="Original complex response")
        axes[0, 1].plot(self.freqs, np.abs(self.data), ".", label="Data")
        axes[0, 1].plot(f, np.abs(prediction), "-", label="Fit")
        axes[0, 1].set(xlabel="Frequency (Hz)", ylabel="|S21|")
        axes[1, 0].plot(self.freqs, np.angle(self.data), ".", label="Data")
        axes[1, 0].plot(f, np.angle(prediction), ".", label="Fit")
        axes[1, 0].set(xlabel="Frequency (Hz)", ylabel="Phase (rad)")
        axes[1, 1].plot(self.freqs, self.residuals.real, ".", label="Real")
        axes[1, 1].plot(self.freqs, self.residuals.imag, ".", label="Imaginary")
        axes[1, 1].set(xlabel="Frequency (Hz)", ylabel="Data - model")
        for axis in axes.flat:
            axis.legend()
        return axes
