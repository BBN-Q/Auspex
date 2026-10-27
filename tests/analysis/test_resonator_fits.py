"""Hanger extraction against independent Q-loss equations and analytic covariance."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from auspex_v2.analysis import ResonatorCircleFit, ResonatorFitWarning, circle_fit


def hanger(frequency, *, fr=6.1e9, qi=4e5, qc_eff=2e5, phi=0.2, a=0.8, alpha=0.7, tau=65e-9):
    """Reference from additive internal/coupling loss, not the implementation's _model."""
    inverse_loaded = 1/qi + 1/qc_eff
    detuning = 2 * (frequency - fr) / fr / inverse_loaded
    fraction = 1 / (qc_eff * inverse_loaded)
    return a * np.exp(1j*alpha - 2j*np.pi*(frequency-fr)*tau) * (
        1 - fraction * (1 + 1j*np.tan(phi)) / (1 + 1j*detuning))


def primary_guess(fit_reference, *, fr=6.1e9, qi=4e5, qc_eff=2e5, phi=0.2,
                  a=0.8, alpha=0.7, tau=65e-9):
    """Express the independent reference in the fitter's explicitly documented units/order."""
    return [fr, 1/(1/qi + 1/qc_eff), qc_eff*np.cos(phi), phi, a,
            np.angle(np.exp(1j*(alpha - 2*np.pi*(fit_reference-fr)*tau))), tau]


@unittest.skipUnless(HAS_SCIPY, "install auspex-v2[analysis] for hanger-fit tests")
class ResonatorFitTest(unittest.TestCase):
    """Test geometry/conventions and uncertainty propagation without legacy swallowed assertions."""

    def test_circle_geometry_normalization_and_degenerate_data(self):
        """Circle center/radius should be invariant to offset, scale and point ordering."""
        angle = np.linspace(-2.7, 2.8, 101)
        for scale in (1e-6, 1., 1e6):
            points = scale * (2 - 3j + 0.7*np.exp(1j*angle))
            radius, x, y = circle_fit(points[::-1])
            np.testing.assert_allclose([radius, x, y], scale*np.array([0.7, 2, -3]), rtol=1e-9)
        for bad in ([1, 1, 1], [1, 2, 3], [0+1j, np.nan, 1], [1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                circle_fit(bad)

    def test_coupling_regimes_delay_and_mismatch_with_automatic_circle_seeds(self):
        """Recover internal and dissipative coupling Q separately, including weak coupling."""
        cases = [(4e5, 2e5, 0.2, 65e-9, 401),
                 (5e4, 1e6, -0.3, 200e-9, 1601),
                 (1e6, 5e4, 0.4, -35e-9, 601),
                 (2e5, 2e5, 0., 0., 401)]
        for qi, coupling, phi, delay, count in cases:
            with self.subTest(qi=qi, qc=coupling, phi=phi):
                ql = 1/(1/qi + 1/coupling)
                width = 6.1e9/ql
                f = 6.1e9 + width*np.linspace(-10, 10, count)
                data = hanger(f, qi=qi, qc_eff=coupling, phi=phi, tau=delay)
                rng = np.random.default_rng(71)
                data += rng.normal(0, 1e-4, count) + 1j*rng.normal(0, 1e-4, count)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ResonatorFitWarning)
                    fit = ResonatorCircleFit(f, data)
                self.assertFalse(caught, [str(w.message) for w in caught])
                p = fit.fit_params
                np.testing.assert_allclose([p["Ql"], p["Qi"], p["Qc_eff"]], [ql, qi, coupling], rtol=0.003)
                self.assertAlmostEqual(p["phi"], phi, delta=0.001)
                self.assertAlmostEqual(p["tau"], delay, delta=1e-11)
                self.assertAlmostEqual(p["a"], 0.8, delta=0.0001)
                self.assertAlmostEqual(p["alpha"], 0.7, delta=0.001)
                self.assertAlmostEqual(p["fr"], 6.1e9, delta=0.002*width)
                np.testing.assert_allclose(p["Qc"], coupling*np.cos(phi)*np.exp(-1j*phi), rtol=0.003)
                self.assertAlmostEqual(1/p["Qi"], 1/p["Ql"] - np.real(1/p["Qc"]), delta=1e-18)
                self.assertEqual(fit.dof, 2*count - 7)
                self.assertEqual(fit.sq_error, np.sum(np.abs(fit.residuals)**2))
                self.assertFalse(hasattr(fit, "Nsigma"))

    def test_reference_phase_nonuniform_grid_and_original_order_snapshots(self):
        """Phase is at f_ref, not zero Hz; unsorted observations remain unsorted in outputs."""
        fr, qi, coupling, tau = 6.1e9, 3e5, 1e5, 1.2e-6
        width = fr*(1/qi + 1/coupling)
        # Mild nonuniformity, offset span and many background turns exercise phase conventions.
        u = np.linspace(-1, 1, 801)
        f = fr + width*(2 + 9*(0.8*u + 0.2*u**3))
        z = hanger(f, qi=qi, qc_eff=coupling, phi=-0.45, tau=tau)
        order = np.random.default_rng(3).permutation(len(f))
        f, z = f[order], z[order]
        fit = ResonatorCircleFit(f, z)
        expected_phase = np.angle(np.exp(1j*(0.7 - 2*np.pi*(fit.f_ref-fr)*tau)))
        self.assertAlmostEqual(fit.fit_params["alpha"], expected_phase, delta=1e-7)
        self.assertAlmostEqual(fit.fit_params["tau"], tau, delta=1e-13)
        np.testing.assert_allclose(fit.model(), z, atol=1e-8)
        np.testing.assert_array_equal(fit.freqs, f)
        prediction = fit.model().copy()
        f[:] = 1
        z[:] = 0
        np.testing.assert_array_equal(fit.model(), prediction)
        np.testing.assert_allclose(fit.fit_function(fit.freqs), prediction)

    def test_primary_covariance_and_derived_q_propagation(self):
        """Compare to analytic complex-model derivatives, then independent numerical derived-Q derivatives."""
        fr = 6.1e9
        f = fr + np.linspace(-1e6, 1e6, 601)
        rng = np.random.default_rng(29)
        z = hanger(f) + rng.normal(0, 2e-4, len(f)) + 1j*rng.normal(0, 2e-4, len(f))
        fit = ResonatorCircleFit(f, z)
        resonance, ql, magnitude, phi, a, alpha, tau = fit.popt
        gain = a*np.exp(1j*alpha - 2j*np.pi*(f-fit.f_ref)*tau)
        denominator = 1 + 2j*ql*(f-resonance)/resonance
        depth = ql/magnitude*np.exp(1j*phi)
        model = gain*(1-depth/denominator)
        jacobian = np.column_stack((
            gain*depth*(-2j*ql*f/resonance**2)/denominator**2,
            -gain*np.exp(1j*phi)/magnitude/denominator**2,
            gain*depth/(magnitude*denominator), -1j*gain*depth/denominator,
            model/a, 1j*model, -2j*np.pi*(f-fit.f_ref)*model))
        scales = np.array([np.ptp(f), ql, magnitude, 1, a, 1, 1/np.ptp(f)])
        jacobian *= scales
        real_jacobian = np.vstack((jacobian.real, jacobian.imag))
        expected_scaled = np.linalg.inv(real_jacobian.T @ real_jacobian) * fit.sq_error / fit.dof
        np.testing.assert_allclose(fit.fit_cov/np.outer(scales, scales), expected_scaled, rtol=2e-4, atol=1e-14)

        def derived(p):
            qc = p[2]*np.exp(-1j*p[3])
            return np.array([1/(1/p[1] - np.real(1/qc)), 1/np.real(1/qc), qc.real, qc.imag])

        gradient = np.zeros((4, 7))
        for index in range(7):
            step = 1e-5 * max(abs(fit.popt[index]), 1e-8)
            displacement = np.eye(7)[index] * step
            gradient[:, index] = (derived(fit.popt + displacement) - derived(fit.popt - displacement))/(2*step)
        propagated = gradient @ fit.fit_cov @ gradient.T
        np.testing.assert_allclose(fit.derived_cov, propagated, rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(fit.Qc_cov, fit.derived_cov[2:, 2:])
        self.assertAlmostEqual(fit.fit_errors["Qi"]**2, fit.derived_cov[0, 0], delta=1e-8)
        self.assertNotIn("Qc", fit.fit_errors)  # a complex number has no single scalar +/- error
        self.assertLess(fit.relative_rms, 0.001)

    def test_unphysical_q_out_of_span_and_nonconvergence_are_errors(self):
        """Do not turn invalid results into positive Q by clipping or absolute values."""
        f = 6.1e9 + np.linspace(-1e6, 1e6, 201)
        invalid = dict(qi=-1e5, qc_eff=5e4, phi=0.2)
        with self.assertRaisesRegex(ValueError, "internal Q is nonpositive"):
            ResonatorCircleFit(f, hanger(f, **invalid), p0=primary_guess(6.1e9, **invalid))
        tail = 6.1e9 + np.linspace(2e5, 1e6, 101)
        with self.assertRaisesRegex(ValueError, "outside the measured"):
            ResonatorCircleFit(tail, hanger(tail), p0=primary_guess((tail[0]+tail[-1])/2))
        bad_start = primary_guess(6.1e9)
        bad_start[0] += 2e5
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            ResonatorCircleFit(f, hanger(f), p0=bad_start, maxfev=1)

    def test_input_errors_and_background_only_trace(self):
        """Validate complex observations and distinguish an exact cable from a resonance."""
        f = np.linspace(6e9, 6.2e9, 101)
        for data in (np.zeros(len(f)), np.ones(len(f)), 0.8*np.exp(-2j*np.pi*(f-6.1e9)*65e-9)):
            with self.assertRaisesRegex(ValueError, "No hanger feature"):
                ResonatorCircleFit(f, data)
        for frequency, data in ((f[:7], np.ones(7)), (f, np.ones(100)),
                                (np.r_[f[:-1], f[-2]], hanger(f)),
                                (-f, hanger(f)), (f, np.full(len(f), np.nan))):
            with self.subTest(count=len(frequency)), self.assertRaises(ValueError):
                ResonatorCircleFit(frequency, data)
        with self.assertRaisesRegex(ValueError, "p0 must have seven"):
            ResonatorCircleFit(f, hanger(f), p0=[1, 2, 3])

    def test_sparse_sampling_and_short_span_warn_without_silent_repairs(self):
        """Sampling/coverage warnings are inspectable, not arbitrary goodness-of-fit rejection."""
        fr = 6.1e9
        width = fr*(1/4e5 + 1/2e5)
        for f, expected in ((fr + width*np.linspace(-30, 30, 31), "Fewer than five"),
                            (fr + width*np.linspace(-1, 1, 101), "Less than two linewidths"),
                            (fr + width*np.linspace(-0.001, 0.001, 101), "ill-conditioned")):
            with self.subTest(expected=expected), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ResonatorFitWarning)
                fit = ResonatorCircleFit(f, hanger(f), p0=primary_guess(fr))
                self.assertTrue(any(expected in str(w.message) for w in caught))
                self.assertTrue(any(expected in message for message in fit.diagnostics))
                self.assertGreater(fit.fit_params["Qi"], 0)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "install matplotlib for plots")
    def test_save_load_and_explicit_matplotlib_plots(self):
        """Use actual stored complex S21/Hz coordinates and real noninteractive rendering."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from auspex_v2.data import Axis, open_data, save_data
        f = 6.1e9 + np.linspace(-1e6, 1e6, 301)
        with tempfile.TemporaryDirectory() as directory:
            path = save_data(Path(directory)/"hanger", hanger(f), frequency=Axis("frequency", f, "Hz"))
            dataset = open_data(path=path)
            before = dict(matplotlib.rcParams)
            fit = ResonatorCircleFit(dataset.coordinates["frequency"].values, dataset.data, make_plots=True)
            self.assertEqual(dict(matplotlib.rcParams), before)
            self.assertAlmostEqual(fit.fit_params["Qi"], 4e5, delta=1)
            figure = plt.gcf()
            self.assertEqual(len(figure.axes), 4)
            figure.savefig(Path(directory)/"hanger.png")
            plt.close(figure)
