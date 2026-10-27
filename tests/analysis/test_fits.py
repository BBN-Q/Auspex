"""Independent analytical models and linear references for the core offline fits."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_MPL = importlib.util.find_spec("matplotlib") is not None
if HAS_SCIPY:
    from auspex_v2.analysis import AuspexFit, Auspex2DFit, GaussianFit, LorentzFit, MultiGaussianFit, QuadraticFit

    class PlaneFit(Auspex2DFit):
        """A user-defined plane exercises the generic 2D contract without qubit models."""

        parameter_names = ("a", "b", "offset")

        @staticmethod
        def _model(x, *p):
            return p[0] * x[0] + p[1] * x[1] + p[2]

        def _initial_guess(self):
            return [0., 0., 0.]


def mixture(x):
    """Independent two-component reference; do not generate tests with the fitted model."""
    return 0.2 + 1.6 * np.exp(-((x + 2) / 0.5)**2 / 2) + 0.8 * np.exp(-((x - 2.1) / 0.8)**2 / 2)


@unittest.skipUnless(HAS_SCIPY, "install auspex-v2[analysis] for fit tests")
class CoreFitTest(unittest.TestCase):
    """Check physical conventions, covariance and failure behavior, not only convergence."""

    def test_lorentz_peak_dip_offsets_and_nonuniform_descending_coordinates(self):
        """Lorentz c is FWHM and A is height*c²/4, for either sign on a background."""
        x = 2 * np.linspace(-1, 1, 501)**3 + 0.4
        noise = np.random.default_rng(11).normal(0, 0.003, len(x))
        for height in (-0.7, 1.2):
            y = 2.3 + height / (1 + (2 * (x - 0.4) / 0.3)**2) + noise
            fit = LorentzFit(x[::-1], y[::-1])
            p = fit.fit_params
            self.assertAlmostEqual(p["b"], 0.4, delta=0.002)
            self.assertAlmostEqual(p["c"], 0.3, delta=0.003)
            self.assertAlmostEqual(p["d"], 2.3, delta=0.003)
            self.assertAlmostEqual(4 * p["A"] / p["c"]**2, height, delta=0.005)
            np.testing.assert_allclose(fit.model([p["b"] - p["c"] / 2, p["b"] + p["c"] / 2]) - p["d"],
                                       (fit.model(p["b"]) - p["d"]) / 2)
            np.testing.assert_array_equal(fit.model(), fit.model(x[::-1]))

    def test_physical_frequency_and_time_scales(self):
        """A GHz origin and sub-nanosecond width must not defeat finite differences."""
        x = np.linspace(6.09e9, 6.11e9, 601)
        y = 1 - 0.6 / (1 + (2 * (x - 6.1003e9) / 5e5)**2)
        fit = LorentzFit(x, y)
        self.assertAlmostEqual(fit.fit_params["b"], 6.1003e9, delta=1)
        self.assertAlmostEqual(fit.fit_params["c"], 5e5, delta=1)
        t = np.linspace(0, 1e-8, 301)
        signal = 0.1 + 0.8 * np.exp(-((t - 4e-9) / 0.6e-9)**2 / 2)
        fit = GaussianFit(t, signal)
        self.assertAlmostEqual(fit.fit_params["μ"] / 1e-9, 4, delta=1e-5)
        self.assertAlmostEqual(fit.fit_params["σ"] / 1e-9, 0.6, delta=1e-5)
        np.testing.assert_allclose(fit.model(), signal, atol=1e-7)

    def test_gaussian_positive_and_negative_height(self):
        """Single-Gaussian initialization supports dips without changing the model convention."""
        x = np.linspace(-4, 5, 401)
        for amplitude in (2., -1.5):
            y = 3.1 + amplitude * np.exp(-0.5 * ((x - 0.54) / 0.7)**2)
            fit = GaussianFit(x, y)
            np.testing.assert_allclose(list(fit.fit_params.values()), [3.1, amplitude, 0.54, 0.7], atol=1e-6)
            self.assertEqual(tuple(fit.fit_errors), ("B", "A", "μ", "σ"))

    def test_quadratic_vertex_and_covariance_against_linear_least_squares(self):
        """Use an independent polynomial solution and analytic covariance transformation."""
        x = np.linspace(-3, 4, 151)
        y = 3.2 * (x - 1.4)**2 + 7.1 + np.random.default_rng(4).normal(0, 0.02, len(x))
        fit = QuadraticFit(x, y)
        design = np.column_stack((x**2, x, np.ones_like(x)))
        a, b, c = np.linalg.lstsq(design, y, rcond=None)[0]
        reference = [a, -b / (2 * a), c - b**2 / (4 * a)]
        residual = y - design @ [a, b, c]
        covariance = np.linalg.inv(design.T @ design) * np.sum(residual**2) / (len(x) - 3)
        transform = np.array([[1, 0, 0], [b / (2 * a**2), -1 / (2 * a), 0],
                              [b**2 / (4 * a**2), -b / (2 * a), 1]])
        np.testing.assert_allclose(fit.popt, reference, atol=1e-7)
        np.testing.assert_allclose(fit.p0, reference, atol=1e-10)
        np.testing.assert_allclose(fit.fit_cov, transform @ covariance @ transform.T, rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(list(fit.fit_errors.values()), np.sqrt(np.diag(fit.fit_cov)))
        np.testing.assert_allclose(fit.residuals, residual, atol=1e-7)
        self.assertEqual(fit.dof, len(x) - 3)
        self.assertEqual(fit.sq_error, np.sum(fit.residuals**2))
        self.assertFalse(hasattr(fit, "Nsigma"))
        prediction = fit.model().copy()
        x[:] = 0
        y[:] = 0
        np.testing.assert_array_equal(fit.model(), prediction)

    def test_multi_gaussian_deterministic_geometry_on_nonuniform_grid(self):
        """Half-prominence sample widths must be converted to actual coordinate widths."""
        u = np.linspace(-1, 1, 1001)
        x = 6 * (0.7 * u + 0.3 * u**3)
        y = mixture(x) + np.random.default_rng(8).normal(0, 0.002, len(x))
        first, second = MultiGaussianFit(x, y), MultiGaussianFit(x, y)
        np.testing.assert_array_equal(first.p0, second.p0)
        np.testing.assert_array_equal(first.popt, second.popt)
        np.testing.assert_allclose(first.popt, [0.2, 1.6, -2, 0.5, 0.8, 2.1, 0.8], atol=0.004)
        np.testing.assert_allclose(first.p0[[3, 6]], [0.5, 0.8], atol=0.03)

    def test_multistart_seed_and_center_sorted_covariance(self):
        """Reordering components must also reorder full covariance, not only dictionary labels."""
        x = np.linspace(-6, 6, 401)
        y = mixture(x) + np.random.default_rng(22).normal(0, 0.005, len(x))
        p0 = [0.15, 1., 2.2, 1., 1.4, -2.2, 0.6]  # deliberately reverse component order
        one = MultiGaussianFit(x, y, p0=p0)
        first = MultiGaussianFit(x, y, p0=p0, n_starts=4, random_state=13)
        second = MultiGaussianFit(x, y, p0=p0, n_starts=4, random_state=13)
        np.testing.assert_array_equal(first.start_guesses, second.start_guesses)
        np.testing.assert_array_equal(first.popt, second.popt)
        self.assertLessEqual(first.sq_error, one.sq_error + 1e-12)
        self.assertLess(first.fit_params["μ0"], first.fit_params["μ1"])
        columns = [np.ones_like(x)]
        for amplitude, center, width in first.popt[1:].reshape(-1, 3):
            g = np.exp(-0.5 * ((x - center) / width)**2)
            columns.extend((g, amplitude * g * (x - center) / width**2,
                            amplitude * g * (x - center)**2 / width**3))
        jacobian = np.column_stack(columns)
        expected = np.linalg.inv(jacobian.T @ jacobian) * first.sq_error / first.dof
        np.testing.assert_allclose(first.fit_cov, expected, rtol=2e-4, atol=1e-11)

    def test_unresolved_mixture_requires_explicit_guesses(self):
        """Do not create invisible components from random samples when only one peak is seen."""
        x = np.linspace(-3, 3, 301)
        y = 0.3 + 1.2 * np.exp(-0.5 * ((x + 0.2) / 0.6)**2) + 0.8 * np.exp(-0.5 * ((x - 0.35) / 0.7)**2)
        with self.assertRaisesRegex(ValueError, "Too few resolved positive peaks"):
            MultiGaussianFit(x, y)
        fit = MultiGaussianFit(x, y, p0=[0.3, 1.2, -0.2, 0.6, 0.8, 0.35, 0.7])
        np.testing.assert_allclose(fit.model(), y, atol=1e-8)

    def test_2d_base_non_square_grid_and_model_default(self):
        """Check first-Y/last-X layout using a model with analytically independent slopes."""
        x, y = np.array([-2., -1., 0., 0.7, 2., 4., 5.]), np.array([-1., 0., 1., 2., 4.])
        xx, yy = np.meshgrid(x, y)
        z = 2 * xx - 3 * yy + 0.4 + np.random.default_rng(6).normal(0, 0.01, xx.shape)
        fit = PlaneFit(x, y, z)
        design = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
        expected = np.linalg.lstsq(design, z.ravel(), rcond=None)[0]
        np.testing.assert_allclose(fit.popt, expected, atol=1e-7)
        self.assertEqual(fit.model().shape, (5, 7))
        self.assertEqual(fit.residuals.shape, z.shape)
        np.testing.assert_allclose(fit.model(np.vstack((xx.ravel(), yy.ravel()))), fit.model().ravel())
        self.assertEqual(fit.dof, z.size - 3)
        with self.assertRaisesRegex(ValueError, "Z must have shape"):
            PlaneFit(x, y, z.T)

    def test_explicit_guesses_bounds_and_invalid_data(self):
        """Malformed data fail before optimization; explicit guesses bypass peak detection."""
        x = np.linspace(-2, 2, 61)
        y = 0.2 + 2 * np.exp(-0.5 * ((x - 0.3) / 0.5)**2)
        fit = GaussianFit(x, y, p0=[0.1, 1., 0., 0.8], bounds=([-1, 0, -1, 0.01], [1, 4, 1, 2]))
        np.testing.assert_allclose(fit.popt, [0.2, 2, 0.3, 0.5], atol=1e-6)
        for bad_x, bad_y in ((x, y[:-1]), (x, y + 1j), (x, np.full_like(x, np.nan)),
                             (np.zeros_like(x), y), (x.reshape(1, -1), y)):
            with self.subTest(shape=np.shape(bad_y)), self.assertRaises(ValueError):
                GaussianFit(bad_x, bad_y)
        with self.assertRaisesRegex(ValueError, "more observations"):
            GaussianFit(x[:4], y[:4], p0=[0.2, 2, 0.3, 0.5])
        with self.assertRaisesRegex(ValueError, "within valid"):
            GaussianFit(x, y, p0=[0.2, -1, 0, 0.5], bounds=([0, 0, -1, 0.1], [1, 4, 1, 2]))
        with self.assertRaisesRegex(ValueError, "requires random_state"):
            MultiGaussianFit(x, y, n_starts=2)
        with self.assertRaisesRegex(RuntimeError, "maximum number"):
            GaussianFit(x, y, p0=[1, 0.2, -1, 1.5], maxfev=1)

    def test_store_load_and_fit_without_implicit_complex_projection(self):
        """Exercise the lab workflow with saved coordinates and an explicitly chosen magnitude."""
        from auspex_v2.data import Axis, open_data, save_data
        with tempfile.TemporaryDirectory() as directory:
            x = np.linspace(4.9, 5.1, 401)
            amplitude = 1 - 0.5 / (1 + ((x - 5.01) / 0.005)**2)
            path = save_data(Path(directory) / "resonance", amplitude * np.exp(0.4j),
                             frequency=Axis("frequency", x, "GHz"))
            dataset = open_data(path=path)
            with self.assertRaisesRegex(ValueError, "finite real"):
                LorentzFit(x, dataset.data)
            fit = LorentzFit(dataset.coordinates["frequency"].values, np.abs(dataset.data))
            self.assertAlmostEqual(fit.fit_params["b"], 5.01, delta=1e-7)
            self.assertAlmostEqual(fit.fit_params["c"], 0.01, delta=1e-7)

    def test_numeric_import_has_no_plotting_or_sklearn_side_effects(self):
        code = '''import sys
import numpy as np
state = np.random.get_state()
from auspex_v2.analysis import GaussianFit
assert "matplotlib.pyplot" not in sys.modules
assert "sklearn" not in sys.modules
x = np.linspace(-3, 3, 101)
GaussianFit(x, np.exp(-x*x/2))
assert "matplotlib.pyplot" not in sys.modules
for old, new in zip(state, np.random.get_state()):
    np.testing.assert_equal(old, new)
'''
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, timeout=20)

    @unittest.skipUnless(HAS_MPL, "install matplotlib for fit plotting tests")
    def test_matplotlib_plots_are_explicit_and_do_not_change_style(self):
        """Use a real noninteractive backend and caller axes; preserve plotting configuration."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        before = dict(matplotlib.rcParams)
        x = np.linspace(-3, 3, 101)
        fig, axis = plt.subplots()
        fit = GaussianFit(x, np.exp(-x**2 / 2), make_plots=True, ax=axis)
        self.assertIs(fit.ax, axis)
        self.assertEqual(len(axis.lines), 2)
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), x)
        xx, yy = np.meshgrid(x[::10], x[::20])
        plane = PlaneFit(x[::10], x[::20], 2 * xx - yy + 1)
        axes = plane.make_plots()
        with tempfile.TemporaryDirectory() as directory:
            fig.savefig(Path(directory) / "gaussian.png")
            axes[0].figure.savefig(Path(directory) / "plane.png")
        self.assertEqual(dict(matplotlib.rcParams), before)
        plt.close(fig)
        plt.close(axes[0].figure)
