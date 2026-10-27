"""Offline real-valued least-squares fits, adapted from Auspex V1 fits.py.

Models use caller-supplied coordinate units. Covariance errors are local,
residual-scaled least-squares estimates, not a test of model validity. Matplotlib
is imported only for explicitly requested plots; no global style is changed.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths


__all__ = ["AuspexFit", "Auspex2DFit", "LorentzFit", "GaussianFit",
           "MultiGaussianFit", "QuadraticFit"]


def _real_array(values, name, ndim):
    """Reject implicit complex projection and missing observations; own input snapshots."""
    array = np.asarray(values)
    if array.dtype.kind not in "biuf" or array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite real {ndim}D array")
    return np.array(array, dtype=float, copy=True)


def _sorted_trace(x, y):
    """Automatic peak widths need unique ordered coordinates, not uniform spacing."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Automatic peak initialization needs distinct X points; supply p0")
    return x, y


def _background(y):
    count = max(1, len(y) // 10)
    return float(np.median(np.r_[y[:count], y[-count:]]))


def _widths(x, signal, peaks):
    """Interpolate fractional sample crossings onto physical X before converting widths."""
    _, _, left, right = peak_widths(signal, peaks, rel_height=0.5)
    samples = np.arange(len(x))
    return np.interp(right, samples, x) - np.interp(left, samples, x)


def _single_peak(x, y):
    """Seed a peak or dip above a constant background; require a bracketed feature."""
    x, y = _sorted_trace(x, y)
    baseline = _background(y)
    index = int(np.argmax(np.abs(y - baseline)))
    height = y[index] - baseline
    if height == 0 or index in (0, len(x) - 1):
        raise ValueError("No bracketed peak/dip for automatic initialization; supply p0")
    signal = np.sign(height) * (y - baseline)
    width = _widths(x, signal, [index])[0]
    if width <= 0:
        raise ValueError("Cannot estimate peak width; supply p0")
    return baseline, height, x[index], width


class AuspexFit:
    """Fit a vectorized real 1D model with SciPy, keeping V1-style result access.

    Subclasses implement _model(x, *parameters), _initial_guess(), and either
    parameter_names or _fit_dict(). p0 bypasses automatic initialization. Bounds
    use the same parameter order; built-in peak models default to positive widths.
    At least one residual degree of freedom is required. Invalid data are rejected,
    never silently projected, sorted for fitting, or dropped.
    """

    xlabel = "X Data"
    ylabel = "Y Data"
    title = "Auspex Fit"
    parameter_names = ()
    width_indices = ()
    bounds = None

    def __init__(self, xpts, ypts, make_plots=False, ax=None, *, p0=None, bounds=None, maxfev=10000):
        """Snapshot observations and fit immediately; plotting is an explicit side effect."""
        self.xpts = _real_array(xpts, "xpts", 1)
        self.ypts = _real_array(ypts, "ypts", 1)
        if self.xpts.shape != self.ypts.shape or not self.xpts.size or np.ptp(self.xpts) == 0:
            raise ValueError("X/Y must have matching nonempty shapes and a nonzero X span")
        self.ax = ax
        self._do_fit(self.xpts, self.ypts, p0, bounds, maxfev)
        if make_plots:
            self.make_plots()

    @staticmethod
    def _model(x, *parameters):
        raise NotImplementedError("Implement a vectorized model")

    def _initial_guess(self):
        raise NotImplementedError("Supply p0 or implement _initial_guess")

    def _fit_dict(self, parameters):
        if len(self.parameter_names) != len(parameters):
            raise ValueError("Parameter names must match the model parameter count")
        return dict(zip(self.parameter_names, parameters))

    def _parameter_scales(self, guess):
        """Subclasses can set characteristic scales for mixed-unit parameters."""
        return np.maximum(np.abs(guess), 1.0)

    def _starting_points(self, guess, lower, upper):
        return [guess]

    def _parameter_order(self, parameters):
        return np.arange(len(parameters))

    def _solve(self, x, y, guess, lower, upper, maxfev):
        """Center/scale optimizer parameters, then transform covariance back to caller units.

        Finite differences on an absolute GHz center or a sub-nanosecond width can
        otherwise step across the entire feature. This changes numerical coordinates,
        not the model or data. The returned parameters/covariance remain physical.
        """
        scales = np.asarray(self._parameter_scales(guess), dtype=float)
        if scales.shape != guess.shape or not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError("Parameter scales must be finite and positive")

        def normalized_model(points, *q):
            return self._model(points, *(guess + scales * q))

        q, covariance = curve_fit(normalized_model, x, y, p0=np.zeros(len(guess)),
            bounds=((lower - guess) / scales, (upper - guess) / scales),
            method="trf", x_scale="jac", max_nfev=maxfev)
        parameters = guess + scales * q
        residuals = y - self._model(x, *parameters)
        if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(residuals)):
            raise RuntimeError("Optimizer returned nonfinite parameters or residuals")
        return parameters, covariance * np.outer(scales, scales), float(np.sum(residuals**2))

    def _do_fit(self, x, y, p0, bounds, maxfev):
        """Select by unweighted RSS; retain actual starts and report all-start failure."""
        guess = _real_array(self._initial_guess() if p0 is None else p0, "p0", 1)
        if not guess.size or y.size <= guess.size:
            raise ValueError("Fit needs more observations than parameters")
        self._fit_dict(guess)  # Validate the public parameter mapping before optimization.
        if type(maxfev) is not int or maxfev < 1:
            raise ValueError("maxfev must be a positive integer")
        if bounds is None:
            bounds = self.bounds
        if bounds is None:
            lower = np.full(guess.size, -np.inf)
            lower[list(self.width_indices)] = 0.0
            bounds = (lower, np.full(guess.size, np.inf))
        lower, upper = [np.broadcast_to(np.asarray(b, dtype=float), guess.shape).copy() for b in bounds]
        if (np.any(np.isnan(lower) | np.isnan(upper)) or np.any(lower >= upper)
                or np.any(guess < lower) or np.any(guess > upper)):
            raise ValueError("Initial guess must lie within valid parameter bounds")
        if not np.all(np.isfinite(self._model(x, *guess))):
            raise ValueError("Initial guess produces a nonfinite model")
        self.p0 = guess.copy()
        self.bounds = (lower, upper)
        self.start_guesses = np.asarray(self._starting_points(guess, lower, upper))
        self.start_errors = []
        best = None
        for start in self.start_guesses:
            try:
                result = self._solve(x, y, start, lower, upper, maxfev)
            except (RuntimeError, ValueError) as exc:
                if len(self.start_guesses) == 1:
                    raise
                self.start_errors.append(str(exc))
                continue
            if best is None or result[2] < best[2]:
                best = result
        if best is None:
            raise RuntimeError("All fitting starts failed: " + "; ".join(self.start_errors))
        parameters, covariance, self.sq_error = best
        order = self._parameter_order(parameters)
        self.popt = parameters[order]
        self.fit_cov = covariance[np.ix_(order, order)]
        self.fit_params = self._fit_dict(self.popt)
        self.fit_errors = self._fit_dict(np.sqrt(np.diag(self.fit_cov)))
        self.fit_function = lambda points: self._model(np.asarray(points), *self.popt)
        self.residuals = y - self.fit_function(x)
        self.sq_error = float(np.sum(self.residuals**2))
        self.dof = y.size - guess.size

    def model(self, x=None):
        """Evaluate fitted values at supplied coordinates, or at the original observations."""
        return self.fit_function(self.xpts if x is None else x)

    def annotation(self):
        return str(self)

    def make_plots(self):
        """Return a Matplotlib Axes without showing figures or changing global style."""
        import matplotlib.pyplot as plt

        if self.ax is None:
            _, self.ax = plt.subplots()
        fine = np.linspace(self.xpts.min(), self.xpts.max(), 501)
        self.ax.plot(self.xpts, self.ypts, ".", label="Data")
        self.ax.plot(fine, self.model(fine), "-", label="Fit")
        self.ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        self.ax.annotate(self.annotation(), xy=(0.03, 0.04), xycoords="axes fraction")
        self.ax.legend()
        return self.ax


class LorentzFit(AuspexFit):
    """Fit A / ((x-b)**2 + (c/2)**2) + d; c is FWHM, A is not peak height."""

    title = "Lorentzian Fit"
    parameter_names = ("A", "b", "c", "d")
    width_indices = (2,)

    @staticmethod
    def _model(x, *p):
        return p[0] / ((x - p[1])**2 + (p[2] / 2)**2) + p[3]

    def _initial_guess(self):
        baseline, height, center, width = _single_peak(self.xpts, self.ypts)
        return [height * width**2 / 4, center, width, baseline]

    def _parameter_scales(self, guess):
        span, signal = np.ptp(self.xpts), max(np.ptp(self.ypts), np.finfo(float).tiny)
        return [signal * span**2, span, span, signal]

    def __str__(self):
        return "A / ((x-b)^2 + (c/2)^2) + d"


class GaussianFit(AuspexFit):
    """Fit B + A exp(-0.5*((x-mu)/sigma)**2), with V1's μ/σ dictionary keys."""

    title = "Gaussian Fit"
    parameter_names = ("B", "A", "μ", "σ")
    width_indices = (3,)

    @staticmethod
    def _model(x, *p):
        return p[0] + p[1] * np.exp(-0.5 * ((x - p[2]) / p[3])**2)

    def _initial_guess(self):
        baseline, height, center, width = _single_peak(self.xpts, self.ypts)
        return [baseline, height, center, width / np.sqrt(8 * np.log(2))]

    def _parameter_scales(self, guess):
        signal = max(np.ptp(self.ypts), np.finfo(float).tiny)
        return [signal, signal, np.ptp(self.xpts), np.ptp(self.xpts)]

    def __str__(self):
        return "B + A exp(-(x-μ)^2 / (2σ^2))"


class MultiGaussianFit(AuspexFit):
    """Fit a constant plus N Gaussians, ordered by increasing fitted center.

    Automatic initialization targets resolved positive peaks. Default prominence
    is max(5 * robust difference-noise estimate, 5% of signal range). An explicit
    prominence overrides this heuristic. Unresolved/negative components need p0
    in order [B, A0, mu0, sigma0, ...]. n_starts includes the unperturbed guess;
    multistart requires an explicit random_state seed or NumPy Generator.
    """

    title = "Sum of Gaussians Fit"

    def __init__(self, x, y, make_plots=False, n_gaussians=2, *, p0=None, bounds=None,
                 ax=None, prominence=None, n_starts=1, random_state=None, maxfev=10000):
        """Use peak geometry, not random synthetic samples or scikit-learn clustering."""
        if type(n_gaussians) is not int or n_gaussians < 2:
            raise ValueError("n_gaussians must be at least two; use GaussianFit for one")
        if type(n_starts) is not int or n_starts < 1 or (n_starts > 1 and random_state is None):
            raise ValueError("n_starts must be positive; multistart requires random_state")
        if prominence is not None and (not np.isfinite(prominence) or prominence < 0):
            raise ValueError("prominence must be finite and nonnegative")
        self.n_gaussians, self.n_starts = n_gaussians, n_starts
        self.random_state, self.prominence = random_state, prominence
        self.parameter_names = ("B",) + tuple(name for j in range(n_gaussians)
                                              for name in (f"A{j}", f"μ{j}", f"σ{j}"))
        self.width_indices = tuple(range(3, 1 + 3 * n_gaussians, 3))
        super().__init__(x, y, make_plots, ax, p0=p0, bounds=bounds, maxfev=maxfev)

    @staticmethod
    def _model(x, *p):
        result = np.full_like(x, p[0], dtype=float)
        for index in range(1, len(p), 3):
            result += p[index] * np.exp(-0.5 * ((x - p[index + 1]) / p[index + 2])**2)
        return result

    def _initial_guess(self):
        """Estimate centers/heights and half-prominence widths on actual X coordinates."""
        x, y = _sorted_trace(self.xpts, self.ypts)
        baseline = _background(y)
        signal = y - baseline
        differences = np.diff(signal)
        noise = np.median(np.abs(differences - np.median(differences))) / (0.67448975 * np.sqrt(2))
        prominence = self.prominence
        if prominence is None:
            prominence = max(5 * noise, 0.05 * np.ptp(signal))
        peaks, properties = find_peaks(signal, prominence=prominence)
        positive = signal[peaks] > 0
        peaks, prominences = peaks[positive], properties["prominences"][positive]
        if len(peaks) < self.n_gaussians:
            raise ValueError("Too few resolved positive peaks; supply p0 or review prominence")
        ranks = np.argsort(prominences, kind="stable")
        peaks = np.sort(peaks[ranks[-self.n_gaussians:]])
        widths = _widths(x, signal, peaks) / np.sqrt(8 * np.log(2))
        if np.any(widths <= 0):
            raise ValueError("Cannot estimate component widths; supply p0")
        return [baseline, *np.column_stack((signal[peaks], x[peaks], widths)).ravel()]

    def _parameter_scales(self, guess):
        signal, span = max(np.ptp(self.ypts), np.finfo(float).tiny), np.ptp(self.xpts)
        return np.array([signal] + [signal, span, span] * self.n_gaussians)

    def _starting_points(self, guess, lower, upper):
        """Jitter heights, centers and widths locally; keep the original first start."""
        starts = [guess.copy()]
        if self.n_starts == 1:
            return starts
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_starts - 1):
            candidate = guess.copy()
            candidate[0] += rng.normal(0, 0.1 * np.ptp(self.ypts))
            for i in range(self.n_gaussians):
                j = 1 + 3 * i
                candidate[j] *= np.exp(rng.normal(0, 0.25))
                candidate[j + 1] += rng.normal(0, abs(guess[j + 2]))
                candidate[j + 2] *= np.exp(rng.normal(0, 0.25))
            starts.append(np.clip(candidate, lower, upper))
        return starts

    def _parameter_order(self, parameters):
        components = np.argsort(parameters[2::3], kind="stable")
        return np.array([0] + [1 + 3 * j + k for j in components for k in range(3)])

    def __str__(self):
        return f"Sum of Gaussians with N={self.n_gaussians}"


class QuadraticFit(AuspexFit):
    """Fit A*(x-x0)**2 + b; automatic initialization requires nonzero curvature."""

    title = "Quadratic Fit"
    parameter_names = ("A", "x0", "b")

    @staticmethod
    def _model(x, *p):
        return p[0] * (x - p[1])**2 + p[2]

    def _initial_guess(self):
        """Solve centered polynomial regression, then correctly convert to vertex form."""
        if len(np.unique(self.xpts)) < 3:
            raise ValueError("Quadratic initialization needs three distinct X points")
        origin, span = np.mean(self.xpts), np.ptp(self.xpts)
        u = (self.xpts - origin) / span
        coefficients = np.linalg.lstsq(np.column_stack((u**2, u, np.ones_like(u))), self.ypts, rcond=None)[0]
        a, slope, intercept = coefficients
        if abs(a) <= np.finfo(float).eps * np.max(np.abs(coefficients)):
            raise ValueError("No resolvable quadratic curvature; supply p0 or another model")
        return [a / span**2, origin - span * slope / (2 * a), intercept - slope**2 / (4 * a)]

    def _parameter_scales(self, guess):
        span, signal = np.ptp(self.xpts), max(np.ptp(self.ypts), np.finfo(float).tiny)
        return [signal / span**2, span, signal]

    def __str__(self):
        return "A(x-x0)^2 + b"


class Auspex2DFit(AuspexFit):
    """Base for real rectangular Z[Y, X] fits, sharing covariance/residual semantics.

    Subclasses provide a vectorized _model(coordinates, *p) taking a (2, N) array
    with X then Y, and _initial_guess()/parameter_names as for AuspexFit. Automatic
    2D model classes (including qubit chevrons) are not part of this core slice.
    """

    def __init__(self, xpts, ypts, zpts, make_plots=False, *, p0=None, bounds=None, maxfev=10000):
        """Own axis vectors/mesh and require the stored last-X, first-Y shape convention."""
        x = _real_array(xpts, "xpts", 1)
        y = _real_array(ypts, "ypts", 1)
        self.zpts = _real_array(zpts, "zpts", 2)
        if self.zpts.shape != (len(y), len(x)) or not x.size or not y.size:
            raise ValueError("Z must have shape (len(ypts), len(xpts))")
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            raise ValueError("Both coordinate axes need nonzero span")
        self.xpts, self.ypts = np.meshgrid(x, y)
        self.ax = None
        points = np.vstack((self.xpts.ravel(), self.ypts.ravel()))
        self._do_fit(points, self.zpts.ravel(), p0, bounds, maxfev)
        self.residuals = self.residuals.reshape(self.zpts.shape)
        if make_plots:
            self.make_plots()

    def model(self, x=None):
        """With no argument return Z-shaped values; explicit (2, N) coordinates return N."""
        if x is None:
            points = np.vstack((self.xpts.ravel(), self.ypts.ravel()))
            return self.fit_function(points).reshape(self.zpts.shape)
        return self.fit_function(x)

    def make_plots(self, axes=None):
        """Compare measured/model values on the original grid; do not invent interpolation."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 1, constrained_layout=True)
        if len(axes) != 2:
            raise ValueError("Supply two axes for the data/model comparison")
        for axis, values, title in zip(axes, (self.zpts, self.model()), ("Data", "Fit")):
            mesh = axis.pcolormesh(self.xpts, self.ypts, values, shading="auto")
            axis.set(xlabel=self.xlabel, ylabel=self.ylabel, title=title)
            axis.figure.colorbar(mesh, ax=axis)
        return axes
