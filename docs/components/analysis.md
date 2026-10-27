# Offline fitting

Core fits are independent of experiments, services and filter pipelines. Install
SciPy and Matplotlib through the optional extra; acquisition still needs only its
existing dependencies. Complex S21 has a separate
[hanger resonator fitter](resonator-fits.md); the models below take real values:

```sh
python -m pip install -e ".[analysis]"
```

```python
import numpy as np
from auspex_v2.analysis import LorentzFit

frequency = np.linspace(4.9, 5.1, 401)  # GHz
magnitude = 1 - 0.5 / (1 + ((frequency - 5.01) / 0.005)**2)
fit = LorentzFit(frequency, magnitude)
center = fit.fit_params["b"]          # GHz
fwhm = fit.fit_params["c"]            # GHz, not half-width
predicted = fit.model()              # fitted values at the original coordinates
axis = fit.make_plots()              # explicit Matplotlib plotting; returns Axes
```

To use saved data, load it with [open_data](data.md#load-a-dataset-with-open_data),
then pass actual coordinates and a deliberately chosen real-valued observable:

```python
from auspex_v2.data import open_data

measurement = open_data(path="measurement.auspex", group="measurement", dataset="s21")
fit = LorentzFit(measurement.coordinates["frequency"].values,
                 np.abs(measurement.data))
```

For a sweep of traces, select a row explicitly, e.g. `measurement.data[0]`.
**Magnitude, power, real and imaginary components are different observables.**
No component, slice or normalization is selected automatically. These models
reject complex values rather than silently discarding imaginary parts.

## Models and parameter conventions

Import classes from `auspex_v2.analysis` or `auspex_v2.analysis.fits`.

| Class | Model | Parameter order / dictionary keys |
| --- | --- | --- |
| `LorentzFit` | `A / ((x-b)**2 + (c/2)**2) + d` | `A, b, c, d` |
| `GaussianFit` | `B + A*exp(-0.5*((x-μ)/σ)**2)` | `B, A, μ, σ` |
| `MultiGaussianFit` | Constant plus N Gaussians | `B, A0, μ0, σ0, A1, μ1, σ1, ...` |
| `QuadraticFit` | `A*(x-x0)**2 + b` | `A, x0, b` |

Lorentzian `c` is FWHM; height above background is `4*A/c**2`. Gaussian `A` is
height above background and `σ` is standard deviation, not FWHM. Single-peak fits
initialize either peaks or dips. Widths default to positive bounds; amplitude
signs are unrestricted. Units follow the input coordinates and values: no unit
conversion or physical interpretation is inferred.

## Guesses, bounds and results

The common 1D constructor is:

```python
fit = GaussianFit(x, y, make_plots=False, ax=None,
                  p0=[0.1, 1.0, 0.0, 0.5],
                  bounds=([-1, 0, -2, 0.01], [1, 3, 2, 2]),
                  maxfev=10000)
```

`p0` and `(lower, upper)` bounds use the table's parameter order; explicit bounds
replace the defaults. Initial guesses must be finite and within bounds. `maxfev`
sets SciPy's `max_nfev` budget per start; finite-difference Jacobian calls can add
evaluations. Fitting uses unweighted nonlinear least squares. Inputs are owned finite real array snapshots, with more observations
than parameters. Nonuniform and descending X coordinates are supported; only
automatic peak initialization sorts a working copy and requires distinct X.
Missing/NaN observations are rejected, never silently dropped.

Automatic peak seeds estimate a constant background from the outer 10% of samples
at each end (at least one sample per end). They need a bracketed feature. For
edge/truncated features, broad backgrounds or unresolved mixtures, supply reviewed
initial guesses rather than treating initialization as feature detection.
Quadratic initialization uses centered polynomial least squares and converts
correctly to vertex form; zero/unresolvable curvature is an explicit error.
Internal optimizer parameter centering/scaling improves numerical differences on
GHz coordinates or very small widths. Returned results stay in caller units.

Results are ordinary inspectable arrays/dictionaries:

- `fit_params`, `fit_errors`: fitted parameters and local one-standard-deviation
  covariance estimates, keyed in the table's order.
- `popt`, `fit_cov`: parameter vector and full covariance in that same result order.
- `residuals`: observed minus fitted values, in observation order/shape.
- `sq_error`: unweighted sum of squared residuals.
- `dof`: number of scalar observations minus number of fitted parameters.
- `p0`, `bounds`, `start_guesses`, `start_errors`: the actual initialization/bounds
  and failed multistart diagnostics. These retain input component order.

`model(x)` evaluates new coordinates; `model()` uses the original coordinates.
`fit_function(x)` remains available. Numerical failure raises an exception;
SciPy covariance warnings are not hidden. Covariance uses SciPy's default
residual-variance scaling and local linearization. Bounds, parameter correlations,
poor sampling and model mismatch can make errors unreliable. Convergence or a
small residual is **not evidence of scientific validity or identifiability**.
There is no `Nsigma` statistic: unweighted RSS without a noise model is not a
chi-squared goodness-of-fit test. Weighted fitting is not part of this slice.

## Multiple Gaussians

```python
from auspex_v2.analysis import MultiGaussianFit

fit = MultiGaussianFit(x, y, n_gaussians=2)
# Explicit seeds for overlapping components; optional reproducible local restarts:
fit = MultiGaussianFit(x, y, n_gaussians=2,
    p0=[0.2, 1.6, -2.0, 0.5, 0.8, 2.1, 0.8],
    n_starts=5, random_state=42)
```

`n_gaussians` must be at least two. The automatic initializer selects the requested
number of **resolved positive peaks**, ranked by SciPy peak prominence. Default
minimum prominence is the larger of five times a robust difference-based noise
estimate and 5% of the signal range; `prominence=` overrides that heuristic.
Centers/heights seed means/amplitudes; half-prominence widths are interpolated
onto actual X coordinates and divided by `sqrt(8*log(2))` to seed sigma. For
mixtures these are only starting guesses, not measured linewidths.

Too few credible peaks raises an error asking for `p0` or a reviewed prominence.
Negative components and strongly overlapping mixtures need explicit guesses.
There is no random synthetic sampling, clustering, scikit-learn dependency or
automatic choice of component count.

`n_starts=1` is deterministic. More starts require an explicit `random_state`
(integer seed or NumPy Generator); the first start remains unchanged. Additional
starts perturb background, amplitudes, centers and widths, clipped to supplied
bounds. The converged solution with lowest RSS is retained. Reusing an integer
seed reproduces starts; a Generator consumes its explicit state. Global NumPy
random state is not used. Multistart does not guarantee a global optimum.
Results are ordered by increasing fitted center, **including covariance rows and
columns**. Component labels may become unstable when peaks are not distinguishable.
V1's `n_samples` argument is intentionally not retained.

## Custom 1D and 2D models

Subclass `AuspexFit`; implement vectorized `_model(x, *p)`, `_initial_guess()` and
`parameter_names` (or the V1-style `_fit_dict(p)`). Supplying `p0` bypasses the
initializer. An optional `_parameter_scales(p0)` supplies positive characteristic
scales for mixed-unit custom parameters.

`Auspex2DFit` uses `xpts`, `ypts` vectors and `zpts` shaped `(len(ypts), len(xpts))`:

```python
from auspex_v2.analysis import Auspex2DFit

class PlaneFit(Auspex2DFit):
    """Fit independent X/Y slopes and an offset on a rectangular grid."""
    parameter_names = ("a", "b", "offset")

    @staticmethod
    def _model(coordinates, *p):
        return p[0]*coordinates[0] + p[1]*coordinates[1] + p[2]

xx, yy = np.meshgrid([-2., 0., 3.], [-1., 1., 4., 5.])
plane = PlaneFit(xx[0], yy[:, 0], 2*xx - yy + 0.4, p0=[0., 0., 0.])
assert plane.model().shape == (4, 3)
```

Both coordinate axes must have nonzero span. The 2D model receives coordinates
shaped `(2, N)`, X first, Y second. `model()` and
`residuals` have the original Z shape; explicit `model(coordinates)` evaluates the
supplied points. `make_plots()` returns two Matplotlib axes comparing measured and
fitted values on the original grid. Ragged/dependent-coordinate fitting is not
provided by this rectangular base.

Matplotlib is imported only when plotting is requested (`make_plots=True` or the
method). The library does not change style, select a backend, or call `show()`.
For a noninteractive script use `axis.figure.savefig(...)`; in an interactive
script call `matplotlib.pyplot.show()` explicitly. No GUI fitting actions, live
pipeline fitter, fit-result persistence or qubit models are included here.
For complex internal/coupling Q extraction, use the separately documented
[hanger resonator fitter](resonator-fits.md).
