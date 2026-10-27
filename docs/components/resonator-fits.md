# Hanger resonator fitting

`ResonatorCircleFit` extracts the internal and coupling Q of **one hanger/notch
S21 resonance**. It is standalone offline analysis, using the same optional
[SciPy/Matplotlib extra](analysis.md). It does not fit reflection, transmission
peaks, multiple resonances or arbitrary frequency-dependent backgrounds.

```python
import numpy as np
from auspex_v2.analysis import ResonatorCircleFit

fr, qi, qc_eff = 6.1e9, 4e5, 2e5
ql = 1 / (1/qi + 1/qc_eff)
frequency = fr + np.linspace(-1e6, 1e6, 401)  # Hz
# Synthetic reference: NOT measured data or a device-calibration procedure.
s21 = 0.8*np.exp(0.7j - 2j*np.pi*(frequency-fr)*65e-9) * (
    1 - (ql/qc_eff)*(1 + 1j*np.tan(0.2)) / (1 + 2j*ql*(frequency-fr)/fr))
fit = ResonatorCircleFit(frequency, s21)
internal_q = fit.fit_params["Qi"]
coupling_q = fit.fit_params["Qc"]          # complex, not its magnitude
axes = fit.make_plots()                    # explicit; no show()/style/backend change
```

For saved complex data, use [open_data](data.md#load-a-dataset-with-open_data),
pass the **Hz** coordinate and complex values directly, and select a trace row
explicitly for a sweep of traces. Unlike the real-valued core fits, this routine
needs both quadratures; do not take magnitude or power first. The reader does not
convert coordinate units: multiply a GHz coordinate by `1e9` yourself.

## Model and conventions

The fitted complex model is

```text
G(f) = a * exp(i*alpha - 2*pi*i*(f-f_ref)*tau)
S21(f) = G(f) * [1 - (Ql/Qc_mag)*exp(i*phi) /
                         (1 + 2*i*Ql*(f-fr)/fr)]
```

- `fr`: resonance frequency in **Hz**.
- `Ql`: loaded Q.
- `Qc_mag`: magnitude of complex coupling Q.
- `phi`: mismatch phase in radians, restricted to `(-pi/2, pi/2)`.
- `a`: positive background gain magnitude.
- `alpha`: background phase in radians **at `f_ref`**, wrapped to `[-pi, pi]`.
- `tau`: cable delay in **seconds**; signed values support de-embedded data too.

`fit.f_ref` is the midpoint of the measured frequency span, not the fitted
resonance or zero Hz. Shifting this reference changes alpha, not the physical
response. Alpha's uncertainty is a local phase uncertainty, modulo `2*pi`.

The sign and meanings of coupling Q are explicit:

```text
Qc       = Qc_mag * exp(-i*phi)
Qc_eff   = 1 / Re(1/Qc) = Qc_mag / cos(phi)
1/Qi     = 1/Ql - Re(1/Qc)
```

Thus `Qc`, `Qc_mag` and the effective **dissipative** coupling `Qc_eff` are not
interchangeable. The model numerator has `exp(+i*phi)` because it uses `Ql/Qc`.

## Initialization and refinement

```python
fit = ResonatorCircleFit(frequency_hz, s21_complex, make_plots=False,
                        p0=None, maxfev=10000)
```

Inputs must have at least eight matching finite samples, with distinct positive
real frequencies. Ascending, descending, nonuniform and unsorted coordinates
are supported: initialization sorts a working copy, while `freqs`, `data`,
`model()` and `residuals` retain the original sample order and owned snapshots.

The automatic initializer:

1. Estimates delay from separate unwrapped phase slopes at the sweep edges.
2. Searches a bounded local delay correction by circle radial error, sampling
   candidate basins before local minimization (the error is not unimodal).
3. Fits the corrected circle and its phase evolution, recovering an initial
   linewidth, off-resonant complex gain and coupling depth.
4. Refines **all seven parameters against original complex samples** using real
   and imaginary residuals. No input points are interpolated or discarded.

This is a local numerical initializer, not a global delay-unwrapping guarantee.
Its delay search covers ±half a phase turn across the span around the edge-slope
estimate. Strong delay aliasing, too little off-resonance data or distorted
backgrounds may require independent calibration or explicit reviewed guesses.
A small residual cannot distinguish aliased delays on a uniform frequency grid.

`p0` bypasses circle initialization and uses the seven parameters in this order:
`[fr, Ql, Qc_mag, phi, a, alpha, tau]`. Alpha must refer to the same midpoint
`f_ref`. Explicit guesses cannot waive physical-result checks. `maxfev` sets
SciPy's `max_nfev` budget for the final refinement, excluding additional numerical
Jacobian calls. Initializers have their own finite iteration limits.

The low-level `circle_fit(complex_samples)` helper returns
`(radius, center_real, center_imag)` from centered/scaled algebraic initialization
and geometric radial least squares. It rejects collinear/degenerate data; it does
not estimate Q or certify that a poorly sampled arc determines a reliable circle.

## Results and uncertainties

`fit_params` contains the seven primary parameters plus `Qi`, complex `Qc`,
`Qc_eff`, `Qc_real` and `Qc_imag`.

- `popt` and `fit_cov`: seven primary values and their covariance, in
  `parameter_names` order above, in physical units.
- `fit_errors`: scalar one-standard-deviation errors for primary parameters,
  `Qi`, `Qc_eff`, `Qc_real` and `Qc_imag`. There is **no single scalar ± error for
  complex `Qc`**.
- `derived_cov`: full covariance for `derived_parameter_names`, ordered
  `Qi, Qc_eff, Qc_real, Qc_imag`. Includes phase/cross-parameter correlations.
- `Qc_cov`: 2×2 covariance of coupling Q's real and imaginary components.
- `model(frequency_hz)` / `model()`: predicted complex S21. `fit_function` is also
  available.
- `residuals`: complex data minus model; `sq_error`: sum of squared complex
  magnitudes; `relative_rms`: RMS complex residual divided by fitted gain `a`.
- `dof`: `2*number_of_complex_samples - 7`; `p0`: actual initial guess.
- `diagnostics`: retained warning messages; `jacobian_condition`: condition of
  the scaled optimizer Jacobian.

Least squares is unweighted, assuming comparable independent noise in the two
quadratures for covariance interpretation. Covariance is scaled by residual
variance, and derived errors use local first-order propagation. Noise-free
synthetic data can therefore yield nearly zero formal errors. Neither these
errors nor convergence establish physical validity; frequency-calibration error,
model mismatch, correlated noise and global ambiguities are not accounted for.
There is no unweighted `Nsigma`/chi-squared goodness-of-fit claim.

## Errors versus warnings

Clear exceptions reject invalid inputs, exact background-only traces, failed
convergence, an out-of-span fitted resonance, or nonpositive/nonfinite extracted
Q. **Negative internal Q is not repaired with an absolute value or clipping.**
Inspect geometry, normalization and background assumptions instead.

`ResonatorFitWarning` reports softer limitations, also recorded in `diagnostics`:

- Fewer than five points inside the fitted FWHM.
- Less than two linewidths of background on either side.
- Rank-deficient covariance (reported as NaN) or a scaled Jacobian condition above
  `1e8`; these are numerical diagnostics, not acceptance probabilities.
- A reached physical parameter bound or Q uncertainty comparable to its value.

These are inspection heuristics, not a replacement for a reviewed noise model.
Plots compare original complex data/model, magnitude, wrapped phase and complex
residuals. They import Matplotlib only when requested; save/show explicitly.

This replaces the unfinished V1 wrapper rather than promising argument/list-return
compatibility: frequency comes first, units are Hz/s, no ignored plotting flag,
no fixed manual-Q override, no determinant root search, and no omitted phase
covariance. The old `resonator_circle_fit` list-return helper is not retained.
Qubit fitting remains deferred. Synthetic correctness tests are **not** attended
validation on real resonator measurements.
