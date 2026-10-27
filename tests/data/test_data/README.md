# Synthetic V2 data for the manual DataBrowser demo

From the repository root, run `python tests/data/gen_test_data.py` to generate
`resonances.auspex/` (noisy Lorentzian dips and a complex S21 trace),
`ramsey.auspex/` (beating, chirped Ramsey-like interference), and
`complex_map.auspex/` (one complex 2D resonance map on a nonuniform grid). The data are
fully synthetic and deterministic; no hardware is used.

Run `python tests/data/visual_data_browser.py` to generate missing fixtures and
open them in DataBrowser, with the 2D map selected. Check stacked/four-panel
modes and colormaps, then select a single component, enable both cuts and drag
their lines. Cuts should track the diagonal resonance and display wrapped phase
when Phase is selected. These `.auspex` folders are gitignored; the generator
never overwrites existing run folders. Move or remove these *synthetic* folders
manually if you want to regenerate them.
