# Synthetic V2 data for the manual DataBrowser demo

From the repository root, run `python tests/data/gen_test_data.py` to generate
`resonances.auspex/` (noisy Lorentzian dips and a complex S21 trace) and
`ramsey.auspex/` (beating, chirped Ramsey-like interference). The data are
fully synthetic and deterministic; no hardware is used.

Run `python tests/data/visual_data_browser.py` to generate missing fixtures and
open them in DataBrowser. These `.auspex` folders are gitignored; the generator
never overwrites existing run folders. Move or remove these *synthetic* folders
manually if you want to regenerate them.
