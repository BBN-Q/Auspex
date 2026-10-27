![Auspex](v1/doc/images/Auspex-Small.png)

# Auspex

Auspex, the AUtomated System for Python-based EXperiments, is a framework for
performing laboratory measurements. Originally developed for experiments on
superconducting qubits and magnetic memory elements, it provides general tools
for controlling instruments, defining measurements, processing results, and
saving scientific data.

Auspex aims to make laboratory software flexible and understandable:

- Instrument drivers should be straightforward to write and use.
- Measurement code should be readable, reusable, and adaptable.
- Data acquisition and processing should support different experimental workflows.
- Experimental data and its coordinates, units, and assumptions should be inspectable.

Use Auspex from Python scripts and Jupyter notebooks, whether working directly
with an instrument or automating a measurement.

**Auspex V2 is a work in progress.** Its APIs and file formats may change. The
original version is preserved independently in [`v1/`](v1/).

## Installation

Auspex V2 requires **Python 3.13 or newer**. From a checkout of this repository,
in an activated virtual environment:

```sh
python -m pip install -e .
```

Optional dependencies can be installed individually or together:

```sh
python -m pip install -e ".[visa]"         # VISA instrument communication
python -m pip install -e ".[gui]"          # Graphical tools
python -m pip install -e ".[tui]"          # Terminal interface
python -m pip install -e ".[analysis]"     # Offline fits and Matplotlib plots
python -m pip install -e ".[all]"          # All optional dependencies
```

The `visa` extra includes PyVISA and PyVISA-py; `gui` includes PySide6 and
pyqtgraph; `tui` includes Textual; `analysis` includes SciPy and Matplotlib.
The `all` extra enables all four; the base install remains NumPy-only.
See [offline fitting](docs/components/analysis.md) for models and uncertainty
conventions. Hardware interfaces may require additional system libraries or permissions.

## Documentation

The [draft V2 manual](docs/index.md) includes a simulated quickstart, an annotated
experiment, and component and utility guides. Install and preview it locally from
the repository root:

```sh
python3.13 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
```

Open the address printed by MkDocs, normally `http://127.0.0.1:8000/Auspex/`.
The documentation workflow runs only when manually triggered; pushes and pull
requests do not build or deploy the site. See [maintenance and deployment instructions](docs/maintaining.md).

## Funding

This software is based in part upon work supported by the Office of the Director
of National Intelligence (ODNI), Intelligence Advanced Research Projects
Activity (IARPA), via contract W911NF-14-C0089 and Army Research Office contract
No. W911NF-14-1-0114. The views and conclusions contained herein are those of
the authors and should not be interpreted as necessarily representing the
official policies or endorsements, either expressed or implied, of the ODNI,
IARPA, or the U.S. Government.

## Development

LLM tools assisted the development of Auspex V2. Responsibility for errors,
omissions, and unintended consequences remains with the humans involved in its
development.
