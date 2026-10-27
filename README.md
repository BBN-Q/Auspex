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
python -m pip install -e ".[visa,gui,tui]" # All three
```

The `visa` extra includes PyVISA and PyVISA-py; `gui` includes PySide6 and
pyqtgraph; `tui` includes Textual. Hardware interfaces may require additional
system libraries or permissions.

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
