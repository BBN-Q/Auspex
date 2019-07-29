![auspex](doc/images/Auspex-Small.png)
<!-- [![build status](https://qiplab.bbn.com/ci/projects/1/status.png?ref=master)](https://qiplab.bbn.com/ci/projects/1?ref=master) -->
[![Build Status](https://travis-ci.org/BBN-Q/Auspex.svg?branch=master)](https://travis-ci.org/BBN-Q/Auspex) [![Documentation Status](https://readthedocs.org/projects/auspex/badge/?version=latest)](http://auspex.readthedocs.io/en/latest/?badge=latest)

Auspex, the automated system for python-based experiments, is a framework for performing laboratory measurements. Auspex was developed by a group that primarily performs measurements on superconducting qubits and magnetic memory elements, but its underpinnings are sufficiently general to allow for extension to arbitrary equipment and experiments. Using several layers of abstraction, we attempt to meet the following goals:

1. Instrument drivers should be easy to write.
1. Measurement code should be flexible and reusable.
1. Data acquisition and processing should be pipelined, asynchronous, and reconfigurable.
1. Experiments should be adaptive, not always pre-defined.

## Features ##

1. Easy driver specification using metaprogramming.
1. Fast multiprocessing pipeline for digitizer (or other) data.
1. Tight integration with QGL for quantum experiment definition.
1. Qubit calibration routines.
1. Run within Jupyter notebooks.
1. Easy pipeline specification and graphical display of pipelines.
1. Separate zmq/matplotlib plotting clients for local or remote use.
1. Custom high-performance file format (numpy mmaps) with full data axis descriptors and metadata.

Regrettably in-notebook plotting is difficult to implement for more than simple 1D sweeps. Notebook caching and
javascript interface limitations make returning large amounts of data impractical. Final plots
may be displayed 

## Documentation ##
Full documentation can be found at [readthedocs](http://auspex.readthedocs.io/en/latest/).

## Funding ##

This software is based in part upon work supported by the Office of the Director
of National Intelligence (ODNI), Intelligence Advanced Research Projects
Activity (IARPA), via contract W911NF-14-C0089 and Army Research Office contract
No. W911NF-14-1-0114. The views and conclusions contained herein are those of
the authors and should not be interpreted as necessarily representing the
official policies or endorsements, either expressed or implied, of the ODNI,
IARPA, or the U.S. Government.
