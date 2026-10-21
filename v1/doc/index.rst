.. auspex documentation master file, created by
   sphinx-quickstart on Thu Sep 15 21:16:14 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Auspex Documentation
.. =======================

.. image:: images/Auspex-Logo-LowRes.png
   :width: 400 px
   :align: center

Introduction
************

Auspex, the *automated system for python-based experiments*, is a framework for performing laboratory measurements. Auspex was developed by a group that primarily performs measurements on superconducting qubits and magnetic memory elements, but its underpinnings are sufficiently general to allow for extension to arbitrary equipment and experiments. Using several layers of abstraction, we attempt to meet the following goals:

1. Instrument drivers should be easy to write.
2. Measurement code should be flexible and reusable.
3. Data acquisition and processing should be asynchronous and reconfigurable.
4. Experiments should be adaptive, not always pre-defined.
5. Experiments should be concerned with information density, and not limited by the convenience of rectilinear sweeps.

A number of inroads towards satisfying points (1) and (2) are made by utilizing metaprogramming to reduce boilerplate code in *Instrument* drivers and to provide a versatile framework for defining an *Experiment*. For (3) we make use of the python *asyncio* library to create a graph-based measurement *Filter* pipeline through which data passes to be processed, plotted, and written to file. For (4), we attempt to mitigate the sharp productivity hits associated with experimentors having to monitor and constantly tweak the parameters of an *Experiment*. This is done by creating a simple interface that allows *Sweeps* to refine themselves based on user-defined criterion functions. Finally, for (5) we build in "unstructured" sweeps that work with parameter tuples rather than "linspace" style ranges for each parameter. The combination of (4) and (5) allows us to take beautiful phase diagrams that require far fewer points than would be required in a rectilinear, non-adaptive scheme.

Installation & Requirements
***************************

Auspex can be cloned from GitHub::

	git clone https://github.com/BBN-Q/auspex.git

And subsequently installed using pip::

	cd auspex
	pip install -e .

Which will automatically fetch and install all of the requirements. If you are using an anaconda python distribution, some of the requirements should be install with *conda install* (like ruamel_yaml for example). The packages enumerated in *requirements.txt* are required by Auspex.

Qubit Experiments
*****************

Auspex is agnostic to the type of experiment being performed, but we include infrastructure for configuring and executing :ref:`qubit experiments <qubits>` using the gate-level `QGL <https://github.com/BBN-Q/QGL>`_ language. In this case, auspex relies on `bbndb <https://github.com/BBN-Q/bbndb>`_ as a database backend for sharing state and keeping track of configurations. Depending on the experiments being run, one may need to install a number of additional driver libraries.

If you're running on a system with a low file descriptor limit you may see a
`ulimit` error when trying to run or simulate experiments.  This will look like a
`too many files error` in python.  This stems from ZMQ asynchronously opening 
and closing a large number of files.  OSX has a default limit per notebook of 
256 open files.  You can easily change this number at the terminal before 
launching a notebook: `ulimit -n 4096` or put this line in your `.bash_prifile`.

Genealogy and Etymology
***********************

Auspex is a synonym for an *augur*, whose role was to interpret divine will through a variety of omens. While most researchers rightfully place their faiths in the scientific method, it is not uncommon to relate to the roles of the augur. Auspex incorporates concepts from BBN's *QLab* project as well as from the *pymeasure* project from Cornell University.

Contents:

.. toctree::
   :maxdepth: 1

   Instrument Drivers <instruments>
   Defining Experiments <experiments>
   Advanced Sweeps <sweeps>
   Qubit Experiments <qubits>
   Plotting Server <plotting>
   Full API <api/auspex>
