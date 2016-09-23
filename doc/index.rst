.. pycontrol documentation master file, created by
   sphinx-quickstart on Thu Sep 15 21:16:14 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pycontrol Documentation
=======================

Introduction
************

Pycontrol is a python-based framework for performing laboratory measurements. The context of its development has resulted in a system that is tailored to measurements of superconducting qubits and magnetic memory elements, but its underpinnings are sufficiently general to allow for extension to arbitrary equipment and measurements. Using several layers of abstraction, we attempt to meet the following goals:

1. Instrument drivers should be easy to write.
2. Measurement code should be flexible and reusable.
3. Data acquisition and processing should be asynchronous and reconfigurable.
4. Experiments should be adaptive, not always pre-defined.
5. Experiments should be concerned with information density, and not limited by the convenience of rectilinear sweeps.

A number of inroads towards satisfying points (1) and (2) are made by utilizing metaprogramming to reduce boilerplate code in *Instrument* drivers and to provide a versatile framework for defining an *Experiment*. For (3) we make use of the python *asyncio* library to create a graph-based measurement *Filter* pipeline through which data passes to be processed, plotted, and written to file. For (4), we attempt to mitigate the sharp productivity hits associated with experimentors having to monitor and constantly tweak the parameters of an *Experiment*. This is done by creating a simple interface that allows *Sweeps* to refine themselves based on user-defined criterion functions. Finally, for (5) we build in "unstructured" sweeps that work with parameter tuples rather than "linspace" style ranges for each parameter. The combination of (4) and (5) allows us to take beautiful phase diagrams that require far fewer points than would be required in a rectilinear, non-adaptive scheme.


Installation & Requirements
***************************

Pycontrol can be cloned from GitHub::

	git clone https://github.com/BBN-Q/pycontrol.git

And subsequently installed using pip::

	cd pycontrol
	pip install -e .

Please refer to the :ref:`installation` section for full details.

Genealogy
*********

Pycontrol incorporates concepts from BBN's *QLab* project as well as from the *pymeasure* project.

Contents:

.. toctree::
   :maxdepth: 1

   Instrument Drivers <instruments>
   Defining Experiments <experiments>
   Advanced Sweeps <sweeps>
   Integration with PyQLab/Quince <integration>
   Full API <api/pycontrol>


