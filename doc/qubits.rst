.. _qubit_experiments:

Qubit Experiments
=================

Auspex and QGL comprise BBN's qubit measurement softwre stack. Both packages utilize the underlying database schema provided by bbndb that allows them to easily share state and allows the user to take advantage of versioned measurement configurations. 

Important Changes in the BBN-Q Software Ecosystem
*************************************************
There has been a recent change in how we do things:

1. Both Auspex and QGL now utilize the bbndb backend for configuration information. YAML has been completely dropped as it was too freeform and error prone.
2. HDF5 has been dropped as the datafile format for auspex and replaced with a simple numpy-backed binary format with metadata.
3. HDF5 has also been dropped as the sequence file format for our APS1 and APS2 arbitrary pulse sequencers, a simple binary format prevails here as well.
4. The plot server and client have been decoupled from the main auspex code and now are executed independently. They can even be run remotely!
5. Bokeh has been replaced with bqplot where possible, which has much better data throughput.

Tutorials
*********
The best way to gain experience is to follow through with these tutorials:

.. toctree::
   :maxdepth: 1

   Q1 Tutorial: Creating Channel Library <examples/Example-Config.ipynb>
   Q2 Tutorial: Saving/Loading Library Versions <examples/Example-Channel-Lib.ipynb>
   Q3 Tutorial: Using the Pipeline Manager <examples/Example-Filter-Pipeline.ipynb>
   Q4 Tutorial: Running a Basic Qubit Experiment <examples/Example-Experiments.ipynb>
   Q5 Tutorial: Adding Sweeps to Experiments <examples/Example-Sweeps.ipynb>
   Q6 Tutorial: Pulse Calibration <examples/Example-Calibrations.ipynb>
   Q7 Tutorial: Single Shot Fidelity <examples/Example-SingleShot-Fid.ipynb>
   Q8 Tutorial: Realistic Two Qubit Tuneup and Experiments <examples/Example-APS2-2Qubit.ipynb>

Instrument Drivers
******************
For `libaps2 <https://github.com/bbn-q/libaps2>`_, `libalazar <https://github.com/bbn-q/libalazar>`_, and `libx6  <https://github.com/bbn-q/libx6>`_, one should be able to *conda install -c bbn-q xxx* in order to obtain binary distributions of the relevant packages. Otherwise, one must obtain and build those libraries (according to their respective documentation), then make the shared library build products and any python packages available to Auspex by placing them on the path.

