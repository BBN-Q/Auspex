.. _qubit_experiments:

Qubit Experiments
=================

Auspex and QGL comprise BBN's qubit measurement system. Both packages utilize the underlying database schema provided by bbndb that allows them to easily share state and allows the user to take advantage of versioned measurement configurations. 

Tutorials
*********
The best way to gain experience is to follow through with these tutorials:

.. toctree::
   :maxdepth: 1

   Q1 Tutorial: Creating Channel Library <examples/Example-Q1.ipynb>
   Q2 Tutorial: Saving/Loading Library Versions <examples/Example-Q2.ipynb>
   Q3 Tutorial: Using the Pipeline Manager <examples/Example-Q3.ipynb>

The Filter Pipeline
*******************
Coming soon.

Pulse Calibration
*****************
Coming soon.

Automated Tuneup
****************
Coming soon.

Instrument Drivers
******************
For `libaps2 <https://github.com/bbn-q/libaps2>`_, `libalazar <https://github.com/bbn-q/libalazar>`_, and `libx6  <https://github.com/bbn-q/libx6>`_, one should be able to *conda install -c bbn-q xxx* in order to obtain binary distributions of the relevant packages. Otherwise, one must obtain and build those libraries (according to their respective documentation), then make the shared library build products and any python packages available to Auspex by placing them on the path.

