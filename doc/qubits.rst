.. _qubit_experiments:

Qubit Experiments
=================


Config Files
************

Auspex uses a `YAML <http://www.yaml.org>`_ configuration file to describe qubit experiments. In a departure from the previous *PyQLab* stack, the configuration is qubit centric, i.e.:

.. code-block:: yaml

   qubits:
     - name: q1
       measure:
           AWG: BBNAPS1 12
           trigger: BBNAPS1 12m1
           receiver: q1-RawSS
           generator: Holz1 
           autodyneFreq: 10e6
           pulseParams:
               amp: 1.0
               cutoff: 2.0
               length: 0.5e-06
               shapeFun: tanh
               sigma: 1e-9
       control:
           AWG: BBNAPS2 12
           generator: Holz2
           pulseParams:
               cutoff: 2.0
               length: 7e-8
               pi2Amp: 0.4
               piAmp: 0.8
               shapeFun: drag
               dragScaling: 0.0
               sigma: 5.0e-9

The control and measurement configurations are specified separately. If a generator is defined for either, Auspex infers that we are mixing up from a lower speed AWG. Otherwise, Auspex infers that direct synthesis is being performed. 

Auspex extends the YAML loader to provide an *!import* macro that can be employed as follows:

.. code-block:: yaml

    instruments: !include instruments.yml

in order to split configuration across multiple files. Auspex will try to repsect these macros, but pathological cases will probably fail. 

Instrument Drivers
******************

For `libaps2 <https://github.com/bbn-q/libaps2>`_, `libalazar <https://github.com/bbn-q/libalazar>`_, and `libx6  <https://github.com/bbn-q/libx6>`_, one should be able to *conda install -c bbn-q xxx* in order to obtain binary distributions of the relevant packages. Otherwise, one must obtain and build those libraries (according to their respective documentation), then make the shared library build products and any python packages available to Auspex by placing them on the path.

The Qubit Experiment Factory
****************************

:ref:`QubitExpFactory <qubitexpfactory>` reads in the configuration YAML flat files and contructs an Auspex *Experiment* from them. It also accepts a *meta_file*, generated directly by `QGL <https://github.com/BBN-Q/QGL>`_, that changes the experiment configuration to conform to a desired pulse sequence.


.. code-block:: python

    # Cavity Sweep
    from QGL import *
    from from auspex.exp_factory import QubitExpFactory
    q = QubitFactory("q1")
    exp = QubitExpFactory.create(PulsedSpec(q))
    exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))
    exp.run_sweeps()

Pulse Calibrations
******************

