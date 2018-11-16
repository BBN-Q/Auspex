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


.. Config Files
.. ************

.. Auspex uses a `YAML <http://www.yaml.org>`_ configuration file to describe qubit experiments. By default, the measurement file is assumed to be in the location specified by the environment variable *BBN_MEAS_FILE*. Otherwise, the user can manually specify where the configuration file is located using the *meas_file* keyword argument of the :ref:`QubitExpFactory <qubitexpfactory>` *create()* and *run()* methods.

.. The config file, which is shared by QGL, must contain a few different blocks. First the *config* section specifies where the waveform files, integration kernels, and logs should reside:

.. .. code-block:: yaml

..   config:
..     AWGDir: /tmp/awg
..     KernelDir: /tmp/kern
..     LogDir: /tmp/alog

.. In a departure from the channel centric behavior of our legacy *PyQLab* stack, the configuration is qubit centric. The *qubit* definition section will resemble the following:

.. .. code-block:: yaml

..   qubits:
..     q1:
..       measure:
..         AWG: BBNAPS1 12
..         trigger: BBNAPS1 12m1
..         receiver: q1-IntegratedSS
..         generator: Holz1
..         autodyne_freq: 10000000.0
..         pulse_params:
..           amp: 1.0
..           cutoff: 2.0
..           length: 5.0e-07
..           shape_fun: tanh
..           sigma: 1.0e-09
..       control:
..         AWG: BBNAPS2 12
..         generator: Holz2
..         frequency: -49910002.0
..         pulse_params:
..           cutoff: 2.0
..           length: 7.0e-08
..           pi2Amp: 0.50045
..           piAmp: 1.0009
..           shape_fun: drag
..           drag_scaling: 0.0
..           sigma: 5.0e-09

.. The control and measurement configurations are specified separately. If a generator is defined for either, Auspex infers that we are mixing up from a lower speed AWG. Otherwise, Auspex infers that direct synthesis is being performed. 

.. The *instruments* section gives the instrument configuration parameters:

.. .. code-block:: yaml

..   instruments:
..     BBNAPS1:
..       type: APS2
..       master: true
..       slave_trig: 12m4
..       address: 192.168.5.20
..       seq_file: thing.h5
..       trigger_interval: 5.0e-06
..       trigger_source: Internal
..       delay: 0.0
..       tx_channels:
..         '12':
..           phase_skew: -11.73
..           amp_factor: 0.898
..           '1':
..             offset: 0.1
..             amplitude: 0.9
..           '2':
..             offset: 0.02
..             amplitude: 0.8
..       markers:
..         12m1:
..           delay: -5.0e-08
..         12m2:
..           delay: 0.0
..         12m3:
..           delay: 0.0
..         12m4:
..           delay: 0.0
..       enabled: true
..     BBNAPS2:
..       type: APS2
..       master: false
..       address: 192.168.5.21
..       seq_file: thing2.h5
..       trigger_interval: 5.0e-06
..       trigger_source: External
..       delay: 0.0
..       tx_channels:
..         '12':
..           phase_skew: 10
..           amp_factor: 0.898
..           '1':
..             offset: 0.10022
..             amplitude: 0.9
..           '2':
..             offset: 0.020220000000000002
..             amplitude: 0.8
..       markers:
..         12m1:
..           delay: -5.0e-08
..         12m2:
..           delay: 0.0
..         12m3:
..           delay: 0.0
..         12m4:
..           delay: 0.0
..       enabled: true
..     X6-1:
..       type: X6
..       address: 0
..       acquire_mode: digitizer
..       gen_fake_data: true
..       ideal_data: cal_fake_data
..       reference: external
..       record_length: 1024
..       nbr_segments: 1
..       nbr_round_robins: 20
..       rx_channels:
..         '1':
..         '2':
..       streams: [raw, result1, result2]
..       enabled: true
..       exp_step: 0
..     Holz1:
..       type: HolzworthHS9000
..       address: HS9004A-009-1
..       power: -10
..       frequency: 6000000000.0
..       enabled: true
..     Holz2:
..       type: HolzworthHS9000
..       address: HS9004A-009-2
..       power: -10
..       frequency: 5000090023.0
..       enabled: true

.. Note how the APS2 devices are defined. Each instrument *should* (have patience) possess the *yaml_template* class property that gives an example of the yaml configuration that can be found by running, e.g.:

.. .. code-block:: python
  
..   from auspex.instruments import APS2
..   APS2.yaml_template 

.. Also, note that the instruments referenced in the *qubits* section are defined in the *instruments* section. The *filter* pipeline, which controls the processing of data, can be defined as follows:

.. .. code-block:: yaml

..   filters:
..     q1-RawSS:
..       type: X6StreamSelector
..       source: X6-1
..       stream_type: Raw
..       channel: 1
..       dsp_channel: 1
..       enabled: true
..     q1-IntegratedSS:
..       type: X6StreamSelector
..       source: X6-1
..       stream_type: Integrated
..       channel: 1
..       dsp_channel: 0
..       kernel: np.ones(1024, dtype=np.float64)
..       enabled: true
..     Demod-q1:
..       type: Channelizer
..       source: q1-RawSS
..       decimation_factor: 4
..       frequency: 10000000.0
..       bandwidth: 5000000.0
..       enabled: true
..     Int-q1:
..       type: KernelIntegrator
..       source: Demod-q1
..       box_car_start: 5.0e-07
..       box_car_stop: 9.0e-07
..       enabled: true
..     avg-q1:
..       type: Averager
..       source: Int-q1
..       axis: round_robins
..       enabled: true
..     avg-q1-int:
..       type: Averager
..       source: q1-IntegratedSS
..       axis: round_robins
..       enabled: true
..     final-avg-buff:
..       type: DataBuffer
..       source: avg-q1 final_average
..       enabled: false
..     final-avgint-buff:
..       type: DataBuffer
..       source: avg-q1-int final_average
..       enabled: false
..     partial-avg-buff:
..       type: DataBuffer
..       source: avg-q1 partial_average
..       enabled: false
..     q1-IntPlot:
..       type: Plotter
..       source: avg-q1 final_average
..       plot_dims: 1
..       plot_mode: real/imag
..       enabled: false
..     q1-DirectIntPlot:
..       type: Plotter
..       source: avg-q1-int final_average
..       plot_dims: 1
..       plot_mode: real/imag
..       enabled: false
..     q1-DirectIntPlot-unroll:
..       type: Plotter
..       source: q1-IntegratedSS final_average
..       plot_dims: 0
..       plot_mode: real/imag
..       enabled: false
..     q1-WriteToHDF5:
..       source: avg-q1-int final_average
..       enabled: true
..       compression: true
..       type: WriteToHDF5
..       filename: .\test
..       groupname: main
..       add_date: false
..       save_settings: false

.. **However**, we advise that the user not directly edit the filter section when possible. Our GUI node editor `Quince <https://github.com/bbn-q/quince>`_ can be used to graphically edit the filter pipeline, and can be easily launched from the python environment by running. 

.. .. code-block:: python

..     from from auspex.exp_factory import quince
..     quince() # takes an optional argument giving the measurement file

.. In order to split configuration across multiple files, Auspex extends the YAML loader to provide an *!import* macro that can be employed as follows:

.. .. code-block:: yaml

..   instruments: !include instruments.yml

.. Auspex will try to repsect these macros, but pathological cases will probably fail.

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
    cl = ChannelLibrary()
    q = QubitFactory("q1")
    exp = QubitExpFactory.create(PulsedSpec(q))
    exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))
    exp.run_sweeps()

Pulse Calibrations
******************

To be added.
