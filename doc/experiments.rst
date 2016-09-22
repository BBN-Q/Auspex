Experiment Documentation
========================

Scripting
*********

Instantiating a few *Instrument* classes as decribed in the :ref:`relevant documentation <instruments>` provides us with an environment sufficient to perform any sort of measurement. Let us revisit our simple example with a few added instruments, and also add a few software averages to our measurement.::

    pspl  = Picosecond10070A("GPIB0::24::INSTR") # Pulse generator
    mag   = AMI430("192.168.5.109")              # Magnet controller
    keith = Keithley2400("GPIB0::25::INSTR")     # Source meter

    pspl.amplitude = 0.944 # V
    mag.field      = 0.010 # T    
    time.sleep(0.1) # Stableize

    resistance_values = []
    for dur in 1e-9*np.arange(1, 11, 0.05):
        pspl.duration = dur
        pspl.trigger()
        time.sleep(0.05)
        resistance_values.append([keith.resistance for i in range(5)])

    avg_res_vals = np.mean(resistance, axis=0)

We have omitted a number of configuration commands for brevity. The above script works perfectly well if we always perform the exact same measurement, i.e. we hold the field and pulse amplitude fixed but vary its duration. This is normally an unrealistic restriction, since the experimentor more often than not will want to repeat the same fundamental measurement for any numbe of sweep conditions.

Defining Experiments
********************

Therefore, we recommend that users package their measurements into *Experiments*, which provide the opportunity for re-use.::

    class SwitchingExperiment(Experiment):
        # Control parameters
        field          = FloatParameter(default=0.0, unit="T")
        pulse_duration = FloatParameter(default=5.0e-9, unit="s")
        pulse_voltage  = FloatParameter(default=0.1, unit="V")

        # Output data connectors
        resistance = OutputConnector()

        # Constants
        samples = 5

        # Instruments
        pspl  = Picosecond10070A("GPIB0::24::INSTR")
        mag   = AMI430("192.168.5.109")
        keith = Keithley2400("GPIB0::25::INSTR")

        def init_instruments(self):
            # Instrument initialization goes here
            
            # Assign methods
            self.field.assign_method(self.mag.set_field)
            self.pulse_duration.assign_method(self.pspl.set_duration)
            self.pulse_voltage.assign_method(self.pspl.set_voltage)

            # Create hooks for relevant delays
            self.pulse_duration.add_post_push_hook(lambda: time.sleep(0.05))
            self.pulse_voltage.add_post_push_hook(lambda: time.sleep(0.05))
            self.field.add_post_push_hook(lambda: time.sleep(0.1))

        def init_streams(self):
            self.resistance.add_axis(DataAxis("samples",
                                              range(self.samples)))

        async def run(self):
            pspl.trigger()
            await self.resistance.push(keith.resistance)

Here the control parameters, output data connectors, and the central measurement have crystallized into separate entities. To run the same experiment as was performed above, we must add a *sweep* to the experiment,::

    # Define a 1D sweep
    exp = SwitchingExperiment()
    exp.add_sweep(exp.pulse_duration, 1e-9*np.arange(1, 11, 0.05))
    exp.run_sweeps()

but we can at this point sweep any *Parameter* or a combination thereof: ::

    # Define a 2D sweep
    exp = SwitchingExperiment()
    exp.add_sweep(exp.field, np.arange(-0.01, 0.015, 0.005))
    exp.add_sweep(exp.pulse_voltage, np.linspace(0.1, 1.0, 20))
    exp.run_sweeps()

These sweeps can be based on *Parameter* tuples in order to accomodate non-rectilinear sweeps, and can be made adaptive by specifying convergence criteria that can modifying the sweeps on the fly. Full documentation is provided here. The time spent writing a full *Experiment* often pays dividends in terms of flexibility.


The Measurement Pipeline
************************

The central ``run`` method of an *Experiment* should not need to worry about file IO and plotting, nor should we bake common analysis routines (filtering, plotting, etc.) into the code that is only responsible for taking data. Therefore, pycontrol relegates these tasks to the measurement pipeline, which provides dataflow such as that in the image below.

.. figure:: images/ExperimentFlow.png
   :align: center

   An example of measurement dataflow starting from the *Experiment* at left.

Connectors and Streams
######################

*OutputConnectors* are "ports" on the experiments through which all measurement data flow occurs. A single *OutputConnector* can send data to any number of desinations (covered in the next section), for each of which 

In the *SwitchingExperiment* class above, the ``init_streams`` method assembles a *DataStreamDescriptor* and adds a *DataAxis* to it. In this way, we establish the dimensions of our data. When adding an additional sweep, a *SweepAxis* is automatically appended to the *DataStreamDescriptors* of any *OutputConnectors*.::



    wr = WriteToHDF5("datafile.h5")



Running Experiments in Jupyter Notebooks
****************************************

You should do this.


