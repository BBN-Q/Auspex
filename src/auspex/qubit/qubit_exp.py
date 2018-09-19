from auspex.log import logger
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
import bbndb

import sys
import os
if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from threading import Thread as Process
    from threading import Event
else:
    from multiprocessing import Process
    from multiprocessing import Event

import time

class QubitExperiment(Experiment):
    """Experiment with a specialized run method for qubit experiments run via the QubitExpFactory."""

    def add_connector(self, qubit):
        logger.debug(f"Adding {qubit.qubit_name} output connector to experiment.")
        oc = OutputConnector(name=qubit.qubit_name, parent=self)
        self.output_connectors[qubit.qubit_name] = oc
        setattr(self, qubit.qubit_name, oc)
        return oc

    def init_instruments(self):
        self.cw_mode = False

        for name, instr in self._instruments.items():
            # Configure with dictionary from the instrument proxy
            if hasattr(instr, "configure_with_proxy"):
                instr.configure_with_proxy(instr.proxy_obj)
            else:
                instr.configure_with_dict(instr.proxy_obj.to_dict())

        self.digitizers = [v for _, v in self._instruments.items() if "Digitizer" in v.instrument_type]
        self.awgs       = [v for _, v in self._instruments.items() if "AWG" in v.instrument_type]

        # Swap the master AWG so it is last in the list
        try:
            master_awg_idx = next(ct for ct,awg in enumerate(self.awgs) if awg.master)
            self.awgs[-1], self.awgs[master_awg_idx] = self.awgs[master_awg_idx], self.awgs[-1]
        except:
            logger.warning("No AWG is specified as the master.")

        # Start socket listening processes, store as keys in a dictionary with exit commands as values
        self.dig_listeners = {}
        for chan, dig in self.chan_to_dig.items():
            socket = dig.get_socket(chan)
            oc = self.chan_to_oc[chan]
            # self.loop.add_reader(socket, dig.receive_data, chan, oc)
            exit = Event()
            self.dig_listeners[Process(target=dig.receive_data, args=(chan, oc, exit))] = exit
        for listener in self.dig_listeners.keys():
            listener.start()

        if self.cw_mode:
            for awg in self.awgs:
                awg.run()

    def add_instrument_sweep(self, instrument, attribute, values):
        pass

    def add_qubit_sweep(self, qubit, measure_or_control, attribute, values):
        """
        Add a *ParameterSweep* to the experiment. Users specify a qubit property that auspex
        will try to link back to the relevant instrument. For example::
            exp = QubitExpFactory.create(PulsedSpec(q1))
            exp.add_qubit_sweep(q1, "measure", "frequency", np.linspace(6e9, 6.5e9, 500))
            exp.run_sweeps()
        """
        param = FloatParameter() # Create the parameter

        if measure_or_control not in ["measure", "control"]:
            raise ValueError(f"Cannot add sweep for something other than measure or control properties of {qubit}")

        if measure_or_control == "measure":
            logger.debug(f"Sweeping {qubit} measurement")
            thing = list(filter(lambda m: m.label=="M-"+qubit.label, self.measurements))
            if len(thing) > 1:
                raise ValueError(f"Found more than one measurement for {qubit}")
            thing = thing[0]
        elif measure_or_control == "control":
            logger.debug(f"Sweeping {qubit} control")
            thing = qubit

        if attribute == "frequency":
            if thing.phys_chan.generator:
                # Mixed up to final frequency
                name  = thing.phys_chan.generator.label
                instr = list(filter(lambda x: x.name == name, self._instruments.values()))[0]
                method = None
            else:
                # Direct synthesis
                name  = thing.phys_chan.awg.label
                instr = list(filter(lambda x: x.name == name, self._instruments.values()))[0]
                def method(value, channel=chan, instr=instr, prop=prop.lower()):
                    # e.g. keysight.set_amplitude("ch1", 0.5)
                    getattr(instr, "set_"+prop)(chan, value)

        if method:
            # Custom method
            param.assign_method(method)
        else:
            # Get method by name
            if hasattr(instr, "set_"+attribute):
                param.assign_method(getattr(instr, "set_"+attribute)) # Couple the parameter to the instrument
            else:
                raise ValueError("The instrument {} has no method {}".format(name, "set_"+attribute))
        # param.instr_tree = [instr.name, attribute] #TODO: extend tree to endpoint
        self.add_sweep(param, values) # Create the requested sweep on this parameter

    def add_avg_sweep(self, num_averages):
        param = IntParameter()
        param.name = "sw_avg"
        setattr(self, param.name, param)
        self._parameters[param.name] = param
        self.add_sweep(param, range(num_averages))

    def shutdown_instruments(self):
        # remove socket listeners
        for listener, exit in self.dig_listeners.items():
            exit.set()
            listener.join()
        if self.cw_mode:
            for awg in self.awgs:
                awg.stop()
        for chan, dig in self.chan_to_dig.items():
            socket = dig.get_socket(chan)
            # self.loop.remove_reader(socket)
        for instr in self.instruments:
            instr.disconnect()

    def run(self):
        print("In Run")
        # Begin acquisition before enabling the AWGs
        for dig in self.digitizers:
            dig.acquire()

        # Start the AWGs
        if not self.cw_mode:
            for awg in self.awgs:
                awg.run()

        # Wait for all of the acquisitions to complete
        timeout = 20
        for dig in self.digitizers:
            dig.wait_for_acquisition(timeout, self.chan_to_oc.values())

        # Bring everything to a stop
        for dig in self.digitizers:
            dig.stop()

        # Stop the AWGs
        if not self.cw_mode:
            for awg in self.awgs:
                awg.stop()
