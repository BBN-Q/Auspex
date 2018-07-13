from auspex.log import logger
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
import bbndb

import asyncio

class QubitExperiment(Experiment):
    """Experiment with a specialized run method for qubit experiments run via the QubitExpFactory."""
    
    def add_connector(self, qubit):
        logger.debug(f"Adding {qubit.qubit_name} output connector to experiment.")
        oc = OutputConnector(name=qubit.qubit_name, parent=self)
        self._output_connectors[qubit.qubit_name] = oc
        self.output_connectors[qubit.qubit_name] = oc
        # self._output_connectors_by_qubit[qubit] = oc
        setattr(self, qubit.qubit_name, oc)
        return oc

    def init_instruments(self):
        for name, instr in self._instruments.items():
            instr_par = self.settings['instruments'][name]
            logger.debug("Setting instr %s with params %s.", name, instr_par)
            instr.set_all(instr_par)

        self.digitizers = [v for _, v in self._instruments.items() if "Digitizer" in v.instrument_type]
        self.awgs       = [v for _, v in self._instruments.items() if "AWG" in v.instrument_type]

        # Swap the master AWG so it is last in the list
        try:
            master_awg_idx = next(ct for ct,awg in enumerate(self.awgs) if 'master' in self.settings['instruments'][awg.name] and self.settings['instruments'][awg.name]['master'])
            self.awgs[-1], self.awgs[master_awg_idx] = self.awgs[master_awg_idx], self.awgs[-1]
        except:
            logger.warning("No AWG is specified as the master.")

        # attach digitizer stream sockets to output connectors
        for chan, dig in self.chan_to_dig.items():
            socket = dig.get_socket(chan)
            oc = self.chan_to_oc[chan]
            self.loop.add_reader(socket, dig.receive_data, chan, oc)

        if self.cw_mode:
            for awg in self.awgs:
                awg.run()

    def add_instrument_sweep(self, instrument, attribute, values):
        pass
   
    def add_qubit_sweep(self, qubit, measure_or_control, attribute, values):
        """
        Add a *ParameterSweep* to the experiment. Users specify a qubit property that auspex 
        will try to link back to the relevant instrument.
        (i.e. *q1 measure frequency* or *q2 control power*). For example::
            exp = QubitExpFactory.create(PulsedSpec(q))
            exp.add_qubit_sweep(q1, "measure", "frequency", np.linspace(6e9, 6.5e9, 500))
            exp.run_sweeps()
        """
        # Create the parameter
        param = FloatParameter()

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
                print(self.sources, name)
                # instr = experiment._instruments[name]
            else:
                # Direct synthesis
                pass


    def add_avg_sweep(self, num_averages):
        param = IntParameter()
        param.name = "sw_avg"
        setattr(self, param.name, param)
        self._parameters[param.name] = param
        self.add_sweep(param, range(num_averages))

    def shutdown_instruments(self):
        # remove socket readers
        if self.cw_mode:
            for awg in self.awgs:
                awg.stop()
        for chan, dig in self.chan_to_dig.items():
            socket = dig.get_socket(chan)
            self.loop.remove_reader(socket)
        for name, instr in self._instruments.items():
            instr.disconnect()

    async def run(self):
        """This is run for each step in a sweep."""
        for dig in self.digitizers:
            dig.acquire()
        await asyncio.sleep(0.75)
        if not self.cw_mode:
            for awg in self.awgs:
                awg.run()

        # Wait for all of the acquisitions to complete
        timeout = 10
        try:
            await asyncio.gather(*[dig.wait_for_acquisition(timeout) for dig in self.digitizers])
        except Exception as e:
            logger.error("Received exception %s in run loop. Bailing", repr(e))
            self.shutdown()
            sys.exit(0)

        for dig in self.digitizers:
            dig.stop()
        if not self.cw_mode:
            for awg in self.awgs:
                awg.stop()
