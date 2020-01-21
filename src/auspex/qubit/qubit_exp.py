from auspex.log import logger
from auspex.config import isnotebook
from auspex.experiment import Experiment, FloatParameter
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
from auspex.instruments import instrument_map
import auspex.filters
import bbndb
import numpy as np

import sys
import os
if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from threading import Thread as Process
    from threading import Event
else:
    from multiprocessing import Process
    from multiprocessing import Event
    from multiprocessing import Value

from . import pipeline
import time
import datetime
import json

stream_hierarchy = [
    bbndb.auspex.Demodulate,
    bbndb.auspex.Integrate,
    bbndb.auspex.Average,
    bbndb.auspex.OutputProxy
]
filter_map = {
    bbndb.auspex.Demodulate: auspex.filters.Channelizer,
    bbndb.auspex.Average: auspex.filters.Averager,
    bbndb.auspex.Framer: auspex.filters.Framer,
    bbndb.auspex.Integrate: auspex.filters.KernelIntegrator,
    bbndb.auspex.Write: auspex.filters.WriteToFile,
    bbndb.auspex.Buffer: auspex.filters.DataBuffer,
    bbndb.auspex.Display: auspex.filters.Plotter,
    bbndb.auspex.FidelityKernel: auspex.filters.SingleShotMeasurement
}
stream_sel_map = auspex.filters.stream_sel_map

class QubitExperiment(Experiment):
    """Create an `Experiment` with specialized config and run methods for qubit experiments.

    Parameters:
        meta_file (string)
            The filename of the QGL metainfo (*.json) corresponding to the desired
            experiment.
        averages (int)
            The number of shots to take. Results are only actually averaged
            if an `Averager` node is present in the processing pipeline.
        exp_name (string)
            Name of experiment. Used by any writers in pipeline to pick a data container name.

        kwargs
            Additional keyword arguments passed to the base Auspex `Experiment`
            class.
    Returns:
        experiment instance (`Experiment`)
            Returns the initialized Auspex `Experiment`.

    Examples:
        Creating a simple experiment.

        >>> mf = RabiAmp(q1, [-1,0,1])
        >>> exp = QubitExperiment(mf, averages=500)

    """

    def __init__(self, meta_file, averages=100, exp_name=None, save_chandb=True, **kwargs):
        super(QubitExperiment, self).__init__(**kwargs)

        if not pipeline.pipelineMgr:
            raise Exception("Could not find pipeline manager, have you declared one using PipelineManager()?")

        self.cw_mode = False
        self.add_date = True # add date to data files?


        self.name = exp_name

        self.outputs_by_qubit = {}
        self.progressbars = None
        self.save_chandb = save_chandb

        self.create_from_meta(meta_file, averages)

    def create_from_meta(self, meta_file, averages):
        """Method called during creation. Implementing a subclass of `QubitExperiment` this method
        may be overridden to provide additional functionality. However, this is a complex method, and
        it is recommended that the user instead override the `modify_graph` method to provide
        custom subclass behavior.
        """
        try:
            with open(meta_file, 'r') as FID:
                meta_info = json.load(FID)
        except:
            raise Exception(f"Could note process meta info from file {meta_file}")

        # Load ChannelLibrary and database information
        db_provider      = meta_info['database_info']['db_provider']
        db_resource_name = meta_info['database_info']['db_resource_name']
        library_name     = meta_info['database_info']['library_name']
        library_id       = meta_info['database_info']['library_id']

        # Respect separate sessions for channel library and pipeline
        self.cl_session = bbndb.get_cl_session()
        self.pl_session = bbndb.get_pl_session()

        # Load the channel library by ID
        self.chan_db     = self.cl_session.query(bbndb.qgl.ChannelDatabase).filter_by(id=library_id).first()
        all_channels     = self.chan_db.channels
        all_generators   = self.chan_db.generators
        all_transmitters = self.chan_db.transmitters
        all_receivers    = self.chan_db.receivers
        all_transceivers = self.chan_db.transceivers
        all_qubits       = [c for c in all_channels if isinstance(c, bbndb.qgl.Qubit)]
        all_measurements = [c for c in all_channels if isinstance(c, bbndb.qgl.Measurement)]
        # Restrict to current qubits, channels, etc. involved in this actual experiment
        self.controlled_qubits = [c for c in self.chan_db.channels if c.label in meta_info["qubits"]]
        self.measurements      = [c for c in self.chan_db.channels if c.label in meta_info["measurements"]]
        self.measured_qubits   = [c for c in self.chan_db.channels if "M-"+c.label in meta_info["measurements"]]
        if 'edges' in meta_info:
            self.edges             = [c for c in self.chan_db.channels if c.label in meta_info["edges"]]
        else:
            self.edges = []
        self.phys_chans        = list(set([e.phys_chan for e in self.controlled_qubits + self.measurements + self.edges]))
        self.receiver_chans    = list(set([e.receiver_chan for e in self.measurements]))
        self.slave_trigs       = [c for c in self.chan_db.channels if c.label == 'slave_trig']
        self.trig_chans        = list(set([e.trig_chan.phys_chan for e in self.measurements])) + [c.phys_chan for c in self.slave_trigs]
        self.transmitters      = list(set([e.phys_chan.transmitter for e in self.controlled_qubits + self.measurements + self.edges + self.slave_trigs]))
        self.receivers         = list(set([e.receiver_chan.receiver for e in self.measurements]))
        self.generators        = list(set([q.phys_chan.generator for q in self.measured_qubits + self.controlled_qubits + self.measurements if q.phys_chan.generator]))
        self.qubits_by_name    = {q.label: q for q in self.measured_qubits + self.controlled_qubits}

        # Load the relevant stream selectors from the pipeline.
        self.stream_selectors = pipeline.pipelineMgr.get_current_stream_selectors()
        if len(self.stream_selectors) == 0:
            raise Exception("No filter pipeline has been created. You can try running the create_default_pipeline() method of the Pipeline Manager")
        org_stream_selectors = self.stream_selectors
        for ss in org_stream_selectors:
            labels = ss.label.split('-')
            for l in labels:
                if l in self.qubits_by_name.keys() and ss not in self.stream_selectors:
                    self.stream_selectors.append(ss)
                    continue

        # Locate transmitters relying on processors
        self.transceivers = list(set([t.transceiver for t in self.transmitters + self.receivers if t.transceiver]))
        self.processors = list(set([p for t in self.transceivers for p in t.processors]))

        # Determine if the digitizer trigger lives on another transmitter that isn't included already
        self.transmitters = list(set([mq.measure_chan.trig_chan.phys_chan.transmitter for mq in self.measured_qubits] + self.transmitters))

        # The exception being any instruments that are declared as standalone
        self.all_standalone = [i for i in self.chan_db.all_instruments() if i.standalone and i not in self.transmitters + self.receivers + self.generators]

        # In case we need to access more detailed foundational information
        self.factory = self

        # If no pipeline is defined, assumed we want to generate it automatically
        if not pipeline.pipelineMgr.meas_graph:
            raise Exception("No pipeline has been created, do so automatically using exp_factory.create_default_pipeline()")
            #self.create_default_pipeline(self.measured_qubits)

        # Add the waveform file info to the qubits
        output_chans = self.transmitters + self.transceivers + self.phys_chans + self.trig_chans
        for xmit, fname in meta_info['instruments'].items():
            awg = [c for c in output_chans if c.label==xmit][0]
            awg.sequence_file = fname

        # Construct the DataAxis from the meta_info
        desc = meta_info["axis_descriptor"]
        data_axis = desc[0] # Data will always be the first axis
        
        # ovverride data axis with repeated number of segments
        if hasattr(self, "repeats") and self.repeats is not None:
            data_axis['points'] = np.tile(data_axis['points'], self.repeats)

        # Search for calibration axis, i.e., metadata
        axis_names = [d['name'] for d in desc]
        if 'calibration' in axis_names:
            meta_axis = desc[axis_names.index('calibration')]
            # There should be metadata for each cal describing what it is
            if len(desc)>1:
                metadata = ['data']*len(data_axis['points']) + meta_axis['points']
                # Pad the data axis with dummy equidistant x-points for the extra calibration points
                avg_step = (data_axis['points'][-1] - data_axis['points'][0])/(len(data_axis['points'])-1)
                points = np.append(data_axis['points'], data_axis['points'][-1] + (np.arange(len(meta_axis['points']))+1)*avg_step)
            else:
                metadata = meta_axis['points'] # data may consist of calibration points only
                points = np.arange(len(metadata)) # dummy axis for plotting purposes
            # If there's only one segment we can ignore this axis
            if len(points) > 1:
                self.segment_axis = DataAxis(data_axis['name'], points, unit=data_axis['unit'], metadata=metadata)
        else:
            # No calibration data, just add a segment axis as long as there is more than one segment
            if len(data_axis['points']) > 1:
                self.segment_axis = DataAxis(data_axis['name'], data_axis['points'], unit=data_axis['unit'])

        # Build a mapping of qubits to self.receivers, construct qubit proxies
        # We map by the unique database ID since that is much safer
        receiver_chans_by_qubit_label = {}
        for m in self.measurements:
            q = [c for c in self.chan_db.channels if c.label==m.label[2:]][0]
            receiver_chans_by_qubit_label[q.label] = m.receiver_chan

        # Now a pipeline exists, so we create Auspex filters from the proxy filters in the db
        self.proxy_to_filter          = {}
        self.filters                  = []
        self.connector_by_sel         = {}
        self.chan_to_dig              = {}
        self.chan_to_oc               = {}
        self.qubit_to_dig             = {}
        self.qubits_by_output         = {}
        self.proxy_name_to_instrument = {}

        # Create microwave sources and receiver instruments from the database objects.
        # We configure the self.receivers later after adding channels.
        self.instrument_proxies = self.generators + self.receivers + self.transmitters + self.transceivers + self.all_standalone + self.processors
        for t in self.transceivers:
            if t.initialize_separately:
                self.instrument_proxies.remove(t)
            else:
                for el in t.transmitters + t.receivers:
                    self.instrument_proxies.remove(el)

        self.instruments = []
        for instrument in self.instrument_proxies:
            if (hasattr(instrument, 'serial_port') and
                instrument.serial_port is not None and
                hasattr(instrument, 'dac') and
                instrument.dac is not None):
                address = (instrument.address, instrument.serial_port, instrument.dac)
            else:
                address = instrument.address
            instr = instrument_map[instrument.model](address, instrument.label) # Instantiate
            # For easy lookup
            instr.proxy_obj = instrument
            
            instrument._locked = False
            instrument.instr = instr # This shouldn't be relied upon
            instrument._locked = True

            self.proxy_name_to_instrument[instrument.label] = instr

            # Add to the experiment's instrument list
            self._instruments[instrument.label] = instr
            self.instruments.append(instr)
            # Add to class dictionary for convenience
            if not hasattr(self, instrument.label):
                setattr(self, instrument.label, instr)

        mq_all_stream_sels = []
        for mq in self.measured_qubits:

            # Stream selectors from the pipeline database:
            # These contain all information except for the physical channel
            mq_stream_sels = [ss for ss in self.stream_selectors if mq.label in ss.label.split("-") and ss not in mq_all_stream_sels]
            mq_all_stream_sels.append(mq_stream_sels)

            # The receiver channel only specifies the physical channel
            rcv = receiver_chans_by_qubit_label[mq.label]

            # Create the auspex stream selectors
            transcvr = rcv.receiver.transceiver
            if transcvr is not None and transcvr.initialize_separately == False:
                dig = rcv.receiver.transceiver
                stream_sel_class = stream_sel_map[rcv.receiver.stream_sel]
            else:
                dig = rcv.receiver
                stream_sel_class = stream_sel_map[dig.stream_sel]

            for mq_stream_sel in mq_stream_sels:
                auspex_stream_sel = stream_sel_class(name=f"{rcv.label}-{mq_stream_sel.stream_type}-stream_sel")
                mq_stream_sel.channel = rcv.channel
                auspex_stream_sel.configure_with_proxy(mq_stream_sel)
                auspex_stream_sel.receiver = auspex_stream_sel.proxy = mq_stream_sel

                # Construct the channel from the receiver channel
                channel = auspex_stream_sel.get_channel(mq_stream_sel)
                # Manually set the physical channel
                channel.phys_channel = rcv.channel

                # Get the base descriptor from the channel
                descriptor = auspex_stream_sel.get_descriptor(mq_stream_sel, rcv)

                # Update the descriptor based on the number of segments
                # The segment axis should already be defined if the sequence
                # is greater than length 1
                if hasattr(self, "segment_axis"):
                    descriptor.add_axis(self.segment_axis)

                # Add averaging if necessary
                if averages > 1:
                    descriptor.add_axis(DataAxis("averages", range(averages)))

                # Add the output connectors to the experiment and set their base descriptor
                self.connector_by_sel[mq_stream_sel] = self.add_connector(mq_stream_sel)
                self.connector_by_sel[mq_stream_sel].set_descriptor(descriptor)

                # Add the channel to the instrument
                dig.instr.add_channel(channel)
                self.chan_to_dig[channel] = dig.instr
                self.chan_to_oc [channel] = self.connector_by_sel[mq_stream_sel]
                self.qubit_to_dig[mq.id]  = dig

        # Find the number of self.measurements
        segments_per_dig = {receiver_chan.receiver: meta_info["receivers"][receiver_chan.label] for receiver_chan in self.receiver_chans
                                                         if receiver_chan.label in meta_info["receivers"].keys()}

        # Configure receiver instruments from the database objects
        # this must be done after adding channels.
        for dig in self.receivers:
            if dig.transceiver is not None and transcvr.initialize_separately == False:
                dig.transceiver.number_averages = averages
                dig.transceiver.number_waveforms = 1
                dig.transceiver.number_segments = segments_per_dig[dig]
            else:
                dig.number_averages  = averages
                dig.number_waveforms = 1
                dig.number_segments  = segments_per_dig[dig]
                dig.instr.proxy_obj  = dig

        # Restrict the graph to the relevant qubits
        self.measured_qubit_names = [q.label for q in self.measured_qubits]
        self.pl_session.commit()

        # Any modifications to be done by subclasses, just a passthrough here
        self.modified_graph = self.modify_graph(pipeline.pipelineMgr.meas_graph)

        # Compartmentalize the instantiation
        self.instantiate_filters(self.modified_graph)

    def is_in_measured_qubit_names(self,qubit_name):
        labels = []
        if qubit_name is not None:
            labels = qubit_name.split('-')
        for l in labels:
            if l in self.measured_qubit_names:
                return True
        return False

    def instantiate_filters(self, graph):
        # Configure the individual filter nodes
        for _, dat in graph.nodes(data=True):
            node = dat['node_obj']
            if isinstance(node, bbndb.auspex.FilterProxy):
                if node.qubit_name in self.measured_qubit_names:
                    new_filt = filter_map[type(node)]()
                    new_filt.configure_with_proxy(node)
                    new_filt.proxy = node
                    self.filters.append(new_filt)
                    self.proxy_to_filter[node] = new_filt
                    if isinstance(node, bbndb.auspex.OutputProxy):
                        self.qubits_by_output[new_filt] = node.qubit_name

        # Connect the filters together
        graph_edges = []
        self.pl_session.commit()
        for l1, l2 in graph.edges():
            node1, node2 = graph.nodes[l1]['node_obj'], graph.nodes[l2]['node_obj']
            if (self.is_in_measured_qubit_names(node1.qubit_name) or self.is_in_measured_qubit_names(node1.label)) and self.is_in_measured_qubit_names(node2.qubit_name):
                if isinstance(node1, bbndb.auspex.FilterProxy):
                    filt1 = self.proxy_to_filter[node1]
                    oc   = filt1.output_connectors[graph[l1][l2]["connector_out"]]
                elif isinstance(node1, bbndb.auspex.StreamSelect):
                    oc   = self.connector_by_sel[node1]
                filt2 = self.proxy_to_filter[node2]
                ic   = filt2.input_connectors[graph[l1][l2]["connector_in"]]
                graph_edges.append([oc, ic])

        # Define the experiment graph
        self.set_graph(graph_edges)

    def modify_graph(self, graph):
        """Method called near the end of `create_from_meta` to allow custom manipulation of the filter
        pipeline. For example, `CalibrationExperiment` implements a version of `modify_graph` that
        selectively removes portions of the graph and creates buffers as needed to perform the desired
        calibrations on specific qubits.
        """
        return graph

    def set_fake_data(self, digitizer_proxy, ideal_data, increment=False, random_mag=0.1):
        """Enabled and use the fake data interface for digitizers in order that auspex can
        be run without hardware.

        Parameters:
            digitizer_proxy (bbndb `Receiver` instance)
                The digitizer instrument proxy to be used for fake data generation.
            ideal_data (numpy array)
                The actual data to be used. If `increment` is False, a 1D array with a single value
                per segment is used. The digitizer drivers automatical convert to a integrated, demodulated,
                or raw signal depending on the stream type being used. If `increment` is True, then this may be a
                2D array, which is incremented through to emulate sweeps such a qubit measurement frequency sweep.
            increment (boolean)
                Whether or not to step through a 2D data array after to incorporate extra sweeps. The behavior is
                defined above.

        Examples:
            Make sure to set auspex dummy mode at import time.

            >>> import auspex.config as config
            >>> config.auspex_dummy_mode = True
            >>> # Configure channels and pipelines here
            >>> amps = np.linspace(-1,1,51)
            >>> exp = QubitExperiment(RabiAmp(q1,amps),averages=50)
            >>> exp.set_fake_data(digitizer_1, np.cos(np.linspace(0, 2*np.pi,51)))
            >>> exp.run_sweeps()

        """
        auspex_instr = self.proxy_name_to_instrument[digitizer_proxy.label]
        auspex_instr.ideal_data = ideal_data
        auspex_instr.increment_ideal_data = increment
        auspex_instr.gen_fake_data = True
        auspex_instr.fake_data_random_mag = random_mag

    def clear_fake_data(self, digitizer_proxy):
        """Disable using fake data interface for a digitizer. Take note that dummy mode may
        still be active.

        Parameters:
            digitizer_proxy (bbndb `Receiver` instance)
                The digitizer instrument proxy to be used for fake data generation.
        """
        auspex_instr = self.proxy_name_to_instrument[digitizer_proxy.label]
        auspex_instr.ideal_data = ideal_data
        auspex_instr.gen_fake_data = False

    def add_connector(self, stream_selector):
        name = stream_selector.qubit_name+'-'+stream_selector.stream_type
        logger.debug(f"Adding {name} output connector to experiment.")
        oc = OutputConnector(name=name, parent=self)
        self.output_connectors[name] = oc
        setattr(self, name, oc)
        return oc

    def init_instruments(self):
        for name, instr in self._instruments.items():
            instr.configure_with_proxy(instr.proxy_obj)

        self.digitizers = [v for _, v in self._instruments.items() if "Digitizer" in v.instrument_type]
        self.awgs       = [v for _, v in self._instruments.items() if "AWG" in v.instrument_type]
        # Swap the master AWG so it is last in the list
        try:
            master_awg_idx = next(ct for ct,awg in enumerate(self.awgs) if awg.master)
            self.awgs[-1], self.awgs[master_awg_idx] = self.awgs[master_awg_idx], self.awgs[-1]
        except:
            logger.warning("No AWG is specified as the master.")

        for gen_proxy in self.generators:
            gen_proxy.instr.output = True

        # Start socket listening processes, store as keys in a dictionary with exit commands as values
        self.dig_listeners = {}
        ready = Value('i', 0)
        self.dig_run  = Event()
        self.dig_exit = Event()
        for chan, dig in self.chan_to_dig.items():
            socket = dig.get_socket(chan)
            oc = self.chan_to_oc[chan]
            p = Process(target=dig.receive_data, args=(chan, oc, self.dig_exit, ready, self.dig_run))
            self.dig_listeners[p] = self.dig_exit
        assert None not in self.dig_listeners.keys()
        for listener in self.dig_listeners.keys():
            listener.start()

        while ready.value < len(self.chan_to_dig):
            time.sleep(0.1)

        if self.cw_mode:
            for awg in self.awgs:
                awg.run()

    def add_instrument_sweep(self, instrument_name, attribute, values, channel=None):
        param = FloatParameter() # Create the parameter
        param.name = f"{instrument_name} {attribute} {channel}"
        instr = self._instruments[instrument_name]
        def method(value, channel=channel, instr=instr, prop=attribute):
            if channel:
                getattr(instr, "set_"+prop)(channel, value)
            else:
                getattr(instr, "set_"+prop)(value)
        param.assign_method(method)
        self.add_sweep(param, values) # Create the requested sweep on this parameter

    def add_manual_sweep(self, label, prompt, values, channel=None):
        param = FloatParameter() # Create the parameter
        param.name = label
        def method(value):
            print(f'Manually set {label} to {value}, then press enter.')
            input()
        param.assign_method(method)
        self.add_sweep(param, values) # Create the requested sweep on this parameter

    def add_qubit_sweep(self, qubit, measure_or_control, attribute, values):
        """
        Add a *ParameterSweep* to the experiment. Users specify a qubit property that auspex
        will try to link back to the relevant instrument. For example::
            exp = QubitExpFactory.create(PulsedSpec(q1))
            self.add_qubit_sweep(q1, "measure", "frequency", np.linspace(6e9, 6.5e9, 500))
            self.run_sweeps()
        """
        param = FloatParameter() # Create the parameter
        param.name = f"{qubit.label} {measure_or_control} {attribute}"

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
        if thing.phys_chan.generator and attribute=="frequency":
            # Mixed up to final frequency
            name  = thing.phys_chan.generator.label
            instr = list(filter(lambda x: x.name == name, self._instruments.values()))[0]
            method = None
        else:
            # Direct synthesis
            name, chan = thing.phys_chan.label.split("-")[0:2]
            instr = self._instruments[name] #list(filter(lambda x: x.name == name, self._instruments.values()))[0]

            #special casing for APS2 channel amplitude sweeps... is there a better way to do this?
            if isinstance(instr, auspex.instruments.APS2) and attribute=="amplitude":
                chan = [1, 2]

            def method(value, channel=chan, instr=instr, prop=attribute,thing=thing):
                # e.g. keysight.set_amplitude("ch1", 0.5)
                try:
                    getattr(instr, "set_"+prop)(chan, value, thing)
                except:
                    getattr(instr, "set_"+prop)(chan, value)
            param.set_pair = (thing.phys_chan.label, attribute)

        if method:
            # Custom method
            param.assign_method(method)

        else:
            # Get method by name
            if hasattr(instr, "set_"+attribute):
                param.assign_method(getattr(instr, "set_"+attribute)) # Couple the parameter to the instrument
                param.add_post_push_hook(lambda: time.sleep(0.05))
            else:
                raise ValueError("The instrument {} has no method {}".format(name, "set_"+attribute))
            param.set_pair = (instr.name, attribute)
        self.add_sweep(param, values) # Create the requested sweep on this parameter

    def add_avg_sweep(self, num_averages):
        param = IntParameter()
        param.name = "sw_avg"
        setattr(self, param.name, param)
        self._parameters[param.name] = param
        self.add_sweep(param, range(num_averages))

    def shutdown_instruments(self):
        # remove socket listeners
        logger.debug("Shutting down instruments")
        try:
            for awg in self.awgs:
                awg.stop()
            for dig in self.digitizers:
                dig.stop()
            for gen_proxy in self.generators:
                gen_proxy.instr.output = False
        except:
            logger.error('Could Not Stop AWGs or Digitizers; Reset Experiment')
        for instr in self.instruments:
            instr.disconnect()
        self.dig_exit.set()
        for listener in self.dig_listeners:
            listener.join(2)
            if listener.is_alive():
                logger.debug(f"Terminating listener {listener} aggressively")
                listener.terminate()
            del listener

        import gc
        gc.collect()

    def final_init(self):
        super(QubitExperiment, self).final_init()

        # In order to fetch data more easily later
        self.outputs_by_qubit =  {q.label: [self.proxy_to_filter[dat['node_obj']] for f,dat in self.modified_graph.nodes(data=True) if (isinstance(dat['node_obj'], (bbndb.auspex.Write, bbndb.auspex.Buffer,)) and q.label == dat['node_obj'].qubit_name)] for q in self.measured_qubits}

    def init_progress_bars(self):
        """ initialize the progress bars."""
        self.progressbars = {}
        ocs = list(self.output_connectors.values())
        if isnotebook():
            from ipywidgets import IntProgress, VBox
            from IPython.display import display
            if len(ocs)>0:
                for oc in ocs:
                    self.progressbars[oc] = IntProgress(min=0, max=oc.output_streams[0].descriptor.num_points(), bar_style='success',
                                                        description=f'Digitizer Data {oc.name}:', style={'description_width': 'initial'})
            for axis in self.sweeper.axes:
                self.progressbars[axis] = IntProgress(min=0, max=axis.num_points(),
                                                        description=f'{axis.name}:', style={'description_width': 'initial'})
            display(VBox(list(self.progressbars.values())))
        else:
            from progress.bar import ShadyBar
            if len(ocs)>0:
                for oc in ocs:
                    self.progressbars[oc] = ShadyBar(f'Digitizer Data {oc.name}:',
                                                max=oc.output_streams[0].descriptor.num_points())
            for axis in self.sweeper.axes:
                self.progressbars[axis] = ShadyBar(f"Sweep {axis.name}", max=axis.num_points())

    def run(self):
        # Begin acquisition before enabling the AWGs
        for dig in self.digitizers:
            dig.acquire()
            dig.last_timestamp.value = datetime.datetime.now().timestamp()

        # Set flag to enable acquisition process
        self.dig_run.set()

        # Start the AWGs
        if not self.cw_mode:
            for awg in self.awgs:
                awg.run()

        # Wait for all of the acquisitions to complete
        timeout = 10
        for dig in self.digitizers:
            dig.wait_for_acquisition(self.dig_run, timeout=timeout, ocs=list(self.chan_to_oc.values()), progressbars=self.progressbars)

        # Bring everything to a stop
        for dig in self.digitizers:
            dig.stop()

        # Pause the receiver processes so they don't time out
        self.dig_run.clear()

        # Stop the AWGs
        if not self.cw_mode:
            for awg in self.awgs:
                awg.stop()
