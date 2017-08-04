# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import yaml
import json
import sys
import os
import importlib
import pkgutil
import inspect
import re
import asyncio
import base64
import datetime

import numpy as np
import networkx as nx

import auspex.config as config
import auspex.instruments
import auspex.filters

from auspex.log import logger
from auspex.experiment import Experiment
from auspex.filters.filter import Filter
from auspex.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument, DigitizerChannel
from auspex.stream import OutputConnector, DataStreamDescriptor, DataAxis
from auspex.experiment import FloatParameter
from auspex.instruments.X6 import X6Channel
from auspex.instruments.alazar import AlazarChannel

def correct_resource_name(resource_name):
    substs = {"USB::": "USB0::", }
    for k, v in substs.items():
        resource_name = resource_name.replace(k, v)
    return resource_name

class QubitExperiment(Experiment):
    """Experiment with a specialized run method for qubit experiments run via the QubitExpFactory."""
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

    def add_qubit_sweep(self, property_name, values):
        """
        Add a *ParameterSweep* to the experiment. By the time this experiment exists, it has already been run
        through the qubit factory, and thus already has its segment sweep defined. This method simply utilizes
        the *load_parameters_sweeps* method of the QubitExpFactory, thus users can provide
        either a space-separated pair of *instr_name method_name* (i.e. *Holzworth1 power*)
        or specify a qubit property that auspex will try to link back to the relevant instrument.
        (i.e. *q1 measure frequency* or *q2 control power*). For example::
            exp = QubitExpFactory.create(PulsedSpec(q))
            exp.add_qubit_sweep("q1 measure frequency", np.linspace(6e9, 6.5e9, 500))
            exp.run_sweeps()
        """
        desc = {property_name:
                {'name': property_name,
                'target': property_name,
                'points': values,
                'type': "Parameter"
               }}
        QubitExpFactory.load_parameter_sweeps(self, manual_sweep_params=desc)


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

class QubitExpFactory(object):
    """The purpose of this factory is to examine the configuration yaml and generate
    and experiment therefrom. The config file is loaded from the location specified 
    in the *config.json* file, which is parsed by *config.py*.
    One can optionally pass meta info generated by QGL to either the *run* or the *create* methods, which
    will override some of the config values depending on the experiment being run."""

    @staticmethod
    def run(meta_file=None, expname=None, calibration=False, cw_mode=False):
        """This passes all of the parameters given to the *create* method 
        and then runs the experiment immediately."""
        exp = QubitExpFactory.create(meta_file=meta_file, expname=expname,
                                     calibration=calibration, cw_mode=cw_mode)
        exp.run_sweeps()
        return exp

    @staticmethod
    def create(meta_file=None, expname=None, calibration=False, cw_mode=False):
        """Create the experiment, but do not run the sweeps. If *cw_mode* is specified
        the AWGs will be operated in continuous waveform mode, and will not be stopped 
        and started between succesive sweep points. The *calibration* argument is used 
        by the calibration routines (not intended for direct use) to automatically convert
        any file writers to IO buffers. The *meta_file* specified here is one output by 
        QGL that specifies which instruments are required and what the SegmentSweep axes
        are. The *expname* argument is simply used to set the output directory relative
        to the data directory."""

        settings = config.yaml_load(config.configFile)

        # This is generally the behavior we want
        auspex.globals.single_plotter_mode = True

        # Instantiaite and perform all of our setup
        experiment = QubitExperiment()
        experiment.settings        = settings
        experiment.calibration     = calibration
        experiment.name            = expname
        experiment.cw_mode         = cw_mode
        experiment.calibration     = calibration

        if meta_file:
            QubitExpFactory.load_meta_info(experiment, meta_file)
        QubitExpFactory.load_instruments(experiment)
        QubitExpFactory.load_qubits(experiment)
        QubitExpFactory.load_filters(experiment)
        if 'sweeps' in settings:
            QubitExpFactory.load_parameter_sweeps(experiment)

        return experiment

    @staticmethod
    def load_meta_info(experiment, meta_file):
        """If we get a meta_file, modify the configurations accordingly. Enable only instruments
        on the graph that connect the relevant *ReceiverChannel* objects to *Writer* or *Plotter* 
        objects."""

        calibration = experiment.calibration

        # Create a mapping from qubits to data writers and inverse
        qubit_to_writer     = {}
        writer_to_qubit     = {}
        qubit_to_stream_sel = {}
        stream_sel_to_qubit = {}

        # Use the meta info to modify the parameters
        # loaded from the human-friendly yaml configuration.
        with open(meta_file, 'r') as FID:
            meta_info = json.load(FID)

        # Construct a graph of all instruments in order to properly enabled those
        # associated with the meta_file. We only need to use string representations
        # here, not actual filter and instrument objects.

        # Strip any spaces, since we only care about the general flow, and not any
        # named connectors.
        def strip_conn_name(text):
            vals = text.strip().split()
            if len(vals) == 0:
                raise ValueError("Please disable filters with missing source.")
            return vals[0]

        # Graph edges for the measurement filters
        edges = [(strip_conn_name(v["source"]), k) for k,v in experiment.settings["filters"].items() if ("enabled" not in v.keys()) or v["enabled"]]
        dag = nx.DiGraph()
        dag.add_edges_from(edges)

        inst_to_enable = []
        filt_to_enable = []

        # Laziness
        instruments = experiment.settings['instruments']
        filters     = experiment.settings['filters']
        qubits      = experiment.settings['qubits']
        if 'sweeps' in experiment.settings:
            sweeps      = experiment.settings['sweeps']

        # Find any writer endpoints of the receiver channels
        for receiver_name, num_segments in meta_info['receivers'].items():
            # Receiver channel name format: RecvChan-StreamSelectorName
            stream_sel_name = receiver_name.replace('RecvChan-', '')
            dig_name = filters[stream_sel_name]['source']
            chan_name = filters[stream_sel_name]['channel']
            
            # Set the correct number of segments for the digitizer
            experiment.settings["instruments"][dig_name]['nbr_segments'] = num_segments

            # Enable the digitizer
            inst_to_enable.append(dig_name)

            # Set number of segments in the digitizer
            instruments[dig_name]['nbr_segments'] = num_segments
            
            # Find the descendants tree for the receiver channels
            writers = []
            plotters = []
            for filt_name, filt in filters.items():
                if filt_name == stream_sel_name:
                    # Find descendants of the channel selector
                    chan_descendants = nx.descendants(dag, filt_name)
                    # Find endpoints within the descendants
                    endpoints = [n for n in chan_descendants if dag.in_degree(n) == 1 and dag.out_degree(n) == 0]
                    # Find endpoints which are enabled writers
                    writers += [e for e in endpoints if filters[e]["type"] == "WriteToHDF5" and (not hasattr(filters[e], "enabled") or filters[e]["enabled"])]
                    plotters += [e for e in endpoints if filters[e]["type"] == "Plotter" and (not hasattr(filters[e], "enabled") or filters[e]["enabled"])]
            
            if calibration:
                # For calibrations the user should only have one writer enabled, otherwise we will be confused.
                if len(writers) > 1:
                    raise Exception("More than one viable data writer was found for a receiver channel {}. Please enabled only one!".format(receiver_name))
                if len(writers) == 0 and len(plotters) == 0:
                    raise Exception("No viable data writer or plotter was found for receiver channel {}. Please enabled only one!".format(receiver_name))

                # If we are calibrating we don't care about storing data, use buffers instead
                buffers = []
                for w in writers:
                    label = filters[w]["label"]
                    buff = {
                            "source": filters[w]["source"],
                            "enabled": True,
                            "label": label,
                            "type": "DataBuffer",
                            }
                    # Remove the writer
                    filters.pop(filters[w]["label"])
                    # Substitute the buffer
                    filters[label] = buff
                    # Store buffer name for local use
                    buffers.append(label)
                writers = buffers

            # For now we assume a single qubit, not a big change for multiple qubits
            qubit_name = next(k for k, v in qubits.items() if v["measure"]["receiver"] == stream_sel_name)
            if calibration:
                if len(writers) == 1:
                    qubit_to_writer[qubit_name] = writers[0]
            else:
                qubit_to_writer[qubit_name] = writers

            if calibration:
                # Trace back our ancestors, using plotters if no writers are available
                if len(writers) == 1:
                    useful_output_filter_ancestors = nx.ancestors(dag, writers[0])
                else:
                    useful_output_filter_ancestors = nx.ancestors(dag, plotters[0])

                # We will have gotten the digitizer, which should be removed since we're already taking care of it
                useful_output_filter_ancestors.remove(dig_name)

                if plotters:
                    plotter_ancestors = set().union(*[nx.ancestors(dag, pl) for pl in plotters])
                    plotter_ancestors.remove(dig_name)
                else:
                    plotter_ancestors = []

                filt_to_enable.extend(set().union(writers, useful_output_filter_ancestors, plotters, plotter_ancestors))

        if calibration:
            # One to one writers to qubits
            writer_to_qubit = {v: k for k, v in qubit_to_writer.items()}
        else:
            # Many to one writers to qubits
            writer_to_qubit = {}
            for q, ws in qubit_to_writer.items():
                for w in ws:
                    writer_to_qubit[w] = q

        # Disable digitizers and APSs and then build ourself back up with the relevant nodes
        for instr_name in instruments.keys():
            if 'tx_channels' in instruments[instr_name].keys() or 'rx_channels' in instruments[instr_name].keys():
                experiment.settings["instruments"][instr_name]['enabled'] = False
        for instr_name in inst_to_enable:
            experiment.settings["instruments"][instr_name]['enabled'] = True

        if calibration:
            for meas_name in filters.keys():
                experiment.settings['filters'][meas_name]['enabled'] = False
            for meas_name in filt_to_enable:
                experiment.settings['filters'][meas_name]['enabled'] = True
        else:        
            #label measurement with qubit name (assuming the convention "M-"+qubit_name)
            for meas_name in filt_to_enable:
                if experiment.settings['filters'][meas_name]["type"] == "WriteToHDF5":
                    experiment.settings['filters'][meas_name]['groupname'] = writer_to_qubit[meas_name] \
                        + "-" + experiment.settings['filters'][meas_name]['groupname']

        for instr_name, chan_data in meta_info['instruments'].items():
            experiment.settings["instruments"][instr_name]['enabled']  = True
            if isinstance(chan_data, str):
                experiment.settings["instruments"][instr_name]['seq_file'] = chan_data # Per-instrument seq file
            elif isinstance(chan_data, dict):
                for chan_name, seq_file in chan_data.items():
                    if "tx_channels" in experiment.settings["instruments"][instr_name] and chan_name in experiment.settings["instruments"][instr_name]["tx_channels"].keys():
                        experiment.settings["instruments"][instr_name]["tx_channels"][chan_name]['seq_file'] = seq_file
                    elif "rx_channels" in experiment.settings["instruments"][instr_name] and chan_name in experiment.settings["instruments"][instr_name]["rx_channels"].keys():
                        experiment.settings["instruments"][instr_name]["rx_channels"][chan_name]['seq_file'] = seq_file
                    else:
                        raise ValueError("Could not find channel {} in of instrument {}.".format(chan_name, instr_name))

        # Now we will construct the DataAxis from the meta_info
        desc = meta_info["axis_descriptor"] 
        data_axis = desc[0] # Data will always be the first axis

        # See if there are multiple partitions, and therefore metadata
        if len(desc) > 1:
            meta_axis = desc[1] # Metadata will always be the second axis
            # There should be metadata for each cal describing what it is
            metadata = ['data']*len(data_axis['points']) + meta_axis['points']

            # Pad the data axis with dummy equidistant x-points for the extra calibration points
            avg_step = (data_axis['points'][-1] - data_axis['points'][0])/(len(data_axis['points'])-1)
            points = np.append(data_axis['points'], data_axis['points'][-1] + (np.arange(len(meta_axis['points']))+1)*avg_step)

            # If there's only one segment we can ignore this axis
            if len(points) > 1:
                experiment.segment_axis = DataAxis(data_axis['name'], points, unit=data_axis['unit'], metadata=metadata)

        else:
            if len(data_axis['points']) > 1:
                experiment.segment_axis = DataAxis(data_axis['name'], data_axis['points'], unit=data_axis['unit'])

        experiment.qubit_to_writer = qubit_to_writer
        experiment.writer_to_qubit = writer_to_qubit

    @staticmethod
    def load_qubits(experiment):
        """Parse the settings files and add the list of qubits to the experiment as *experiment.qubits*,
        as well as creating the *experiment.qubit_to_stream_sel* mapping and its inverse."""
        experiment.qubits = []
        experiment.qubit_to_stream_sel = {}
        experiment.stream_sel_to_qubit = {}
        for name, par in experiment.settings['qubits'].items():
            experiment.qubits.append(name)
            experiment.qubit_to_stream_sel[name] = par["measure"]["receiver"]
            experiment.stream_sel_to_qubit[par["measure"]["receiver"]] = name

    @staticmethod
    def load_instruments(experiment):
        """Parse the instruments settings and instantiate the corresponding Auspex instruments by name. 
        This function first traverses all vendor instrument modules and constructs a map from the instrument names
        to the relevant module members."""
        modules = (
            importlib.import_module('auspex.instruments.' + name)
            for loader, name, is_pkg in pkgutil.iter_modules(auspex.instruments.__path__)
        )

        module_map = {}
        for mod in modules:
            instrs = (_ for _ in inspect.getmembers(mod) if inspect.isclass(_[1]) and
                                                            issubclass(_[1], Instrument) and
                                                            _[1] != Instrument and _[1] != SCPIInstrument and
                                                            _[1] != CLibInstrument)
            module_map.update(dict(instrs))
        logger.debug("Found instruments %s.", module_map)

        # Loop through instruments, and add them to the experiment if they are enabled.
        for name, par in experiment.settings['instruments'].items():
            # Assume we're enabled by default, or enabled is not False
            if 'enabled' not in par or par['enabled']:
                # This should go away as auspex and pyqlab converge on naming schemes
                instr_type = par['type']
                par['name'] = name
                # Instantiate the desired instrument
                if instr_type in module_map:
                    logger.debug("Found instrument class %s for '%s' at loc %s when loading experiment settings.", instr_type, name, par['address'])
                    try:
                        inst = module_map[instr_type](correct_resource_name(str(par['address'])), name=name)
                    except Exception as e:
                        logger.error("Initialization of caused exception:", name, str(e))
                        inst = None
                    # Add to class dictionary for convenience
                    if not hasattr(experiment, name):
                        setattr(experiment, name, inst)
                    # Add to _instruments dictionary
                    experiment._instruments[name] = inst
                else:
                    logger.error("Could not find instrument class %s for '%s' when loading experiment settings.", instr_type, name)

    @staticmethod
    def load_parameter_sweeps(experiment, manual_sweep_params=None):
        """Create parameter sweeps (non-segment sweeps) from the settings. Users can provide
        either a space-separated pair of *instr_name method_name* (i.e. *Holzworth1 power*)
        or specify a qubit property that auspex will try to link back to the relevant instrument.
        (i.e. *q1 measure frequency* or *q2 control power*). Auspex will create a *SweepAxis* 
        for each parameter sweep, and add this axis to all output connectors."""
        if manual_sweep_params:
            sweeps = manual_sweep_params
            order = [list(sweeps.keys())[0]]
        else:
            sweeps = experiment.settings['sweeps']
            order = experiment.settings['sweepOrder']
        qubits = experiment.settings['qubits']

        for name in order:
            par = sweeps[name]
            # Treat segment sweeps separately since they are DataAxes rather than SweepAxes
            if par['type'] != 'Segment':
                # Here we create a parameter for experiment and associate it with the
                # relevant method of the relevant experiment.

                # Add a parameter to the experiment corresponding to the thing we want to sweep
                if "unit" in par:
                    param = FloatParameter(unit=par["unit"])
                else:
                    param = FloatParameter()
                param.name = name
                setattr(experiment, name, param)
                experiment._parameters[name] = param

                # Figure our what we are sweeping
                target_info = par["target"].split()
                if target_info[0] in experiment.qubits:
                    # We are sweeping a qubit, so we must lookup the instrument
                    name, meas_or_control, prop = par["target"].split()
                    qubit = qubits[name]

                    # We should allow for either mixed up signals or direct synthesis
                    if 'generator' in qubit[meas_or_control]:
                        name = qubit[meas_or_control]['generator']
                        instr = experiment._instruments[name]
                        method_name = 'set_' + prop.lower()
                    else:
                        name, chan = experiment.settings['qubits'][meas_or_control]['AWG']
                        instr = experiment._instruments[name]
                        method_name = "set_{}_{}".format(chan, prop.lower())
                        
                elif target_info[0] in experiment._instruments:
                    # We are sweeping an instrument directly
                    # Get the instrument being swept, and find the relevant method
                    name, prop = par["target"].split()
                    instr = experiment._instruments[name]
                    method_name = 'set_' + prop.lower()

                # If there's a "points" property, use those directly. Otherwise, we
                # use numPoints or the step interval.
                if "points" in par:
                    points = par["points"]
                elif "numPoints" in par:
                    points = np.linspace(par['start'], par['stop'], par['numPoints'])
                elif "step" in par:
                    points = np.arange(par['start'], par['stop'], par['step'])

                if hasattr(instr, method_name):
                    param.assign_method(getattr(instr, method_name)) # Couple the parameter to the instrument
                    experiment.add_sweep(param, points) # Create the requested sweep on this parameter
                else:
                    raise ValueError("The instrument {} has no method set_{}".format(name, par['type'].lower()))

    @staticmethod
    def load_filters(experiment):
        """This function first traverses all filter modules and constructs a map from the filter names
        to the relevant module members. Then it parses the settings and instantiates Auspex filters for 
        each filter found therein. Finally, all of the relevant connections are established between filters
        and back to the experiment class instance, to which *OutputConnectors* are added as needed."""

        # These store any filters we create as well as their connections
        filters = {}
        graph   = []

        # ============================================
        # Find all of the filter modules by inspection
        # ============================================

        modules = (
            importlib.import_module('auspex.filters.' + name)
            for loader, name, is_pkg in pkgutil.iter_modules(auspex.filters.__path__)
        )

        module_map = {}
        for mod in modules:
            filts = (_ for _ in inspect.getmembers(mod) if inspect.isclass(_[1]) and
                                                            issubclass(_[1], Filter) and
                                                            _[1] != Filter)
            module_map.update(dict(filts))

        # ==================================================
        # Find out which output connectors we need to create
        # ==================================================

        # Get the enabled measurements, or those which aren't explicitlu 
        enabled_meas = {k: v for k, v in experiment.settings['filters'].items() if 'enabled' not in v or v['enabled'] }

        # First look for digitizer streams (Alazar or X6)
        dig_settings = {k: v for k, v in enabled_meas.items() if "StreamSelector" in v['type']}

        # These stream selectors are really just a convenience
        # Remove them from the list of "real" filters
        for k in dig_settings.keys():
            enabled_meas.pop(k)

        # Map from Channel -> OutputConnector
        # and from Channel -> Digitizer for future lookup
        chan_to_oc  = {}
        chan_to_dig = {}

        for name, settings in dig_settings.items():

            # Create and add the OutputConnector
            logger.debug("Adding %s output connector to experiment.", name)
            oc = OutputConnector(name=name, parent=experiment)
            experiment._output_connectors[name] = oc
            experiment.output_connectors[name] = oc
            setattr(experiment, name, oc)

            # Find the digitizer instrument and settings
            source_instr          = experiment._instruments[settings['source']]
            source_instr_settings = next(v for k,v in experiment.settings['instruments'].items() if k == settings['source'])

            # Construct the descriptor from the stream
            stream_type = settings['type']
            stream = module_map[stream_type](name=name)
            channel, descrip = stream.get_descriptor(source_instr_settings, settings)

            # Add the channel to the instrument
            source_instr.add_channel(channel)

            # Add the segment axis, which should already be defined...
            if hasattr(experiment, 'segment_axis'):
                # This should contains the proper range and units based on the sweep descriptor
                descrip.add_axis(experiment.segment_axis)
            else:
                # This is the generic axis based on the instrument parameters
                # If there is only one segement, we should omit this axis.
                if source_instr_settings['nbr_segments'] > 1:
                    descrip.add_axis(DataAxis("segments", range(source_instr_settings['nbr_segments'])))

            # Digitizer mode preserves round_robins, averager mode collapsing along them:
            if 'acquire_mode' not in source_instr_settings.keys() or source_instr_settings['acquire_mode'] == 'digitizer':
                descrip.add_axis(DataAxis("round_robins", range(source_instr_settings['nbr_round_robins'])))

            oc.set_descriptor(descrip)

            # Add to our mappings
            chan_to_oc[channel]    = oc
            chan_to_dig[channel]   = source_instr

        # ========================
        # Process the measurements
        # ========================

        for name, settings in enabled_meas.items():
            filt_type = settings['type']

            if filt_type in module_map:
                filt = module_map[filt_type](**settings)
                filt.name = name
                filters[name] = filt
                logger.debug("Found filter class %s for '%s' when loading experiment settings.", filt_type, name)
            else:
                logger.error("Could not find filter class %s for '%s' when loading experiment settings.", filt_type, name)

        # ====================================
        # Establish all of the connections
        # ====================================

        for name, filt in filters.items():

            # Multiple data sources are comma separated, with optional whitespace.
            # If there is a space in the individual names, then we are to hook up to a specific connector
            # Otherwise we can safely assume that the name is "source"

            sources = [s.strip() for s in enabled_meas[name]['source'].split(",")]

            for source in sources:
                source = source.split()
                node_name = source[0]
                conn_name = "source"
                if len(source) == 2:
                    conn_name = source[1]

                if node_name in filters:
                    source = filters[node_name].output_connectors[conn_name]
                elif node_name in experiment.output_connectors:
                    source = experiment.output_connectors[node_name]
                else:
                    raise ValueError("Couldn't attach the source of the specified filter {} to {}".format(name, source))

                logger.debug("Connecting %s@%s ---> %s", node_name, conn_name, filt)
                graph.append([source, filt.sink])

        experiment.chan_to_oc  = chan_to_oc
        experiment.chan_to_dig = chan_to_dig
        experiment.set_graph(graph)

        # For convenient lookup
        experiment.filters = filters
