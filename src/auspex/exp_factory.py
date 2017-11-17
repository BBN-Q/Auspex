# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

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
import subprocess

import numpy as np
import networkx as nx

import auspex.config as config
import auspex.instruments
import auspex.filters
import auspex.globals

from auspex.log import logger
from auspex.experiment import Experiment
from auspex.filters.filter import Filter
from auspex.filters.io import DataBuffer
from auspex.filters.plot import Plotter, ManualPlotter
from auspex.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument, DigitizerChannel
from auspex.stream import OutputConnector, DataStreamDescriptor, DataAxis
from auspex.experiment import FloatParameter
from auspex.instruments.X6 import X6Channel
from auspex.instruments.alazar import AlazarChannel
from auspex.mixer_calibration import MixerCalibrationExperiment, find_null_offset

def correct_resource_name(resource_name):
    substs = {"USB::": "USB0::", }
    for k, v in substs.items():
        resource_name = resource_name.replace(k, v)
    return resource_name

def quince(filepath = config.configFile):
    if (os.name == 'nt'):
        subprocess.Popen(['run-quince.bat', config.configFile], env=os.environ.copy())
    else:
        subprocess.Popen(['run-quince.py', config.configFile], env=os.environ.copy())

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

class QubitExpFactory(object):
    """The purpose of this factory is to examine the configuration yaml and generate
    and experiment therefrom. The config file is loaded from the location specified
    in the *config.json* file, which is parsed by *config.py*.
    One can optionally pass meta info generated by QGL to either the *run* or the *create* methods, which
    will override some of the config values depending on the experiment being run."""

    @staticmethod
    def run(meta_file=None, expname=None, calibration=False, save_data=True, cw_mode=False, repeats=None):
        """This passes all of the parameters given to the *create* method
        and then runs the experiment immediately."""
        exp = QubitExpFactory.create(meta_file=meta_file, expname=expname,
                                     calibration=calibration, save_data=save_data, cw_mode=cw_mode,
                                    repeats=repeats)
        exp.run_sweeps()
        return exp

    @staticmethod
    def create(meta_file=None, expname=None, calibration=False, save_data = True, cw_mode=False, instr_filter = None, repeats=None):
        """Create the experiment, but do not run the sweeps. If *cw_mode* is specified
        the AWGs will be operated in continuous waveform mode, and will not be stopped
        and started between succesive sweep points. The *calibration* argument is used
        by the calibration routines (not intended for direct use) to automatically convert
        any file writers to IO buffers. The *meta_file* specified here is one output by
        QGL that specifies which instruments are required and what the SegmentSweep axes
        are. The *expname* argument is simply used to set the output directory relative
        to the data directory. If *repeats* is defined this will overide the
        number of segments gleaned from the meta_info"""

        settings = config.yaml_load(config.configFile)

        # This is generally the behavior we want
        auspex.globals.single_plotter_mode = True

        # Instantiate and perform all of our setup
        experiment = QubitExperiment()
        experiment.settings        = settings
        experiment.calibration     = calibration
        experiment.save_data       = save_data
        experiment.name            = expname
        experiment.cw_mode         = cw_mode
        experiment.calibration     = calibration
        experiment.repeats         = repeats

        if meta_file:
            QubitExpFactory.load_meta_info(experiment, meta_file)
        QubitExpFactory.load_instruments(experiment, instr_filter=instr_filter)
        QubitExpFactory.load_qubits(experiment)
        QubitExpFactory.load_filters(experiment)
        if 'sweeps' in settings:
            QubitExpFactory.load_parameter_sweeps(experiment)

        return experiment

    @staticmethod
    def calibrate_mixer(qubit, mixer="control", first_cal="phase", write_to_file=True,
    offset_range = (-0.2,0.2), amp_range = (0.6,1.4), phase_range = (-np.pi/6,np.pi/6), nsteps = 51):
        """Calibrates IQ mixer offset, amplitude imbalanace, and phase skew.
        See Analog Devices Application note AN-1039. Parses instrument connectivity from
        the experiment settings YAML.
        Arguments:
            qubit: Qubit identifier string.
            mixer: One of ("control", "measure") to select which IQ channel is calibrated.
            first_cal: One of ("phase", "amplitude") to select which adjustment is attempted
            first. You should pick whichever the particular mixer is most sensitive to.
            For example, a mixer with -40dBc sideband supression at 1 degree of phase skew
            and 0.1 dB amplitude imbalance should calibrate the phase first.
        """
        spm = auspex.globals.single_plotter_mode
        auspex.globals.single_plotter_mode = True

        def sweep_offset(name, pts):
            mce.clear_sweeps()
            mce.add_sweep(getattr(mce, name), pts)
            mce.keep_instruments_connected = True
            mce.run_sweeps()

        offset_pts = np.linspace(offset_range[0], offset_range[1], nsteps)
        amp_pts = np.linspace(amp_range[0], amp_range[1], nsteps)
        phase_pts = np.linspace(phase_range[0], phase_range[1], nsteps)

        buff = DataBuffer()
        plt = ManualPlotter(name="Mixer offset calibration", x_label='{} {} offset (V)'.format(qubit, mixer), y_label='Power (dBm)')
        plt.add_data_trace("I-offset", {'color': 'C1'})
        plt.add_data_trace("Q-offset", {'color': 'C2'})
        plt.add_fit_trace("Fit I-offset", {'color': 'C1'}) #TODO: fix axis labels
        plt.add_fit_trace("Fit Q-offset", {'color': 'C2'})

        plt2 = ManualPlotter(name="Mixer  amp/phase calibration", x_label='{} {} amplitude (V)/phase (rad)'.format(qubit, mixer), y_label='Power (dBm)')
        plt2.add_data_trace("phase_skew", {'color': 'C3'})
        plt2.add_data_trace("amplitude_factor", {'color': 'C4'})
        plt2.add_fit_trace("Fit phase_skew", {'color': 'C3'})
        plt2.add_fit_trace("Fit amplitude_factor", {'color': 'C4'})

        mce = MixerCalibrationExperiment(qubit, mixer=mixer)
        mce.add_manual_plotter(plt)
        mce.add_manual_plotter(plt2)
        mce.leave_plot_server_open = True
        QubitExpFactory.load_instruments(mce, mce.instruments_to_enable)
        edges = [(mce.amplitude, buff.sink)]
        mce.set_graph(edges)

        sweep_offset("I_offset", offset_pts)
        I1_amps = np.array([x[1] for x in buff.get_data()])
        I1_offset, xpts, ypts = find_null_offset(offset_pts[1:], I1_amps[1:])
        plt["I-offset"] = (offset_pts, I1_amps)
        plt["Fit I-offset"] = (xpts, ypts)
        logger.info("Found first pass I offset of {}.".format(I1_offset))
        mce.I_offset.value = I1_offset

        mce.first_exp = False # slight misnomer to indicate that no new plot is needed
        sweep_offset("Q_offset", offset_pts)
        Q1_amps = np.array([x[1] for x in buff.get_data()])
        Q1_offset, xpts, ypts = find_null_offset(offset_pts[1:], Q1_amps[1:])
        plt["Q-offset"] = (offset_pts, Q1_amps)
        plt["Fit Q-offset"] = (xpts, ypts)
        logger.info("Found first pass Q offset of {}.".format(Q1_offset))
        mce.Q_offset.value = Q1_offset

        sweep_offset("I_offset", offset_pts)
        I2_amps = np.array([x[1] for x in buff.get_data()])
        I2_offset, xpts, ypts = find_null_offset(offset_pts[1:], I2_amps[1:])
        plt["I-offset"] = (offset_pts, I2_amps)
        plt["Fit I-offset"] = (xpts, ypts)
        logger.info("Found second pass I offset of {}.".format(I2_offset))
        mce.I_offset.value = I2_offset

        #this is a bit hacky but OK...
        cals = {"phase": "phase_skew", "amplitude": "amplitude_factor"}
        cal_pts = {"phase": phase_pts, "amplitude": amp_pts}
        cal_defaults = {"phase": 0.0, "amplitude": 1.0}
        if first_cal not in cals.keys():
            raise ValueError("First calibration should be one of ('phase, amplitude'). Instead got {}".format(first_cal))
        second_cal = list(set(cals.keys()).difference({first_cal,}))[0]

        mce.sideband_modulation = True

        sweep_offset(cals[first_cal], cal_pts[first_cal])
        amps1 = np.array([x[1] for x in buff.get_data()])
        offset1, xpts, ypts = find_null_offset(cal_pts[first_cal][1:], amps1[1:], default=cal_defaults[first_cal])
        plt2[cals[first_cal]] = (cal_pts[first_cal], amps1)
        plt2["Fit "+cals[first_cal]] = (xpts, ypts)
        logger.info("Found {} offset of {}.".format(first_cal, offset1))
        getattr(mce, cals[first_cal]).value = offset1

        sweep_offset(cals[second_cal], cal_pts[second_cal])
        amps2 = np.array([x[1] for x in buff.get_data()])
        offset2, xpts, ypts = find_null_offset(cal_pts[second_cal][1:], amps2[1:], default=cal_defaults[second_cal])
        plt2[cals[second_cal]] = (cal_pts[second_cal], amps2)
        plt2["Fit "+cals[second_cal]] = (xpts, ypts)
        logger.info("Found {} offset of {}.".format(second_cal, offset2))
        getattr(mce, cals[second_cal]).value = offset2

        mce.disconnect_instruments()
        mce.extra_plot_server.stop()

        if write_to_file:
            mce.write_to_file()
        logger.info(("Mixer calibration: I offset = {}, Q offset = {}, "
                    "Amplitude Imbalance = {}, Phase Skew = {}").format(mce.I_offset.value,
                                                                        mce.Q_offset.value,
                                                                        mce.amplitude_factor.value,
                                                                        mce.phase_skew.value))
        auspex.globals.single_plotter_mode = spm

    @staticmethod
    def load_meta_info(experiment, meta_file):
        """If we get a meta_file, modify the configurations accordingly. Enable only instruments
        on the graph that connect the relevant *ReceiverChannel* objects to *Writer* or *Plotter*
        objects."""

        calibration = experiment.calibration
        save_data = experiment.save_data

        # Create a mapping from qubits to data writers and inverse
        qubit_to_writer     = {}
        writer_to_qubit     = {}
        qubit_to_stream_sel = {}
        stream_sel_to_qubit = {}

        # shortcuts
        instruments = experiment.settings['instruments']
        filters     = experiment.settings['filters']
        qubits      = experiment.settings['qubits']
        if 'sweeps' in experiment.settings:
            sweeps      = experiment.settings['sweeps']

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
            val_list = []
            # multiple sourcs are separated by commas
            all_vals = text.strip().split(',')
            for vals in all_vals:
                val = vals.strip().split()
                if len(val) == 0:
                    raise ValueError("Please disable filters with missing source.")
                elif len(val) > 2:
                    raise ValueError("Spaces are reserved to separate filters and connectors. Please rename {}.".format(vals))
                val_list.append(val[0])
            return val_list

        # Graph edges for the measurement filters
        # switch stream selector to raw (by default) before building the graph
        if experiment.__class__.__name__ == "SingleShotFidelityExperiment":
            receivers = [s for s in meta_info['receivers'].items()]
            if len(receivers) > 1:
                raise NotImplementedError("Single shot fidelity for more than one qubit is not yet implemented.")
            stream_sel_name_orig = receivers[0][0].replace('RecvChan-', '')
            X6_stream_selectors = [k for k,v in filters.items() if v["type"] == 'X6StreamSelector' and v["source"] == filters[stream_sel_name_orig]['source'] and v['channel'] == filters[stream_sel_name_orig]['channel']]
            for s in X6_stream_selectors:
                if filters[s]['stream_type'] == experiment.ss_stream_type:
                    filters[s]['enabled'] = True
                    stream_sel_name = s
                else:
                    filters[s]['enabled'] = False
        edges = [[(s, k) for s in strip_conn_name(v["source"])] for k,v in filters.items() if ("enabled" not in v.keys()) or v["enabled"]]
        edges = [e for edge in edges for e in edge]
        dag = nx.DiGraph()
        dag.add_edges_from(edges)

        inst_to_enable = []
        filt_to_enable = set()

        # Find any writer endpoints of the receiver channels
        for receiver_name, num_segments in meta_info['receivers'].items():
            # Receiver channel name format: RecvChan-StreamSelectorName
            if not experiment.__class__.__name__ == "SingleShotFidelityExperiment":
                stream_sel_name = receiver_name.replace('RecvChan-', '')
                stream_sel_name_orig = stream_sel_name
            dig_name = filters[stream_sel_name]['source']
            chan_name = filters[stream_sel_name]['channel']

            if experiment.repeats is not None:
                num_segments *= experiment.repeats

            # Set the correct number of segments for the digitizer
            instruments[dig_name]['nbr_segments'] = num_segments

            # Enable the digitizer
            inst_to_enable.append(dig_name)

            # Set number of segments in the digitizer
            instruments[dig_name]['nbr_segments'] = num_segments

            # Find the enabled X6 stream selectors with the same channel as the receiver. Allow to plot/save raw/demod/int streams belonging to the same receiver
            if calibration:
                X6_stream_selectors = []
            else:
                X6_stream_selectors = [k for k,v in filters.items() if (v["type"] == 'X6StreamSelector' and v["source"] == filters[stream_sel_name]['source'] and v["enabled"] == True and v["channel"] == filters[stream_sel_name]["channel"] and v["dsp_channel"] == filters[stream_sel_name]["dsp_channel"])]

            # Enable the tree for single-shot fidelity experiment. Change stream_sel_name to raw (by default)
            writers = []
            plotters = []
            singleshot = []
            buffers = []

            def check_endpoint(endpoint_name, endpoint_type):
                source_type = filters[filters[endpoint_name]['source'].split(' ')[0]]['type']
                return filters[endpoint_name]['type'] == endpoint_type and (not hasattr(filters[endpoint_name], 'enabled') or filters[endpoint_name]['enabled']) and not (calibration and source_type == 'Correlator') and (not source_type == 'SingleShotMeasurement' or experiment.__class__.__name__ == 'SingleShotFidelityExperiment')
            for filt_name, filt in filters.items():
                if filt_name in [stream_sel_name] + X6_stream_selectors:
                    # Find descendants of the channel selector
                    chan_descendants = nx.descendants(dag, filt_name)
                    # Find endpoints within the descendants
                    endpoints = [n for n in chan_descendants if dag.in_degree(n) == 1 and dag.out_degree(n) == 0]
                    # Find endpoints which are enabled writers, plotters or singleshot filters without an output. Disable outputs of single-shot filters when not used.
                    writers += [e for e in endpoints if check_endpoint(e, "WriteToHDF5")]
                    plotters += [e for e in endpoints if check_endpoint(e, "Plotter")]
                    buffers += [e for e in endpoints if check_endpoint(e, "DataBuffer")]
                    singleshot += [e for e in endpoints if check_endpoint(e, "SingleShotMeasurement") and experiment.__class__.__name__ == "SingleShotFidelityExperiment"]
            filt_to_enable.update(set().union(writers, plotters, singleshot, buffers))
            if calibration:
                # For calibrations the user should only have one writer enabled, otherwise we will be confused.
                if len(writers) > 1:
                    raise Exception("More than one viable data writer was found for a receiver channel {}. Please enable only one!".format(receiver_name))
                if len(writers) == 0 and len(plotters) == 0 and len(singleshot) == 0 and len(buffers) == 0:
                    raise Exception("No viable data writer, plotter or single-shot filter was found for receiver channel {}. Please enable one!".format(receiver_name))

            if writers and not save_data:
                # If we are calibrating we don't care about storing data, use buffers instead
                buffers = []
                for w in writers:
                    source_filt = filters[w]["source"].split(" ")[0]
                    if filters[source_filt]["type"] == "Averager":
                        sources = ", ".join([source_filt + " final_average", source_filt + " final_variance"])
                    else:
                        sources = filters[w]["source"]
                    buff = {
                            "source": sources,
                            "enabled": True,
                            "type": "DataBuffer",
                            }
                    # Remove the writer
                    filters.pop(w)
                    # Substitute the buffer
                    filters[w] = buff
                    # Store buffer name for local use
                    buffers.append(w)
                writers = buffers

            # For now we assume a single qubit, not a big change for multiple qubits
            qubit_name = next(k for k, v in qubits.items() if v["measure"]["receiver"] in (stream_sel_name, stream_sel_name_orig))
            if calibration:
                if len(writers) == 1:
                    qubit_to_writer[qubit_name] = writers[0]
            else:
                qubit_to_writer[qubit_name] = writers

            writer_ancestors = []
            plotter_ancestors = []
            singleshot_ancestors = []
            buffer_ancestors = []
            # Trace back our ancestors, using plotters if no writers are available
            if len(writers) == 1:
                writer_ancestors = nx.ancestors(dag, writers[0])
                # We will have gotten the digitizer, which should be removed since we're already taking care of it
                writer_ancestors.remove(dig_name)
            if plotters:
                plotter_ancestors = set().union(*[nx.ancestors(dag, pl) for pl in plotters])
                plotter_ancestors.remove(dig_name)
            if singleshot:
                singleshot_ancestors = set().union(*[nx.ancestors(dag, ss) for ss in singleshot])
                singleshot_ancestors.remove(dig_name)
            if buffers:
                buffer_ancestors = set().union(*[nx.ancestors(dag, bf) for bf in buffers])
                buffer_ancestors.remove(dig_name)
            filt_to_enable.update(set().union(writer_ancestors, plotter_ancestors, singleshot_ancestors, buffer_ancestors))

        if calibration:
            # One to one writers to qubits
            writer_to_qubit = {v: [k] for k, v in qubit_to_writer.items()}
        else:
            # Many to one writers to qubits or viceversa
            writer_to_qubit = {}
            for q, ws in qubit_to_writer.items():
                for w in ws:
                    if w not in writer_to_qubit:
                        writer_to_qubit[w] = []
                    writer_to_qubit[w].append(q)
        # Disable digitizers and APSs and then build ourself back up with the relevant nodes
        for instr_name in instruments.keys():
            if 'tx_channels' in instruments[instr_name].keys() or 'rx_channels' in instruments[instr_name].keys():
                instruments[instr_name]['enabled'] = False
        for instr_name in inst_to_enable:
            instruments[instr_name]['enabled'] = True

        for meas_name in filters.keys():
            filters[meas_name]['enabled'] = False
        for meas_name in filt_to_enable:
            filters[meas_name]['enabled'] = True

        #label measurement with qubit name (assuming the convention "M-"+qubit_name)
        for meas_name in filt_to_enable:
            if filters[meas_name]["type"] == "WriteToHDF5":
                filters[meas_name]['groupname'] = ''.join(writer_to_qubit[meas_name]) \
                    + "-" + filters[meas_name]['groupname']

        for instr_name, chan_data in meta_info['instruments'].items():
            instruments[instr_name]['enabled']  = True
            if isinstance(chan_data, str):
                instruments[instr_name]['seq_file'] = chan_data # Per-instrument seq file
            elif isinstance(chan_data, dict):
                for chan_name, seq_file in chan_data.items():
                    if "tx_channels" in instruments[instr_name] and chan_name in instruments[instr_name]["tx_channels"].keys():
                        instruments[instr_name]["tx_channels"][chan_name]['seq_file'] = seq_file
                    elif "rx_channels" in instruments[instr_name] and chan_name in instruments[instr_name]["rx_channels"].keys():
                        instruments[instr_name]["rx_channels"][chan_name]['seq_file'] = seq_file
                    else:
                        raise ValueError("Could not find channel {} in of instrument {}.".format(chan_name, instr_name))

        # Now we will construct the DataAxis from the meta_info
        desc = meta_info["axis_descriptor"]
        data_axis = desc[0] # Data will always be the first axis

        if experiment.repeats is not None:
            #ovverride data axis with repeated number of segments
            data_axis['points'] = np.tile(data_axis['points'], experiment.repeats)

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
    def load_instruments(experiment, instr_filter=None):
        """Parse the instruments settings and instantiate the corresponding Auspex instruments by name.
        This function first traverses all vendor instrument modules and constructs a map from the instrument names
        to the relevant module members. To select only a subset of instruments, use an instrument_filter by either
        passing a list of instruments to be loaded by name, or a callable object that takes a (key, value) pair
        and returns a boolean. Example: instr_filter = lambda x: 'Holzworth' in x[1]['type'] or 'APS' in x[0]"""
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

        #only select the instruments we want. not super happy with this code so
        #if anyone has a better way to do this please fix. -GJR
        if instr_filter is not None:
            if isinstance(instr_filter, (list, tuple)):
                filtfun = lambda x: x[0] in instr_filter
            elif hasattr(instr_filter, "__call__"):
                filtfun = instr_filter
            else:
                raise TypeError("Instrument filter must be either a list, tuple, or callable. Got a {} instead.".format(type(instr_filter)))
        instruments = {k: v for k, v in experiment.settings['instruments'].items() if instr_filter is None or filtfun((k,v))}
        if not instruments:
            logger.warning("No instruments are being loaded by the experiment! Is this what you want to do?")

        # Loop through instruments, and add them to the experiment if they are enabled.
        for name, par in instruments.items():
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

                # We might need to return a custom function rather than just a method_name
                method = None

                # Figure our what we are sweeping
                target_info = par["target"].split()
                if target_info[0] in experiment.qubits:
                    # We are sweeping a qubit, so we must lookup the instrument
                    name, meas_or_control, prop = par["target"].split()
                    qubit = qubits[name]
                    method_name = "set_{}".format(prop.lower())

                    # If sweeping frequency, we should allow for either mixed up signals or direct synthesis.
                    # Sweeping power is always through the AWG channels.
                    if 'generator' in qubit[meas_or_control] and prop.lower() == "frequency":
                        name = qubit[meas_or_control]['generator']
                        instr = experiment._instruments[name]
                    else:
                        # Construct a function that sets a per-channel property
                        name, chan = qubit[meas_or_control]['AWG'].split()
                        instr = experiment._instruments[name]

                        def method(value, channel=chan, instr=instr, prop=prop.lower()):
                            # e.g. keysight.set_amplitude("ch1", 0.5)
                            getattr(instr, "set_"+prop)(chan, value)

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

                if method:
                    # Custom method
                    param.assign_method(method)
                else:
                    # Get method by name
                    if hasattr(instr, method_name):
                        param.assign_method(getattr(instr, method_name)) # Couple the parameter to the instrument
                    else:
                        raise ValueError("The instrument {} has no method {}".format(name, method_name))
                param.instr_tree = [instr.name, prop] #TODO: extend tree to endpoint
                experiment.add_sweep(param, points) # Create the requested sweep on this parameter

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

        # Get the enabled measurements, or those which aren't explicitly
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
