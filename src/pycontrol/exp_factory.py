# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import json
import importlib
import pkgutil
import inspect
import re
import asyncio
import base64

import numpy as np

import pycontrol.config as config
import pycontrol.instruments
import pycontrol.filters

from pycontrol.log import logger
from pycontrol.experiment import Experiment
from pycontrol.filters.filter import Filter
from pycontrol.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument
from pycontrol.stream import OutputConnector, DataStreamDescriptor, DataAxis

def strip_vendor_names(instr_name):
    vns = ["Agilent", "Alazar", "Keysight", "Holzworth", "Yoko", "Yokogawa"]
    for vn in vns:
        instr_name = instr_name.replace(vn, "")
    return instr_name

def correct_resource_name(resource_name):
    substs = {"USB::": "USB0::", }
    for k, v in substs.items():
        resource_name = resource_name.replace(k, v)
    return resource_name

class QubitExperiment(Experiment):
    """Experiment with a specialized run method for qubit experiments run via factory below."""
    def init_instruments(self):
        for name, instr in self._instruments.items():
            instr_par = self.exp_settings['instruments'][name]
            logger.debug("Setting instr %s with params %s.", name, instr_par)
            instr.set_all(instr_par)

        self.digitizers = [v for _, v in self._instruments.items() if v.instrument_type == "Digitizer"]
        self.awgs       = [v for _, v in self._instruments.items() if v.instrument_type == "AWG"]

        # Swap the master AWG so it is last in the list
        master_awg_idx = next(ct for ct,awg in enumerate(self.awgs) if self.exp_settings['instruments'][awg.name]['is_master'])
        self.awgs[-1], self.awgs[master_awg_idx] = self.awgs[master_awg_idx], self.awgs[-1]

    async def run(self):
        """This is run for each step in a sweep."""
        for dig in self.digitizers:
            dig.acquire()
        for awg in self.awgs:
            awg.run()

        async def wait_for_acq(digitizer):
            for _ in range(digitizer.number_acquisitions):
                while not digitizer.wait_for_acquisition():
                    await asyncio.sleep(0.1)

                # Find the associated output stream and push the data
                oc = self.dig_to_oc[digitizer]

                await oc.push(digitizer.ch2_buffer)
                logger.debug("Digitizer %s got data.", digitizer)
            logger.debug("Digitizer %s finished getting data.", digitizer)

        # Wait for all of the acquisition to complete
        await asyncio.wait([wait_for_acq(dig) for dig in self.digitizers])

        for dig in self.digitizers:
            dig.stop()
        for awg in self.awgs:
            awg.stop()

class QubitExpFactory(object):
    """The purpose of this factory is to examine DefaultExpSettings.json and construct
    and experiment therefrom."""

    @staticmethod
    def create():
        with open(config.instrumentLibFile, 'r') as FID:
            instrument_settings = json.load(FID)

        with open(config.measurementLibFile, 'r') as FID:
            measurement_settings = json.load(FID)

        with open(config.sweepLibFile, 'r') as FID:
            sweep_settings = json.load(FID)

        experiment = QubitExperiment()
        experiment.instrument_settings  = instrument_settings
        experiment.measurement_settings = measurement_settings
        experiment.sweep_settings       = sweep_settings

        QubitExpFactory.load_instruments(experiment)
        QubitExpFactory.load_sweeps(experiment)
        QubitExpFactory.load_filters(experiment)

        return experiment

    @staticmethod
    def load_instruments(experiment):
        # Inspect all vendor modules in pycontrol instruments and construct
        # a map to the instrument names.
        modules = (
            importlib.import_module('pycontrol.instruments.' + name)
            for loader, name, is_pkg in pkgutil.iter_modules(pycontrol.instruments.__path__)
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
        for instr_name, instr_par in experiment.instrument_settings['instrDict'].items():
            if instr_par['enabled']:
                # This should go away as pycontrol and pyqlab converge on naming schemes
                instr_type = strip_vendor_names(instr_par['device_name'])
                # Instantiate the desired instrument
                if instr_type in module_map:
                    logger.debug("Found instrument class %s for '%s' at loc %s when loading experiment settings.", instr_type, instr_name, instr_par['address'])
                    try:
                        inst = module_map[instr_type](correct_resource_name(instr_par['address']), name=instr_name)
                    except Exception as e:
                        logger.error("Initialization caused exception:", str(e))
                        inst = None
                    # Add to class dictionary for convenience
                    setattr(experiment, 'instr_name', inst)
                    # Add to _instruments dictionary
                    experiment._instruments[instr_name] = inst
                else:
                    logger.error("Could not find instrument class %s for '%s' when loading experiment settings.", instr_type, instr_name)


    @staticmethod
    def load_sweeps(experiment):
        # Load the active sweeps from the sweep ordering
        for name in experiment.sweep_settings['sweepOrder']:
            par = experiment.sweep_settings['sweepDict']['sweeps']

            if par['usePointsList']:
                points = np.array(par['points'])
            else:
                points = np.arange(par['start'], par['stop'] + 0.5*par['step'], par['step'])

            # Treat segment sweeps separately since they are DataAxes rather than SweepAxes
            if par['x__class__'] == 'SegmentNum':
                pass
            if par['x__class__'] == 'SegmentNumWithCals':
                points = np.append(points, np.zeros(numCals))
                data.metadata = ['data' for p in points] + ['cal' for s in range(par['numCals'])]
            else:
                # Here we create a parameter for experiment and associate it with the 
                # relevant method in the instrument

                # Add a parameter to the experiment corresponding to the thing we want to sweep
                param = FloatParameter()
                param.name = name
                setattr(experiment, name, param)
                experiment._parameters[name] = param

                # Get the instrument
                instr = experiment._instruments[par['instr']]
                method_name = 'set_' + par['x__class__'].lower()
                if hasattr(instr, method_name):
                    param.assign_method(getattr(instr, method_name)) # Couple the parameter to the instrument
                    experiment.add_sweep(param, points) # Create the requested sweep on this parameter
                else:
                    raise ValueError("The instrument {} has no method set_{}".format(name, par['x__class__'].lower()))
            
                # Add the sweep to the exeriment
                experiment.add_sweep()

    @staticmethod
    def load_filters(experiment):
        # These store any filters we create as well as their connections
        filters = {}
        graph   = []

        # ============================================
        # Find all of the filter modules by inspection
        # ============================================

        modules = (
            importlib.import_module('pycontrol.filters.' + name)
            for loader, name, is_pkg in pkgutil.iter_modules(pycontrol.filters.__path__)
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

        # Get the enabled measurements
        enabled_meas = {k: v for k, v in experiment.exp_settings['filterDict'].items() if v['enabled']}

        # First look for digitizer streams (Alazar or X6)
        dig_settings    = {k: v for k, v in enabled_meas if "StreamSelector" in v['x__class__']}
        
        # Remove these "Fake" filters from the exp_settings
        for k in dig_settings.keys():
            enabled_meas.pop(k)

        # Map from Digitizer -> OutputConnector for convenience
        dig_to_oc = {} 
        
        for name, settings in dig_settings.items():
            
            # Create and add the OutputConnector
            logger.debug("Adding %s output connector to experiment.", name)
            oc = OutputConnector(name=name, parent=experiment)
            experiment._output_connectors.append(oc)
            experiment.output_connectors[name] = oc
            setattr(experiment, name, oc)

            # Find the digitizer instrument and settings
            source_instr          = experiment._instruments[settings['data_source']]
            source_instr_settings = experiment.instr_settings['instrDict'][settings['data_source']]
            
            # Construct the descriptor
            descrip = DataStreamDescriptor()

            # If this is an X6, check to see what stream we are getting and modify the axis descriptor.
            # If it's an integrated stream, then the time axis has already been eliminated.
            if 'X6' in settings['x__class__']:
                if settings['stream_type'] != 'Integrated':
                    samp_time = 1.0e-9
                    descrip.add_axis(DataAxis("time", samp_time*np.array(range(source_instr_settings['record_length']))))
            else:
                samp_time = 1.0/source_instr_settings["sampling_rate"]
                descrip.add_axis(DataAxis("time", samp_time*np.array(range(source_instr_settings['record_length']))))

            descrip.add_axis(DataAxis("segments",     range(source_instr_settings['nbr_segments'])))
            descrip.add_axis(DataAxis("round_robins", range(source_instr_settings['nbr_round_robins'])))
            oc.set_descriptor(descrip)

            # Add to our mapping
            dig_to_oc[source_instr] = oc

        # ========================
        # Process the measurements
        # ========================

        for name, settings in enabled_meas.items():
            filt_type = settings['x__class__']

            if filt_type in module_map:
                if filt_type == "KernelIntegrator":
                    settings['kernel'] = np.fromstring( base64.b64decode(settings['kernel']), dtype=np.complex128)
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
            source_name = experiment.exp_settings['measurements'][name]['data_source']
            if source_name in filters:
                source = filters[source_name].source
            elif source_name in experiment.output_connectors:
                source = experiment.output_connectors[source_name]
            else:
                raise ValueError("Couldn't find anywhere to attach the source of the specified filter {}".format(name))

            # # ========================
            # # Create plot if requested
            # # ========================

            # has_plot = experiment.exp_settings['measurements'][name]['plot_scope']
            # if has_plot:
            #     averager = module_map['Averager'](name=name+"_average", axis='round_robins')
            #     plot = module_map['Plotter'](name=name, plot_mode=experiment.exp_settings['measurements'][name]['plot_mode'])
            #     graph.append([filt.source, averager.sink])
            #     graph.append([averager.final_average, plot.sink])

            # # ==========================
            # # Create writer if requested
            # # ==========================

            # # import ipdb; ipdb.set_trace()
            # if 'save_records' in experiment.exp_settings['measurements'][name]:
            #     has_writer = experiment.exp_settings['measurements'][name]['save_records']
            #     if has_writer:
            #         filename = experiment.exp_settings['measurements'][name]['records_file_path']
            #         writer = module_map['WriteToHDF5'](filename, name=name)
            #         graph.append([filt.source, writer.sink])

            logger.debug("Connecting %s to %s", source, filt)
            graph.append([source, filt.sink])

        experiment.dig_to_oc = dig_to_oc
        experiment.set_graph(graph)
