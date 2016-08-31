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

import pycontrol.config as config
import pycontrol.instruments
import pycontrol.filters

from pycontrol.log import logger
from pycontrol.experiment import Experiment
from pycontrol.filters.filter import Filter
from pycontrol.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument
from pycontrol.stream import OutputConnector

# Convert from camelCase to pep8 compliant labels
# http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def snakeify(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

# Recursively re-label dictionary
def rec_snakeify(dictionary):
    new = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = rec_snakeify(v)
        new[snakeify(k)] = v
    return new

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
            logger.debug("Setting instr %s with params %s.", name, rec_snakeify(instr_par))
            instr.set_all(rec_snakeify(instr_par))

        self.digitizers = [v for _, v in self._instruments.items() if v.instrument_type == "Digitizer"]
        self.awgs       = [v for _, v in self._instruments.items() if v.instrument_type == "AWG"]

        master_awg_idx = next(ct for ct,awg in enumerate(self.awgs) if self.exp_settings['instruments'][awg.name]['master'])
        self.awgs[-1], self.awgs[master_awg_idx] = self.awgs[master_awg_idx], self.awgs[-1]

    async def run(self):
        """This is run for each step in a sweep."""
        for dig in self.digitizers:
            dig.acquire()
        for awg in self.awgs:
            awg.run()

        # for _ in range(self.alazar.numberAcquistions):
        #     while not self.alazar.wait_for_acquisition():
        #         await asyncio.sleep(0.1)
        #     print("Got data!")
        #     await self.source.push(self.alazar.ch2Buffer)

        for dig in self.digitizers:
            dig.stop()
        for awg in self.awgs:
            awg.stop()

class QubitExpFactory(object):
    """The purpose of this factory is to examine DefaultExpSettings.json and construct
    and experiment therefrom."""

    @staticmethod
    def create():
        with open(config.expSettingsFile, 'r') as FID:
            exp_settings = json.load(FID)

        with open(config.channelLibFile, 'r') as FID:
            chan_settings = json.load(FID)

        experiment = QubitExperiment()
        experiment.exp_settings = rec_snakeify(exp_settings)

        QubitExpFactory.load_instruments(experiment)
        QubitExpFactory.load_filters(experiment)
        QubitExpFactory.load_sweeps(experiment)

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

        for instr_name, instr_par in experiment.exp_settings['instruments'].items():
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
        for chan_name, chan_par in experiment.exp_settings['sweeps'].items():
            pass

    @staticmethod
    def load_filters(experiment):
        # Change PyQLab measurements to Pycontrol filters
        name_changes = {'KernelIntegration': 'KernelIntegrator',
                        'DigitalDemod': 'Channelizer' }

        # These stores any filters we create as well as their connections
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

        # First look for raw streams
        raw_stream_settings = {k: v for k, v in experiment.exp_settings['measurements'].items() if v['filter_type'] == "RawStream"}
        # Remove them from the exp_settings
        for k in raw_stream_settings:
            experiment.exp_settings['measurements'].pop(k)
        # Next look for other types of streams
        # TODO: look for X6 type streams 
        for stream_name, stream_settings in raw_stream_settings.items():
            logger.debug("Added %s output connector to experiment.", stream_name)
            oc = OutputConnector(name=stream_name, parent=experiment)
            experiment._output_connectors.append(oc)
            experiment.output_connectors[stream_name] = oc
            setattr(experiment, stream_name, oc)
                  
        # ========================
        # Process the measurements
        # ========================

        for filt_name, filt_par in experiment.exp_settings['measurements'].items():
            filt_type = filt_par['filter_type']
            
            # Translate if necessary
            if filt_type in name_changes:
                filt_type = name_changes[filt_type]

            if filt_type in module_map:
                filt = module_map[filt_type](filt_par)
                filt.name = filt_name
                filters[filt_name] = filt
                logger.debug("Found filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)
            else:
                logger.error("Could not find filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)

        # ====================================
        # Establish all of the connections
        # ====================================

        for name, filt in filters.items():
            source_name = snakeify(experiment.exp_settings['measurements'][name]['data_source'])
            if source_name in filters:
                source = filters[source_name].source
            elif source_name in experiment.output_connectors:
                source = experiment.output_connectors[source_name]
            else:
                raise ValueError("Couldn't find anywhere to attach the source of the specified filter {}".format(name))
            
            # ========================
            # Create plot if requested
            # ========================

            has_plot = experiment.exp_settings['measurements'][name]['plot_scope']
            if has_plot:
                plot = module_map['Plotter'](name=name, plot_mode=experiment.exp_settings['measurements'][name]['plot_mode'])
                graph.append([filt.source, plot.sink])

            # ==========================
            # Create writer if requested
            # ==========================

            # import ipdb; ipdb.set_trace()
            if 'save_records' in experiment.exp_settings['measurements'][name]:
                has_writer = experiment.exp_settings['measurements'][name]['save_records']
                if has_writer:
                    filename = experiment.exp_settings['measurements'][name]['records_file_path']
                    writer = module_map['WriteToHDF5'](filename, name=name)
                    graph.append([filt.source, writer.sink])

            logger.debug("Connecting %s to %s", source, filt)
            graph.append([source, filt.sink])

        experiment.set_graph(graph)

