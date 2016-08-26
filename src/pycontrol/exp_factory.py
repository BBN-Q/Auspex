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
from pycontrol.instruments.instrument import Instrument

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

class QubitExpFactory(object):
    """The purpose of this factory is to examine DefaultExpSettings.json and construct
    and experiment therefrom.""" 
    
    @staticmethod
    def create():
        with open(config.expSettingsFile, 'r') as FID: 
            exp_settings = json.load(FID)

        with open(config.channelLibFile, 'r') as FID: 
            chan_settings = json.load(FID)

        experiment = Experiment()

        QubitExpFactory.load_instruments(experiment, exp_settings)
        QubitExpFactory.load_channels(experiment, chan_settings)
        QubitExpFactory.load_filters(experiment, exp_settings)
        QubitExpFactory.load_sweeps(experiment, exp_settings)

        return experiment

    @staticmethod
    def load_instruments(experiment, exp_settings):
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
                                                            _[1] != Instrument)
            module_map.update(dict(instrs))

        for instr_name, instr_par in exp_settings['instruments'].items():
            instr_type = strip_vendor_names(instr_par['deviceName'])
            # Instantiate the desired instrument
            if instr_type in module_map:
                logger.debug("Found instrument class %s for '%s' at loc %s when loading experiment settings.", instr_type, instr_name, instr_par['address'])
                try:
                    logger.debug("Setting instr %s with params %s.", instr_name, rec_snakeify(instr_par))
                    inst = module_map[instr_type](correct_resource_name(instr_par['address']))
                    inst.set_all(rec_snakeify(instr_par))
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
    def load_channels(experiment, chan_settings):
        # Add output connectors for each defined channel
        for chan_name, chan_par in chan_settings['channelDict'].items():
            pass

    @staticmethod
    def load_sweeps(experiment, exp_settings):
        for chan_name, chan_par in exp_settings['sweeps'].items():
            pass

    @staticmethod
    def load_filters(experiment, exp_settings):
        modules = (
            importlib.import_module('pycontrol.filters.' + name)
            for loader, name, is_pkg in pkgutil.iter_modules(pycontrol.filters.__path__)
        )

        module_map = {}
        for mod in modules:
            filters = (_ for _ in inspect.getmembers(mod) if inspect.isclass(_[1]) and 
                                                            issubclass(_[1], Filter) and
                                                            _[1] != Filter)
            module_map.update(dict(filters))

        for filt_name, filt_par in exp_settings['measurements'].items():
            filt_type = filt_par['filterType']
            if filt_type in module_map:
                filt = module_map[filt_type]()
                logger.debug("Found filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)
            else:
                logger.error("Could not find filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)