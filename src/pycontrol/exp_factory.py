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

import pycontrol.config as config
import pycontrol.instruments
import pycontrol.filters

from pycontrol.log import logger
from pycontrol.experiment import Experiment
from pycontrol.filters.filter import Filter
from pycontrol.instruments.instrument import Instrument

class QubitExpFactory(object):
    """The purpose of this factory is to examine ExpSettings.json""" 
    def __init__(self):
        with open(config.expSettingsFile, 'r') as FID: 
            self.exp_settings = json.load(FID)

        with open(config.channelLibFile, 'r') as FID: 
            self.chan_settings = json.load(FID)

        self.experiment = Experiment()

        self.load_instruments()
        self.load_channels()
        self.load_filters()
        self.load_sweeps()

    def load_instruments(self):
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

        for instr_name, instr_par in self.exp_settings['instruments'].items():
            # Instantiate the desired instrument
            if instr_name in module_map:
                inst = module_map[instr_name](instr_par['address'])
                inst.set_all(instr_par)
                # Add to class dictionary for convenience
                setattr(self.experiment, 'instr_name', inst)
                # Add to _instruments dictionary
                self.experiment._instruments[instr_name] = inst
                logger.debug("Found instrument class for '%s' when loading experiment settings.", instr_name)
            else:
                logger.error("Could not find instrument class for '%s' when loading experiment settings.", instr_name)

    def load_channels(self):
        # Add output connectors for each defined channel
        for chan_name, chan_par in self.chan_settings['channelDict'].items():
            pass

    def load_sweeps(self):
        for chan_name, chan_par in self.exp_settings['sweeps'].items():
            pass

    def load_filters(self):
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

        for filt_name, filt_par in self.exp_settings['measurements'].items():
            filt_type = filt_par['filterType']
            if filt_type in module_map:
                filt = module_map[filt_type]()
                logger.debug("Found filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)
            else:
                logger.error("Could not find filter class %s for '%s' when loading experiment settings.", filt_type, filt_name)