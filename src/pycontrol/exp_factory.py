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

import pycontrol.config as config
from pycontrol.experiment import Experiment
import pycontrol.instruments

class QubitExpFactory(object):
    """The purpose of this factory is to examine ExpSettings.json""" 
    def __init__(self):
        with open(config.expSettingsFile, 'r') as FID: 
            self.exp_settings = json.load(FID)

            self.experiment = Experiment()

            self.load_instruments()
            self.load_channels()
            self.load_measurements()

    def load_instruments():
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

        for instr_name, instr_par in self.exp_settings['instruments'].items()
            # Instantiate the desired instrument
            inst = module_map[instr_name](instr_par['address'])
            inst.set_all(instr_par)
            # Add to class dictionary for convenience
            setattr(self.experiment, 'instr_name', inst)
            # Add to _instruments dictionary
            self.experiment._instruments[instr_name] = inst

    def load_channels():
        pass

    def load_measurements():
        pass