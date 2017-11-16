# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# This file is originally from PyQLab (http://github.com/bbn-q/PyQLab)

import json
import os, os.path
import sys
import auspex.globals
from shutil import move
from io import StringIO
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml

meas_file = None
AWGDir    = None
KernelDir = None
LogDir    = None

def find_meas_file():
    # First default to any manually set options in the globals
    if auspex.globals.meas_file:
        return os.path.abspath(auspex.globals.meas_file)
    # Next use the meas file location in the environment variables
    if os.getenv('BBN_CFG_FILE'):
        return os.getenv('BBN_CFG_FILE')
    raise Exception("Could not find the measurement file in the environment variables or the auspex globals.")

class Include():
    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as f:
            self.data = yaml.load(f, Loader=yaml.RoundTripLoader)
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def items(self):
        return self.data.items()
    def keys(self):
        return self.data.keys()
    def write(self):
        with open(self.filename+".tmp", 'w') as fid:
            yaml.dump(self.data, fid, Dumper=yaml.RoundTripDumper)
        move(self.filename+".tmp", self.filename)
    def pop(self, key):
        return self.data.pop(key)

class Loader(yaml.RoundTripLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def include(self, node):
        shortname = self.construct_scalar(node)
        filename = os.path.abspath(os.path.join(
            self._root, shortname
        ))
        return Include(filename)

class Dumper(yaml.RoundTripDumper):
    def include(self, data):
        data.write()
        return self.represent_scalar(u'!include', data.filename)

class FlatDumper(yaml.RoundTripDumper):
    def include(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data.data)

def load_meas_file(filename):
    global LogDir, KernelDir, AWGDir, meas_file

    meas_file = filename

    with open(filename, 'r') as fid:
        Loader.add_constructor('!include', Loader.include)
        load = Loader(fid)
        code = load.get_single_data()
        load.dispose()

    # Get the config values out of the measure_file, but override with 
    # any auspex.globals that are manually set.
    # abspath allows the use of relative file names in the config file
    if auspex.globals.AWGDir:
        AWGDir = os.path.abspath(auspex.globals.AWGDir)
    else:
        AWGDir = os.path.abspath(code['config']['AWGDir'])

    if auspex.globals.KernelDir:
        KernelDir = os.path.abspath(auspex.globals.KernelDir)
    else:
        KernelDir = os.path.abspath(code['config']['KernelDir'])

    if auspex.globals.LogDir:
        LogDir = os.path.abspath(auspex.globals.LogDir)
    else:
        LogDir = os.path.abspath(code['config']['LogDir'])
    
    if not os.path.isdir(KernelDir):
        os.mkdir(KernelDir)
    if not os.path.isdir(LogDir):
        os.mkdir(LogDir)

    return code

def yaml_dump(data, filename = "", flatten=False):
    d = Dumper if filename and not flatten else FlatDumper
    d.add_representer(Include, d.include)

    if filename:
        with open(filename+".tmp", 'w+') as fid:

            yaml.dump(data, fid, Dumper=d)
        # Upon success
        move(filename+".tmp", filename)
        with open(filename, 'r') as fid:
            contents = fid.read()
        return contents
    else:
        # dump to an IO stream:
        # note you need to use the FlatDumper for this to work
        out = StringIO()
        yaml.dump(data, out, Dumper=d)
        ret_string = out.getvalue()
        out.close()
        return ret_string
