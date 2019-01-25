# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# This file is originally from PyQLab (http://github.com/bbn-q/PyQLab)

import os, os.path
import sys
from shutil import move
from io import StringIO
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml

# Use when wanting to generate fake data
# or to avoid loading libraries that may
# interfere with desired operation. (e.g.
# when scraping modules in Auspex)
auspex_dummy_mode = False

# If this is True, then close the last
# plotter before starting a new one.
single_plotter_mode = False

# This holds a reference to the most
# recent plotters.
last_plotter_process = None
last_extra_plotter_process = None

# Config directory
meas_file         = None
AWGDir            = None
ConfigurationFile = None
KernelDir         = None
LogDir            = None


def find_meas_file():
    global meas_file
    # First default to any manually set options in the globals
    if meas_file:
        return os.path.abspath(meas_file)
    # Next use the meas file location in the environment variables
    if os.getenv('BBN_MEAS_FILE'):
        return os.getenv('BBN_MEAS_FILE')
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

def load_meas_file(filename=None):
    global LogDir, KernelDir, AWGDir, meas_file

    if filename:
        meas_file = filename
    else:
        meas_file = find_meas_file()

    with open(meas_file, 'r') as fid:
        Loader.add_constructor('!include', Loader.include)
        load = Loader(fid)
        code = load.get_single_data()
        load.dispose()

    # Get the config values out of the measure_file.
    if not 'config' in code.keys():
        raise KeyError("Could not find config section of the yaml file.")

    if 'AWGDir' in code['config'].keys():
        AWGDir = os.path.abspath(code['config']['AWGDir'])
    else:
        raise KeyError("Could not find AWGDir in the YAML config section")

    if 'KernelDir' in code['config'].keys():
        KernelDir = os.path.abspath(code['config']['KernelDir'])
    else:
        raise KeyError("Could not find KernelDir in the YAML config section")

    if 'LogDir' in code['config'].keys():
        LogDir = os.path.abspath(code['config']['LogDir'])
    else:
        raise KeyError("Could not find LogDir in the YAML config section")

    # Create directories if necessary
    for d in [KernelDir, LogDir]:
        if not os.path.isdir(d):
            os.mkdir(d)

    return code

def dump_meas_file(data, filename = "", flatten=False):
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
