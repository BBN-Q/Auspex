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
import os.path
import sys
import auspex.globals
from shutil import move
from io import StringIO
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml

# Run this code by importing config.py
# Load the configuration from the json file and populate the global configuration dictionary
if auspex.globals.config_file:
    config_file = os.path.abspath(auspex.globals.config_file)
else:
    root_path   = os.path.dirname( os.path.abspath(__file__) )
    root_path   = os.path.abspath(os.path.join(root_path, "../.." ))
    config_dir  = os.path.join(root_path, 'config')
    config_file = os.path.join(config_dir, 'config.json')

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

def yaml_load(filename):
    with open(filename, 'r') as fid:
        Loader.add_constructor('!include', Loader.include)
        load = Loader(fid)
        code = load.get_single_data()
        load.dispose()
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

if not os.path.isfile(config_file):
    # build a config file from the template
    template_file = os.path.join(config_dir, 'config.example.json')
    with open(template_file, 'r') as ifid:
        template = json.load(ifid)
    cfg = {}
    for k,v in template.items():
        cfg[k] = os.path.join(root_path, v.replace("/my/path/to/", "examples/"))

    with open(config_file, 'w') as ofid:
        json.dump(cfg, ofid, indent=2)
else:
    with open(config_file, 'r') as f:
        cfg = json.load(f)

# pull out the variables
# abspath allows the use of relative file names in the config file
if auspex.globals.AWGDir:
    AWGDir = os.path.abspath(auspex.globals.AWGDir)
else:
    AWGDir = os.path.abspath(cfg['AWGDir'])
if auspex.globals.ConfigurationFile:
    configFile = os.path.abspath(auspex.globals.ConfigurationFile)
else:
    configFile = os.path.abspath(cfg['ConfigurationFile'])
if auspex.globals.KernelDir:
    KernelDir = os.path.abspath(auspex.globals.KernelDir)
else:
    KernelDir = os.path.abspath(cfg['KernelDir'])
if auspex.globals.LogDir:
    LogDir = os.path.abspath(auspex.globals.LogDir)
else:
    LogDir = os.path.abspath(cfg['LogDir'])
if not os.path.isdir(KernelDir):
    os.mkdir(KernelDir)
if not os.path.isdir(LogDir):
    os.mkdir(LogDir)

try:
    import QGL.config
    AWGDir = QGL.config.AWGDir
    configFile = QGL.config.configFile
except:
    pass
