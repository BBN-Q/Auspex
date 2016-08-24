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

# Run this code by importing config.py
# Load the configuration from the json file and populate the global configuration dictionary
rootFolder    = os.path.dirname( os.path.abspath(__file__) )
rootFolder    = os.path.abspath(os.path.join(rootFolder, "../.." ))
rootFolder    = rootFolder.replace('\\', '/') # use unix-like convention
configFolder  = os.path.join(rootFolder, 'config')
PyQLabCfgFile = os.path.join(configFolder, 'config.json')

if not os.path.isfile(PyQLabCfgFile):
	# build a config file from the template
	templateFile = os.path.join(configFolder, 'config.example.json')
	ifid = open(templateFile, 'r')
	ofid = open(PyQLabCfgFile, 'w')
	for line in ifid:
		ofid.write(line.replace('/my/path/to', configFolder))
	ifid.close()
	ofid.close()


with open(PyQLabCfgFile, 'r') as f:
	PyQLabCfg = json.load(f)

# pull out the variables
# abspath allows the use of relative file names in the config file
AWGDir = os.path.abspath(PyQLabCfg['AWGDir'])
instrumentLibFile = os.path.abspath(PyQLabCfg['InstrumentLibraryFile'])
sweepLibFile = os.path.abspath(PyQLabCfg['SweepLibraryFile'])
measurementLibFile = os.path.abspath(PyQLabCfg['MeasurementLibraryFile'])
quickpickFile = os.path.abspath(PyQLabCfg['QuickPickFile']) if 'QuickPickFile' in PyQLabCfg else ''
