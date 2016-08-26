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
root_path      = os.path.dirname( os.path.abspath(__file__) )
root_path      = os.path.abspath(os.path.join(root_path, "../.." ))
config_dir    = os.path.join(root_path, 'config')
config_file     = os.path.join(config_dir, 'config.json')

if not os.path.isfile(config_file):
	# build a config file from the template
	template_file = os.path.join(config_dir, 'config.example.json')
	with open(template_file, 'r') as ifid:
		template = json.load(ifid)
	cfg = {}
	for k,v in template.items():
		cfg[k] = os.path.join(config_dir, v.replace("/my/path/to/", ""))

	with open(config_file, 'w') as ofid:
		json.dump(cfg, ofid, indent=2)
else:
	with open(config_file, 'r') as f:
		cfg = json.load(f)

# pull out the variables
# abspath allows the use of relative file names in the config file
AWGDir             = os.path.abspath(cfg['AWGDir'])
instrumentLibFile  = os.path.abspath(cfg['InstrumentLibraryFile'])
channelLibFile     = os.path.abspath(cfg['ChannelLibraryFile'])
sweepLibFile       = os.path.abspath(cfg['SweepLibraryFile'])
measurementLibFile = os.path.abspath(cfg['MeasurementLibraryFile'])
expSettingsFile    = os.path.abspath(cfg['ExpSettingsFile'])
