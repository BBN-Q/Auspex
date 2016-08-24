# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import json
import os.path
import sys

import pycontrol.config as config
from pycontrol.experiment import Experiment

class QubitExpFactory(object):
	"""The purpose of this factory is to examine ExpSettings.json""" 
