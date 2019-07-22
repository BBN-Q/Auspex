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

# Profiling
profile = False

# Use when wanting to generate fake data
# or to avoid loading libraries that may
# interfere with desired operation. (e.g.
# when scraping modules in Auspex)
auspex_dummy_mode = False

# Set generator from qubit sidebanding
# This sets the generator frequency based on 
# the requested qubit frequency and sidebanding.
qubit_IF_priority = False

ConfigurationFile = None
LogDir            = None

# The db file, where the channel libraries are stored
db_file        = None

def load_db():
    global db_file
    if os.getenv('BBN_DB'):
        db_file = os.getenv("BBN_DB")

def isnotebook():
	# https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter