# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import logging
import sys
import importlib

def in_jupyter():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

logger = logging.getLogger('auspex')

if in_jupyter():
    importlib.reload(logging)
    logger.handlers = [logging.StreamHandler(sys.stdout)]
    formatter = logging.Formatter('%(name)s-%(levelname)s: %(asctime)s ----> %(message)s')
    logger.handlers[0].setFormatter(formatter)
else:
	logging.basicConfig(format='%(name)s-%(levelname)s: %(asctime)s ----> %(message)s')

logger.setLevel(logging.INFO)
