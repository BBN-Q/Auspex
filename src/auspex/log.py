# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import logging

logger = logging.getLogger('auspex')
logging.basicConfig(format='%(name)s-%(levelname)s: %(asctime)s ----> %(message)s')
logger.setLevel(logging.INFO)
