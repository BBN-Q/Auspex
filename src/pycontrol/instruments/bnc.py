# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import Instrument, StringCommand, FloatCommand, IntCommand
import time
import numpy as np

class BNC845(Instrument):
    """For controlling the BNC845 microwave source"""

    frequency = FloatCommand(scpi_string="SOURCE:FREQUENCY:FIXED")
    power     = FloatCommand(scpi_string="SOURCE:POWER:LEVEL:IMMEDIATE:AMPLITUDE")

    output    = StringCommand(scpi_string="SOURCE:ROSC:OUTPUT:STATE",value_map={True: 'ON', False: 'OFF'})
    pulse     = StringCommand(scpi_string="PULSE", value_map={True: 'ON', False: 'OFF'})
    mod       = StringCommand(scpi_string="MOD", value_map={True: 'ON', False: 'OFF'})
    alc       = StringCommand(scpi_string="SOURCE:POWER:ALC ", value_map={True: 'ON', False: 'OFF'})

    pulse_source = StringCommand(scpi_string=":PULSE:SOUR", 
                          value_map={'INTERNAL': 'INT', 'EXTERNAL': 'EXT'})
    freq_source  = StringCommand(scpi_string=":FREQ:SOUR",
                          value_map={'INTERNAL': 'INT', 'EXTERNAL': 'EXT'})

    def __init__(self, resource_name, *args, **kwargs):
        super(BNC845, self).__init__(resource_name, *args, **kwargs)

        # Setup the reference every time
        # Output 10MHz for daisy-chaining and lock to 10MHz external
        # reference
        self.output = True
        self.interface.write('SOURCE:ROSC:EXT:FREQ 10E6')
        self.interface.write('SOUR:ROSC:SOUR EXT')
        
        # Check that it locked
        ct = 0;
        while ct < 10:
            locked = self.interface.query('SOURCE:ROSC:LOCKED?')
            if locked == '1':
                break
            time.sleep(0.5)
            ct = ct + 1
        if locked != '1':
            logging.error('BNC845 at %s is unlocked.', resource_name);
