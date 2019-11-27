# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['BNC845']

from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, is_valid_ipv4, BoolCommand
from auspex.log import logger
import time
import numpy as np
import socket

class BNC845(SCPIInstrument):
    """SCPI instrument driver for Berkely Nucleonics BNC845-M RF Signal Generator.

    Properties:
        frequency: Set the RF generator frequency, in Hz. 0.01-20 GHz.
        power: Set the RF generator output power, in dBm. No effect, BNC845M output is always +16dBm.
        output: Toggle RF signal output on/off.
        pulse: Toggle RF pulsed mode on/off.
        alc: Toggle source Auto Leveling on/off.
        mod: Toggle amplitude modulation on/off.
        pulse_source: Set pulse trigger to INTERNAL or EXTERNAL.
        freq_source: Set frequency source to INTERNAL or EXTERNAL.
    """

    frequency = FloatCommand(scpi_string="SOURCE:FREQUENCY:FIXED")
    power     = FloatCommand(scpi_string="SOURCE:POWER:LEVEL:IMMEDIATE:AMPLITUDE")
    output    = BoolCommand(scpi_string="OUTPUT:STATE",value_map={True: '1', False: '0'})
    pulse     = StringCommand(scpi_string="PULSE", value_map={True: '0', False: '1'})
    mod       = StringCommand(scpi_string="MOD", value_map={True: '0', False: '1'})
    alc       = StringCommand(scpi_string="SOURCE:POWER:ALC ", value_map={True: '0', False: '1'})
    pulse_source = StringCommand(scpi_string=":PULSE:SOUR",
                          value_map={'INTERNAL': 'INT', 'EXTERNAL': 'EXT'})
    freq_source  = StringCommand(scpi_string=":FREQ:SOUR",
                          value_map={'INTERNAL': 'INT', 'EXTERNAL': 'EXT'})
    instrument_type = "Microwave Source"
    reference = "10MHz"

    def __init__(self, resource_name=None, *args, **kwargs):
        """Berkely Nucleonics BNC845-M RF Signal Generator

        Args:
            resource_name: The IP address of the source to conenct to, as string.
        """
        if resource_name is not None:
            if is_valid_ipv4(resource_name):
                resource_name = resource_name + "::inst0::INSTR"
            else:
                logger.error("Invalid IP address for BNC845: {}.".format(resource_name))
        super(BNC845, self).__init__(resource_name, *args, **kwargs)

    def connect(self, resource_name=None, interface_type="VISA"):
        """Connect to the RF source via a specified physical interface. Defaults
        to the IP address given at instatiation and the VISA interface if these
        arguments are not given.

        Args:
            resource_name: IP address of BNC. Defaults to None.
            interface_type: Physical interface for communication. Default is None,
                indicating use of PyVISA.
        Returns:
            None.
        """
        if resource_name is not None:
            if is_valid_ipv4(resource_name):
                resource_name = resource_name + "::inst0::INSTR"
            else:
                logger.error("Invalid IP address for BNC845: {}.".format(resource_name))
        super(BNC845, self).connect(resource_name, interface_type)
        self.interface._resource.read_termination = '\n'
        self.interface._resource.write_termination = '\n'
        # Setup the reference every time
        # Output 10MHz for daisy-chaining and lock to 10MHz external
        # reference
        self.output = True
        self.interface.write('SOURCE:ROSC:EXT:FREQ 10E6')
        self.interface.write('SOUR:ROSC:SOUR EXT')
        # Check that it locked -- it does not lock.
        for ct in range(10):
            locked = self.interface.query('SOURCE:ROSC:LOCKED?')
            logger.debug("Lock attempt {}: {}".format(ct, locked))
            if locked == '1':
                break
            time.sleep(0.5)
        if locked != '1':
            logger.warning('BNC845 at %s is unlocked.', self.resource_name.split("::")[1]);
