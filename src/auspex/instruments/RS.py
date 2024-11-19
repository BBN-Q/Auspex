__all__ = ['SMB100B']

import socket
import time
import copy
import re
import numpy as np
from itertools import product
from .instrument import SCPIInstrument, Command, StringCommand, BoolCommand, FloatCommand, IntCommand, is_valid_ipv4
from auspex.log import logger
import pyvisa.util as util

class SMB100B(SCPIInstrument):
    """Rohde and Schwarz SMB100B signal generator"""
    intrument_type = "Microwave Source"

    frequency   =   FloatCommand(scpi_string="SOUR1:FREQ")
    power   =   FloatCommand(scpi_string=":SOUR1:POWER")
    phase   =   FloatCommand(scpi_string=":SOUR1:PHASE")

    output  =   StringCommand(scpi_string="OUTP",value_map={True: '1',False: '0'})

    reference   =   StringCommand(scpi_string=":SOUR:ROSC:SOUR")

    def __init__(self,resource_name=None, name="SMB100B", *args,**kwargs):
        if resource_name is not None:
            self.resource_name = resource_name
        if is_valid_ipv4(self.resource_name):
            if "::hislip0" not in self.resource_name:
                self.resource_name += "::hislip0::INSTR"
        super(SMB100B, self).__init__(resource_name=self.resource_name,*args,**kwargs)

    def connect(self,resource_name=None, interface_type="VISA"):
        """Connect to the RF source via a specified physical interface. Defaults to the IP address given at instatiation and the VISA interface if these arguments are not given.

        Args:
            resource_name: IP address of RS. Defaults to None.
            interface_type: Physical interface for communication. Default is None, indicating use of PyVISA.
        Returns:
            None.
        """
        if resource_name is not None:
            if is_valid_ipv4(resource_name):
                resource_name = resource_name + "::hislip0::INSTR"
            else:
                logger.error("Invalid IP address for SMB100B: {}.".format(resource_name))
        
        super(SMB100B, self).connect(resource_name, interface_type)
        self.interface._resource.read_termination = u"\n"
        self.interface._resource.write_termination = u"\n"
    
    def set_all(self, settings):
      super(SMB100B, self).set_all(settings)


  