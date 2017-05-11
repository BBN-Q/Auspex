# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['Picosecond10070A']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand

class Picosecond10070A(SCPIInstrument):
    """Picosecond 10070A Pulser"""
    amplitude      = FloatCommand(scpi_string="amplitude")
    delay          = FloatCommand(scpi_string="delay")
    duration       = FloatCommand(scpi_string="duration")
    trigger_level  = FloatCommand(scpi_string="level")
    period         = FloatCommand(scpi_string="period")
    frequency      = FloatCommand(scpi_string="frequency")
    offset         = FloatCommand(scpi_string="offset")
    trigger_source = StringCommand(scpi_string="trigger", allowed_values=["INT", "EXT", "GPIB"])

    def __init__(self, resource_name, *args, **kwargs):
        super(Picosecond10070A, self).__init__(resource_name, *args, **kwargs)
        self.name = "Picosecond 10070A Pulser"

    def connect(self, resource_name=None, interface_type=None):
        super(Picosecond10070A, self).connect(resource_name=resource_name, interface_type=interface_type)
        self.interface.write("header off")
        self.interface.write("trigger GPIB")
        self.interface._resource.read_termination = u"\n"

    # This command is syntactically screwy
    @property
    def output(self):
        return self.interface.query("enable?") == "YES"
    @output.setter
    def output(self, value):
        if value:
            self.interface.write("enable")
        else:
            self.interface.write("disable")

    def trigger(self):
        self.interface.write("*TRG")
