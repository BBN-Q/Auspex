# Copyright 2016 Raytheon BBN Technologies
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""Keithley VISA sourcemeter ported from V1."""

import time

import numpy as np

from .core import (SCPIInstrument, StringCommand, FloatCommand, IntCommand,
                   experimental, managed_property, operation)
from ..kinds import InstrumentKind

__all__ = ["Keithley2400"]


@experimental
class Keithley2400(SCPIInstrument):
    """2400 sourcemeter; mode-dependent settings use current hardware readback.

    Select a single sense function before using compliance/sense_range. Configure
    measurement before source settings to avoid the range issues noted in V1.
    """

    KIND = InstrumentKind.DC_SOURCE
    IDN_STRING = "MODEL 2400"
    SOUR_VALS = ("VOLT", "CURR")
    SENSE_VALS = ("VOLT", "CURR", "RES")
    MODE_VALS = ("SWE", "LIST", "FIX")
    source = StringCommand(scpi_string=":SOUR:FUNC", allowed_values=SOUR_VALS)
    sweep_range = StringCommand(scpi_string=":SOUR:SWE:RANG", allowed_values=("BEST", "AUTO", "FIX"))
    sweep_space = StringCommand(scpi_string=":SOUR:SWE:SPAC", allowed_values=("LIN", "LOG"))
    sweep_direction = StringCommand(scpi_string=":SOUR:SWE:DIRE", allowed_values=("UP", "DOWN"))
    sweep_abort = StringCommand(scpi_string=":SOUR:SWE:CAB", allowed_values=("NEV", "EARL", "LATE"))
    sweep_points = IntCommand(scpi_string=":SOUR:SWE:POIN")
    output = StringCommand(scpi_string=":OUTP", value_map={"ON": "1", "OFF": "0"})
    sense = StringCommand(get_string=":SENS:FUNC?", set_string=":SENS:FUNC '{:s}'", allowed_values=SENSE_VALS)
    concurrent = StringCommand(scpi_string=":SENS:FUNC:CONC", value_map={"ON": "1", "OFF": "0"})
    current = FloatCommand(get_string=":MEAS:CURR?")
    voltage = FloatCommand(get_string=":MEAS:VOLT?")
    resistance = FloatCommand(get_string=":MEAS:RES?")

    def connect(self, resource_name=None):
        """Select ASCII replies for parsing without configuring measurement/source."""
        super().connect(resource_name)
        try:
            self.interface.read_termination = "\n"
            self.interface.write("format:data ascii")
        except Exception:
            self.disconnect()
            raise

    @operation
    def triad(self, freq=440, duration=0.2, minor=False, down=False):
        """Play V1's audible notification, blocking for each note's duration."""
        ratio = 6 / 5 if minor else 5 / 4
        notes = (freq, freq * ratio, freq * 6 / 4)
        for note in reversed(notes) if down else notes:
            self.beep(note, duration)
            time.sleep(duration)

    @operation
    def beep(self, freq, dur):
        self._transport().write(f":SYST:BEEP {freq:g}, {dur:g}")

    @operation
    def sweep(self):
        """Run the configured sweep, enabling output and disabling it on success.

        Retain V1's twelve-hour read timeout. Restore the transport timeout even
        on error; a failed sweep can leave output enabled and is not retried.
        """
        io = self._transport()
        variables = self.sense.replace(":DC", "").replace('"', "").strip().split(",")
        variables.append(self.source.strip())
        io.write(":FORM:ELEM " + ",".join(variables))
        nvar = len(io.query(":FORM:ELEM?").split(","))
        points = self.sweep_points
        io.write(f":TRIG:COUN {points:d}")
        previous_timeout = io.timeout
        try:
            io.timeout = 12 * 60 * 60 * 1000
            if self.output == "OFF":
                self.output = "ON"
            result = np.array(io.query(":READ?").split(","), dtype=np.float32).reshape(points, nvar)
            self.output = "OFF"
        finally:
            io.timeout = previous_timeout
        self.beep(261.6, 0.25)
        return result

    @managed_property
    def level(self):
        return float(self._transport().query(f":SOUR:{self.source.strip()}:LEV?"))

    @level.setter
    def level(self, value):
        self._transport().write(f":SOUR:{self.source.strip()}:LEV {value:g}")

    @managed_property
    def mode(self):
        return self._transport().query(f":SOUR:{self.source.strip()}:MODE?").strip()

    @mode.setter
    def mode(self, value):
        if value not in self.MODE_VALS:
            raise ValueError(f"mode must be one of {self.MODE_VALS}")
        self._transport().write(f":SOUR:{self.source.strip()}:MODE {value}")

    def _get_range(self, prefix):
        io = self._transport()
        if int(io.query(prefix + ":RANG:AUTO?")) == 1:
            return "AUTO"
        return float(io.query(prefix + ":RANG?"))

    def _set_range(self, prefix, value):
        io = self._transport()
        if value == "AUTO":
            io.write(prefix + ":RANG:AUTO 1")
        else:
            io.write(f"{prefix}:RANG:AUTO 0;{prefix}:RANG {value:g}")

    @managed_property
    def source_range(self):
        return self._get_range(f":SOUR:{self.source.strip()}")

    @source_range.setter
    def source_range(self, value):
        self._set_range(f":SOUR:{self.source.strip()}", value)

    @managed_property
    def sweep_start(self):
        return float(self._transport().query(f":SOUR:{self.source.strip()}:STAR?"))

    @sweep_start.setter
    def sweep_start(self, value):
        self._transport().write(f":SOUR:{self.source.strip()}:STAR {value:g}")

    @managed_property
    def sweep_stop(self):
        return float(self._transport().query(f":SOUR:{self.source.strip()}:STOP?"))

    @sweep_stop.setter
    def sweep_stop(self, value):
        self._transport().write(f":SOUR:{self.source.strip()}:STOP {value:g}")

    @managed_property
    def sweep_step(self):
        return float(self._transport().query(f":SOUR:{self.source.strip()}:STEP?"))

    @sweep_step.setter
    def sweep_step(self, value):
        self._transport().write(f":SOUR:{self.source.strip()}:STEP {value:g}")

    def _sense_function(self):
        return self.sense.replace('"', "").strip()

    @managed_property
    def compliance(self):
        return float(self._transport().query(f":SENS:{self._sense_function()}:PROT?"))

    @compliance.setter
    def compliance(self, value):
        self._transport().write(f":SENS:{self._sense_function()}:PROT {value:g}")

    @managed_property
    def sense_range(self):
        return self._get_range(f":SENS:{self._sense_function()}")

    @sense_range.setter
    def sense_range(self, value):
        self._set_range(f":SENS:{self._sense_function()}", value)

    @operation
    def conf_meas_res(self, NPLC=1, res_range=1000.0, auto_range=True):
        """Explicitly configure resistance measurement before source configuration."""
        io = self._transport()
        io.write(f':sens:func "res";:sens:res:mode man;:sens:res:nplc {NPLC:f};:form:elem res;')
        if auto_range:
            io.write(":sens:res:rang:auto 1;")
        else:
            io.write(f":sens:res:rang:auto 0;:sens:res:rang {res_range:g}")
