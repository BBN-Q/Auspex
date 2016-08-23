# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import datetime
import time

from instruments.kepco import BOP2020M
from instruments.stanford import SR830, SR865
from instruments.magnet import Electromagnet
from instruments.hall_probe import HallProbe

from bokeh.io import output_file, show, hplot
from bokeh.plotting import figure

# Data containers
fields   = []
currents = []
times    = []

# Define Instruments
bop         = BOP2020M("GPIB1::1::INSTR")
lock        = SR830("GPIB1::9::INSTR")
hp          = HallProbe("calibration/HallProbe.cal", lock.set_ao1, lock.get_ai1)
mag         = Electromagnet('calibration/GMW.cal', hp.get_field, bop.set_current, bop.get_current)

# This will ramp the current slowly
bop.current = 5
time.sleep(8)

# Initiate immediate current change after recording time
bop.interface.write(":CURR:LEV:IMM 5.2")
loop_start = datetime.datetime.now()

for i in range(50):
	times.append( (datetime.datetime.now()-loop_start).total_seconds() )
	currents.append(float(bop.interface.query("MEAS:CURR?") ))
	fields.append(mag.field)

time.sleep(1)
times.append( (datetime.datetime.now()-loop_start).total_seconds() )
currents.append(float(bop.interface.query("MEAS:CURR?") ))
fields.append(mag.field)

# Initiate immediate current change after recording time
bop.interface.write(":CURR:LEV:IMM 5.0")

for i in range(50):
	times.append( (datetime.datetime.now()-loop_start).total_seconds() )
	currents.append(float(bop.interface.query("MEAS:CURR?") ))
	fields.append(mag.field)

time.sleep(1)
times.append( (datetime.datetime.now()-loop_start).total_seconds() )
currents.append(float(bop.interface.query("MEAS:CURR?") ))
fields.append(mag.field)

# Initiate immediate current change after recording time
bop.interface.write(":CURR:LEV:IMM 5.2")

for i in range(50):
	times.append( (datetime.datetime.now()-loop_start).total_seconds() )
	currents.append(float(bop.interface.query("MEAS:CURR?") ))
	fields.append(mag.field)

bop.current = 0

p1 = figure(title="Magnetic Field vs. Time", x_axis_label='Time (s)', y_axis_label='Field (G)')
p1.circle(times, fields, color="firebrick")
p1.line(times, fields, color="firebrick")

p2 = figure(title="BOP Current vs. Time", x_axis_label='Time (s)', y_axis_label='BOP Sense Current (A)')
p2.circle(times, currents, color="firebrick")
p2.line(times, currents, color="firebrick")

q = hplot(p1, p2)
output_file("BOP-Plots.html")
show(q)

