import numpy as np
from scipy.optimize import curve_fit
import time

from instruments.kepco import BOP2020M
from instruments.stanford import SR830

bop       = BOP2020M("Kepco Power Supply", "GPIB1::1::INSTR")
lock      = SR830("Lockin Amplifier", "GPIB1::9::INSTR")

currents = np.arange(-20, 20.1, 2)
voltages = []
fields = []

bop.output = True

for c in np.arange(-20, 20.1, 2):
	print "Setting current to", c
	bop.current = c
	print bop.current
	time.sleep(4)
	field = float(input("Field value: "))
	fields.append(field)
	voltages.append(np.mean([lock.ai1 for i in range(20)]))

print currents
print fields
print voltages

np.savetxt('TempCalibration.txt', np.transpose(np.array([currents, fields, voltages])))

p_hall = np.polyfit(voltages, fields, 1)
print p_hall
p_gmw = np.polyfit(fields, currents, 1)
print p_gmw
p_gmw = np.polyfit(currents, fields, 1)
print p_gmw
