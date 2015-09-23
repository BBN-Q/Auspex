import numpy as np
import time

currents = np.arange(-20, 20.1, 2)
voltages = []
fields = []

for c in np.arange(-20, 20.1, 2):
	print "Setting field to", c
	bop.current = c
	time.sleep(4)
	field = float(input("Field value: "))
	fields.append(field)
	voltages.append(np.mean([lock.ai1 for i in range(20)]))

print currents
print fields
print voltages