# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand
import numpy as np

class LakeShore335(SCPIInstrument):
	"""LakeShore 335 Temperature Controller"""

# Allowed value arrays

	T_VALS          = ['A', 'B']
	SENTYPE_VALS    = [0, 1, 2, 3, 4]
	ZO_VALS         = [0, 1]
	R_VALS          = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	UNIT_VALS       = [1, 2, 3]
	HTR_VALS        = [0, 1, 2, 3, 4, 5]
	HTR_CTR_VALS	= [0, 1, 2]
	HTR_RNG_VALS	= [0, 1, 2, 3]


# Generic init and connect methods

	def __init__(self, resource_name=None, *args, **kwargs):
		super(LakeShore335, self).__init__(resource_name, *args, **kwargs)
		self.name = "LakeShore 335 Temperature Controller"

	def connect(self, resource_name=None, interface_type=None):
		if resource_name is not None:
			self.resource_name = resource_name
		super(LakeShore335, self).connect(resource_name=self.resource_name, interface_type=interface_type)
		self.interface._resource.read_termination = u"\r\n"

# Message checkers
	def check_sense_msg(self, vals):
		if len(vals) != 5: 
			raise Exception("Invalid number of parameters. Must specify: Sensor Type, Auto Range Option, Range, Compensation Option, Units")
		if vals[0] not in self.SENTYPE_VALS: 
			raise Exception("Invalid sensor type:\n0 = Disabled\n1 = Diode\n2 = Platinum RTD\n 3 = NTC RTD\n 4 = Thermocouple")
		if vals[1] not in self.ZO_VALS:
			raise Exception("Invalid autoranging option:\n0 = OFF\n1 = ON")
		if vals[2] not in self.R_VALS: 
			raise Exception("Invalid range specificed. See Lake Shore 335 Manual")
		if vals[3] not in self.ZO_VALS:
			raise Exception("Invalid compenstation option:\n0 = OFF\n1 = ON")
		if vals[4] not in self.UNIT_VALS:
			raise Exception("Invalid units specified:\n1 = Kelvin\n2 = Celsius\n3 = Sensor (Ohms or Volts)")

	def check_htr_msg(self, vals):
		if len(vals) != 3: 
			raise Exception("Invalid number of parameters. Must specify: Control Mode, Input, Powerup Enable")
		if vals[0] not in self.HTR_VALS:
			raise Exception("Invalid Control mode:\n0 = OFF\n1 = PID Loop\n2 = Zone\n3 = Open Loop\n4 = Monitor Out\n5 = Warmup Supply")
		if vals[1] not in self.HTR_CTR_VALS:
			raise Exception("Invalid Control input:\n0 = None\n1 = A\n2 = B")
		if vals[2] not in self.ZO_VALS:
			raise Exception("Invalid Powerup Enable mode:\n0 = OFF\n1 = ON")

	def check_hconf_msg(self, vals):
		if len(vals) != 5: 
			raise Exception("Invalid number of parameters. Must specify: Output Type, Heater Resistance, Max Heater current, Max User Current, Output Display")
		if vals[0] not in self.ZO_VALS:
			raise Exception("Invalid Output type:\n0 = Current\n1 = Voltage")
		if vals[1] not in self.UNIT_VALS or vals[1]==3:
			raise Exception("Invalid Heater resistance:\n0 = 25 Ohm\n1 = 50 Ohm")
		if vals[2] not in self.SENTYPE_VALS:
			raise Exception("Invalid Max current:\n0 = User Specified\n1 = 0.707 A\n2 = 1 A\n3 = 1.141 A\n 4 = 1.732 A")
		if 1<vals[3] and vals[2]==2:
			raise Exception("Max current is 1 A for 50 Ohm Heater")
		if 1.414<vals[3] and vals[0]==0:
			raise Exception("Max current is 1.414 A for Current output")
		if 1.732<vals[3] and vals[0]==1:
			raise Exception("Max current is 1.732 A for Voltage output")
		if vals[4] not in self.UNIT_VALS or vals[4]==3:
			raise Exception("Invalid Output Display:\n1 = Current\n2 = Power")


	def check_pid_msg(self, vals):
		if len(vals) != 3: 
			raise Exception("Invalid number of parameters. Must specify: P, I, D")
		if vals[0]<0.1 or 1000<vals[0]:
			raise Exception("Invalid Proportional (gain) range. 0.1 < P < 1000")
		if vals[1]<0.1 or 1000<vals[1]:
			raise Exception("Invalid Integral (reset) range. 0.1 < I < 1000")
		if vals[2]<0 or 200<vals[2]:
			raise Exception("Invalid Derivative (rate) range. 0 < D < 200")

	def check_range_msg(self, val):
		if val not in self.HTR_RNG_VALS:
			raise Exception("Invalid Range setting:\n0 = Off\n1 = Low\n2 = Medium\n3 = High")

# Read Temperature

	def Temp(self,sense='A'):
		if sense not in self.T_VALS: 
			raise Exception("Must read sensor A or B")
		else: 
			return float(self.interface.query("KRDG? {}".format(sense)))

# Configure T senses

	@property
	def  config_sense_A(self):
		return self.interface.query_ascii_values("INTYPE? A",converter=u'd')
	   

	@config_sense_A.setter
	def config_sense_A(self,vals):
		self.check_sense_msg(vals)
		self.interface.write(("INTYPE A,"+','.join(['{:d}']*len(vals))).format(*vals))

	@property
	def  config_sense_B(self):
		return self.interface.query_ascii_values("INTYPE? B",converter=u'd')
	   

	@config_sense_B.setter
	def config_sense_B(self, vals):
		self.check_sense_msg(vals)
		self.interface.write(("INTYPE B,"+','.join(['{:d}']*len(vals))).format(*vals))

# Heater and PID Control

	@property
	def  config_htr_1(self):
		return self.interface.query_ascii_values("HTRSET? 1",converter=u'e')
	   

	@config_htr_1.setter
	def config_htr_1(self, vals):
		self.check_hconf_msg(vals)
		self.interface.write("HTRSET 1,{:d},{:d},{:d},{:.3f},{:d}".format(int(vals[0]),int(vals[1]),int(vals[2]),vals[3],int(vals[4])))

	@property
	def  config_htr_2(self):
		return self.interface.query_ascii_values("HTRSET? 2",converter=u'e')
	   

	@config_htr_2.setter
	def config_htr_2(self, vals):
		self.check_hconf_msg(vals)
		self.interface.write("HTRSET 2,{:d},{:d},{:d},{:.3f},{:d}".format(int(vals[0]),int(vals[1]),int(vals[2]),vals[3],int(vals[4])))
	
	@property
	def  control_htr_1(self):
		return self.interface.query_ascii_values("OUTMODE? 1",converter=u'd')
	   

	@control_htr_1.setter
	def control_htr_1(self, vals):
		self.check_htr_msg(vals)
		self.interface.write(("OUTMODE 1,"+','.join(['{:d}']*len(vals))).format(*vals))

	@property
	def  control_htr_2(self):
		return self.interface.query_ascii_values("OUTMODE? 2",converter=u'd')
	   

	@control_htr_2.setter
	def control_htr_2(self, vals):
		self.check_htr_msg(vals)
		self.interface.write(("OUTMODE 2,"+','.join(['{:d}']*len(vals))).format(*vals))

	@property
	def  pid_htr_1(self):
		return self.interface.query_ascii_values("PID? 1",converter=u'e')
	   

	@pid_htr_1.setter
	def pid_htr_1(self, vals):
		self.check_pid_msg(vals)
		self.interface.write(("PID 1,"+','.join(['{:.1f}']*len(vals))).format(*vals))

	@property
	def  pid_htr_2(self):
		return self.interface.query_ascii_values("PID? 2",converter=u'e')
	   

	@pid_htr_2.setter
	def pid_htr_2(self, vals):
		self.check_pid_msg(vals)
		self.interface.write(("PID 2,"+','.join(['{:.1f}']*len(vals))).format(*vals))

	@property
	def  temp_htr_1(self):
		return self.interface.query_ascii_values("SETP? 1",converter=u'e')
	   

	@temp_htr_1.setter
	def temp_htr_1(self, val):
		self.interface.write("SETP 1,{:.3f}".format(val)) 

	@property
	def  temp_htr_2(self):
		return self.interface.query_ascii_values("SETP? 2",converter=u'e')
	   

	@temp_htr_2.setter
	def temp_htr_2(self, val):
		self.interface.write("SETP 2,{:.3f}".format(val)) 

	@property
	def range_htr_1(self):
		return self.interface.query_ascii_values("RANGE? 1",converter=u'd')

	@range_htr_1.setter
	def range_htr_1(self,val):
		self.check_range_msg(val)
		self.interface.write("RANGE 1,{:d}".format(val))

	@property
	def range_htr_2(self):
		return self.interface.query_ascii_values("RANGE? 2",converter=u'd')

	@range_htr_2.setter
	def range_htr_2(self,val):
		self.check_range_msg(val)
		self.interface.write("RANGE 2,{:d}".format(val))

