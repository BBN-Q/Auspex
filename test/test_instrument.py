# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import unittest

import auspex.config as config
config.auspex_dummy_mode = True

from auspex.instruments.instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand

class TestInstrument(SCPIInstrument):
	frequency     = FloatCommand(get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
	serial_number = IntCommand(get_string="serial?")
	mode          = StringCommand(name="enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class InstrumentTestCase(unittest.TestCase):
	"""
	Tests instrument commands
	"""

	def test_properties(self):
		"""Check that property and setter/getter are implemented"""
		self.instrument = TestInstrument("DUMMY::RESOURCE")
		self.instrument.connect()
		self.assertTrue(hasattr(self.instrument, "frequency")) #property
		self.assertTrue(hasattr(self.instrument, "get_frequency") and callable(self.instrument.get_frequency)) #getter
		self.assertTrue(hasattr(self.instrument, "set_frequency") and callable(self.instrument.get_frequency)) #setter

	def test_getters(self):
		"""Check that Commands with only `get_string` have only get methods"""
		self.instrument = TestInstrument("DUMMY::RESOURCE")
		self.instrument.connect()
		self.assertTrue(hasattr(self.instrument, "serial_number")) #property
		self.assertTrue(hasattr(self.instrument, "get_serial_number") and callable(self.instrument.get_frequency)) #getter
		self.assertFalse(hasattr(self.instrument, "set_serial_number")) #setter should be deleted

	def test_allowed_values(self):
		"""Check that allowed values raises error on unallowed value."""
		self.instrument = TestInstrument("DUMMY::RESOURCE")
		self.instrument.connect()
		self.instrument.mode = "A"
		with self.assertRaises(ValueError):
			self.instrument.mode = "D"

	def test_value_range(self):
		"""Check that setting value outside range raises error."""
		self.instrument = TestInstrument("DUMMY::RESOURCE")
		self.instrument.connect()
		with self.assertRaises(ValueError):
			self.instrument.frequency = 11

	def test_locked_class(self):
		self.instrument = TestInstrument("DUMMY::RESOURCE")
		self.instrument.connect()
		with self.assertRaises(TypeError):
			self.instrument.nonexistent_property = 16

if __name__ == '__main__':
	unittest.main()
