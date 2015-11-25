import unittest

from pycontrol.instruments.instrument import Instrument, FloatCommand, IntCommand

class TestInstrument(Instrument):
	frequency = FloatCommand("frequency", get_string="frequency?", set_string="frequency {:g}")
	serial_number = IntCommand("serial number", get_string="serial?")

class InstrumentTestCase(unittest.TestCase):
	"""
	Tests instrument commands
	"""

	def setUp(self):
		self.instrument = TestInstrument("testing instrument", "DUMMY::RESOURCE")

	def test_properties(self):
		"""Check that property and setter/getter are implemented"""
		self.assertTrue(hasattr(self.instrument, "frequency")) #property
		self.assertTrue(hasattr(self.instrument, "get_frequency") and callable(self.instrument.get_frequency)) #getter
		self.assertTrue(hasattr(self.instrument, "set_frequency") and callable(self.instrument.get_frequency)) #setter

	def test_getters(self):
		"""Check that Commands with only `get_string` have only get methods"""
		self.assertTrue(hasattr(self.instrument, "serial_number")) #property
		self.assertTrue(hasattr(self.instrument, "get_serial_number") and callable(self.instrument.get_frequency)) #getter
		self.assertFalse(hasattr(self.instrument, "set_serial_number")) #setter should be deleted

if __name__ == '__main__':
	unittest.main()
