import unittest

from pycontrol.instruments.instrument import Instrument, Command, FloatCommand, IntCommand
from pycontrol.procedure import Procedure, FloatParameter, Quantity

class TestInstrument1(Instrument):
	frequency = FloatCommand("frequency", get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
	serial_number = IntCommand("serial number", get_string="serial?")
	mode = Command("enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument2(Instrument):
	frequency = FloatCommand("frequency", get_string="frequency?", set_string="frequency {:g}", value_range=(0.1, 10))
	serial_number = IntCommand("serial number", get_string="serial?")
	mode = Command("enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestInstrument3(Instrument):
	power = FloatCommand("power", get_string="power?")
	serial_number = IntCommand("serial number", get_string="serial?")
	mode = Command("enumerated mode", scpi_string=":mode", allowed_values=["A", "B", "C"])

class TestProcedure(Procedure):

    # Create instances of instruments
    fake_instr_1 = TestInstrument1("Fake Instrument 1", "FAKE::RESOURE::NAME")
    fake_instr_2 = TestInstrument2("Fake Instrument 2", "FAKE::RESOURE::NAME")
    fake_instr_3 = TestInstrument3("Fake Instrument 2", "FAKE::RESOURE::NAME")

    # Parameters
    freq_1 = FloatParameter("Frequency 1", unit="Hz")
    freq_2 = FloatParameter("Frequency 2", unit="Hz")

    # Quantities
    power = Quantity("Power", unit="Watts")
    clout = Quantity("Clout", unit="Trumps")

    def run(self):
        pass

class ProcedureTestCase(unittest.TestCase):
	"""
	Tests procedure class
	"""

	def setUp(self):
		pass

	def test_parameters(self):
		"""Check that parameters have been appropriately gathered"""
		self.assertTrue(hasattr(TestProcedure, "_parameters")) # should have parsed these parameters from class dir
		self.assertTrue(len(TestProcedure._parameters) == 2 ) # should have parsed these parameters from class dir
		self.assertTrue(TestProcedure._parameters['freq_1'] == TestProcedure.freq_1) # should contain this parameter
		self.assertTrue(TestProcedure._parameters['freq_2'] == TestProcedure.freq_2) # should contain this parameter

	def test_quantities(self):
		"""Check that quantities have been appropriately gathered"""
		self.assertTrue(hasattr(TestProcedure, "_quantities")) # should have parsed these quantities from class dir
		self.assertTrue(len(TestProcedure._quantities) == 2 ) # should have parsed these quantities from class dir
		self.assertTrue(TestProcedure._quantities['power'] == TestProcedure.power) # should contain this quantity
		self.assertTrue(TestProcedure._quantities['clout'] == TestProcedure.clout) # should contain this quantity

	def test_instruments(self):
		"""Check that instruments have been appropriately gathered"""
		self.assertTrue(hasattr(TestProcedure, "_instruments")) # should have parsed these instruments from class dir
		self.assertTrue(len(TestProcedure._instruments) == 3 ) # should have parsed these instruments from class dir
		self.assertTrue(TestProcedure._instruments['fake_instr_1'] == TestProcedure.fake_instr_1) # should contain this instrument
		self.assertTrue(TestProcedure._instruments['fake_instr_2'] == TestProcedure.fake_instr_2) # should contain this instrument
		self.assertTrue(TestProcedure._instruments['fake_instr_3'] == TestProcedure.fake_instr_3) # should contain this instrument

if __name__ == '__main__':
	unittest.main()