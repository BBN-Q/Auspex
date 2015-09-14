from instruments.instrument import Instrument, Command

class AgilentWhatever(Instrument):
	power = Command("power", get_string=":pow?", set_string=":pow %g dbm;")
	frequency = Command("frequency", get_string=":freq?", set_string=":freq %g Hz;")
	current = Command("current", get_string=":curr?")


uwSource = AgilentWhatever()
