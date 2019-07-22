import pkgutil
import importlib
import pyvisa

instrument_map = {}
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
	module = importlib.import_module('auspex.instruments.' + name)
	if hasattr(module, "__all__"):
		globals().update((name, getattr(module, name)) for name in module.__all__)
		for name in module.__all__:
			instrument_map.update({name:getattr(module,name)})

def enumerate_visa_instruments():
	rm = pyvisa.ResourceManager("@py")
	print(rm.list_resources())

def probe_instrument_ids():
	rm = pyvisa.ResourceManager("@py")
	for instr_label in rm.list_resources():
		instr = rm.open_resource(instr_label)
		try:
			print(instr_label, instr.query('*IDN?'))
		except:
			print(instr_label, "Did not respond")
		instr.close()
