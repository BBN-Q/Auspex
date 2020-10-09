import pkgutil
import importlib
import visa
import os

instrument_map = {}
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
	module = importlib.import_module('auspex.instruments.' + name)
	if hasattr(module, "__all__"):
		globals().update((name, getattr(module, name)) for name in module.__all__)
		for name in module.__all__:
			instrument_map.update({name:getattr(module,name)})

def enumerate_visa_instruments():
	try:
		if os.name == "nt":
			visa_loc = 'C:\\windows\\system32\\visa64.dll'
			rm = visa.ResourceManager(visa_loc)
		else:
			rm = visa.ResourceManager("@py")
	except Exception as e:
		raise Exception(f"Unable to open VISA library with exception: {str(e)}")
	print(rm.list_resources())

def probe_instrument_ids():
	try:
		if os.name == "nt":
			visa_loc = 'C:\\windows\\system32\\visa64.dll'
			rm = visa.ResourceManager(visa_loc)
		else:
			rm = visa.ResourceManager("@py")
	except Exception as e:
		raise Exception(f"Unable to open VISA library with exception: {str(e)}")
	for instr_label in rm.list_resources():
		instr = rm.open_resource(instr_label)
		try:
			print(instr_label, instr.query('*IDN?'))
		except:
			print(instr_label, "Did not respond")
		instr.close()
