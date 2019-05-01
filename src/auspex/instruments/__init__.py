import pkgutil
import importlib

instrument_map = {}
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
	module = importlib.import_module('auspex.instruments.' + name)
	if hasattr(module, "__all__"):
		globals().update((name, getattr(module, name)) for name in module.__all__)
		for name in module.__all__:
			instrument_map.update({name:getattr(module,name)})
