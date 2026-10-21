__all__ = ['libchannelizer']

import pkgutil
import importlib

stream_sel_map = {}
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
	module = importlib.import_module('auspex.filters.' + name)
	if hasattr(module, "__all__"):
		globals().update((name, getattr(module, name)) for name in module.__all__)
		for name in module.__all__:
			stream_sel_map.update({name:getattr(module,name)})
