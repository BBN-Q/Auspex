import pkgutil
import importlib
from auspex.log import logger

for loader, name, is_pkg in pkgutil.iter_modules(__path__):
	logger.debug("Import module: auspex.instruments.%s" %name)
	module = importlib.import_module('auspex.instruments.' + name)
	if hasattr(module, "__all__"):
		logger.debug("Move the following attributes from module %s to auspex.instruments namespace: %s" %(name,module.__all__))
		globals().update((n, getattr(module, n)) for n in module.__all__)
