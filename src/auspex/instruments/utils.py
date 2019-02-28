import importlib
import pkgutil
import inspect

from . import bbn
import auspex.config
from auspex.log import logger
from auspex.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument, ReceiverChannel

def correct_resource_name(resource_name):
    substs = {"USB::": "USB0::", }
    for k, v in substs.items():
        resource_name = resource_name.replace(k, v)
    return resource_name

def pulse_marker(mkr, length = 100e-9):
    """ Utility to generate a square pulse on a APS2 marker. Used for instance to switch a signal between spectrum analyzer and input line
    marker_name"""

    from QGL import TRIG
    from QGL.Compiler import compile_to_hardware

    APS = bbn.APS2(mkr.phys_chan.transmitter.address)
    APS.connect()
    APS.set_trigger_source('Software')
    seq = [[TRIG(mkr, length)]]
    APS.set_sequence_file(compile_to_hardware(seq, 'Switch/Switch').replace('meta.json', mkr.phys_chan.transmitter.label+'.aps2'))
    APS.run()
    APS.trigger()
    APS.stop()
    APS.disconnect()
    logger.info('Switched marker {} ({})'.format(mkr.label, mkr))


# class InstrumentFactory(object):
#     """Utility class to load an instrument by name from a measurement file."""
#
#     def __new__(cls, name, set_all=True, meas_file=None):
#
#         settings = auspex.config.load_meas_file(meas_file)
#
#         modules = (
#             importlib.import_module('auspex.instruments.' + name)
#             for loader, name, is_pkg in pkgutil.iter_modules(auspex.instruments.__path__)
#         )
#
#         module_map = {}
#         for mod in modules:
#             instrs = (_ for _ in inspect.getmembers(mod) if inspect.isclass(_[1]) and
#                                                             issubclass(_[1], Instrument) and
#                                                             _[1] != Instrument and _[1] != SCPIInstrument and
#                                                             _[1] != CLibInstrument)
#             module_map.update(dict(instrs))
#
#         try:
#             instr_par = settings['instruments'][name]
#         except KeyError as e:
#             e.args = ('Could not find instrument {} in measurement file: {}'.format(e.args[0],
#                                                             auspex.config.find_meas_file()))
#             raise
#
#         instr_type = instr_par['type']
#         instr_par['name'] = name
#         if instr_type in module_map:
#             logger.debug("Found instrument class %s for '%s' at loc %s when loading experiment settings.", instr_type, name, instr_par['address'])
#             try:
#                 inst = module_map[instr_type](correct_resource_name(str(instr_par['address'])), name=name)
#             except Exception as e:
#                 logger.error("Initialization of caused exception:", name, str(e))
#                 inst = None
#
#         if inst:
#             inst.connect()
#             if set_all:
#                 inst.set_all(instr_par)
#
#         return inst
