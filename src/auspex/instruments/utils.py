import importlib
import pkgutil
import inspect

from . import bbn
import auspex.config
from auspex.log import logger
# from auspex.instruments.instrument import Instrument, SCPIInstrument, CLibInstrument, ReceiverChannel

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
