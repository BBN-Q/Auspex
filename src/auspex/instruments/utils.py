from . import bbn
import auspex.config
from auspex.log import logger
# from QGL import *

def pulse_marker(marker_name, length = 100e-9):
    """ Utility to generate a square pulse on a APS2 marker. Used for instance to switch a signal between spectrum analyzer and input line
    marker_name as defined in measure.yaml """

    ChannelLibrary()

    settings =  auspex.config.load_meas_file(auspex.config.find_meas_file())
    mkr = settings['markers'][marker_name]
    marker = MarkerFactory(marker_name)
    APS_name = mkr.split()[0]
    APS = bbn.APS2()
    APS.connect(settings['instruments'][APS_name]['address'])
    APS.set_trigger_source('Software')
    seq = [[TRIG(marker,length)]]
    APS.set_seq_file(compile_to_hardware(seq, 'Switch\Switch').replace('meta.json', APS_name+'.h5'))
    APS.run()
    APS.trigger()
    APS.stop()
    APS.disconnect()
    logger.info('Switched marker {} ({})'.format(marker_name, mkr))
