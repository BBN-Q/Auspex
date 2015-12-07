from .instrument import Instrument, Command, FloatCommand

class DPO72004C(Instrument):
    """Tektronix DPO72004C Oscilloscope"""

    def __init__(self, name, resource_name, *args, **kwargs):
        resource_name += "::4000::SOCKET" #user guide recommends HiSLIP protocol
        super(DPO72004C, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\n"
