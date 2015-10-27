from .instrument import Instrument, VisaInterface
from types import MethodType

class Attenuator(Instrument):
    NUM_CHANNELS = 3

    """BBN 3 Channel Instrument"""
    def __init__(self, name, resource_name):
        super(Attenuator, self).__init__(name, resource_name, interface_type="VISA")
        self.name = name
        self.interface._resource.baud_rate = 115200
        self.interface._resource.read_termination = u"\r\n"
        self.interface._resource.write_termination = u"\n"

        #Clear "unknown command" from connect
        #TODO: where the heck does this come from
        # self.interface.read()
        # self.interface.read()

        #Override query to look for ``end``
        def query(self, query_string):
            val = self._resource.query(query_string)
            assert self.read() == "END"
            return val

        self.interface.query = MethodType(query, self.interface, VisaInterface)

    def get_attenuation(self, chan):
        return float(self.interface.query("GET {:d}".format(chan)))

    def set_attenuation(self, chan, val):
        self.interface.write("SET {:d} {:.1f}".format(chan, val))
        assert self.interface.read() == "Setting channel {:d} to {:.2f}".format(chan, val)
        assert self.interface.read() == "END"
