__all__ = ['HDO6104']

from auspex.log import logger
from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, Command
import numpy as np
import time

class HDO6104(SCPIInstrument):
    channel_enabled = Command(scpi_string="C{channel}:TRA",
            additional_args=["channel"],value_map={True:"ON",False:"OFF"})
    sample_points = IntCommand(scpi_string="MEMORY_SIZE")
    trig_mode = StringCommand(scpi_string="TRIG_MODE")
    time_div = FloatCommand(scpi_string="TIME_DIV")
    trig_delay = FloatCommand(scpi_string="TRIG_DELAY")
    vol_div = Command(scpi_string="C{channel}:VOLT_DIV",additional_args=["channel"])
    vol_offset = Command(scpi_string="C{channel}:OFFSET",additional_args=["channel"])


    def connect(self, resource_name=None, interface_type=None):
        super(HDO6104,self).connect(resource_name=resource_name,interface_type=interface_type)
        self.interface.write("COMM_HEADER OFF")
        self.interface._resource.read_termination = u"\n"

    def get_info(self,channel=1):
        raw_info = self.interface.query("C%d:INSPECT? WAVEDESC" %channel).split("\r\n")[1:-1]
        info = [item.split(':') for item in raw_info]
        return {k[0].strip(): k[1].strip() for k in info}

    def fetch_waveform(self,channel):
        # Send the MSB first
        self.interface.write("COMM_ORDER HI")
        self.interface.write("COMM_FORMAT DEF9,WORD,BIN")
        mydict = self.get_info(channel=channel)
        points = int(mydict["PNTS_PER_SCREEN"])
        xincrement = float(mydict["HORIZ_INTERVAL"])
        xorigin = float(mydict["HORIZ_OFFSET"])
        yincrement = float(mydict["VERTICAL_GAIN"])
        yorigin = float(mydict["VERTICAL_OFFSET"])
        # Read waveform data
        y_axis = np.array(self.interface.query_binary_values('C%d:WAVEFORM? DAT1' % channel, datatype='h', is_big_endian=True))
        y_axis = y_axis*yincrement - yorigin
        x_axis = xorigin + np.arange(0, xincrement*len(y_axis), xincrement)
        return x_axis, y_axis
