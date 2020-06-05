
#!/usr/bin/env python


#Grabbed from github instead of reinventing a wheel

# Author: Kearney Lackas
# References
#  - http://www.batronix.com/pdf/Rigol/ProgrammingGuide/DP800_ProgrammingGuide_EN.pdf
#  - https://github.com/freq0ut/Python-PyVisa
#  - http://juluribk.com/2015/05/08/controlling-rigol-dp832-with-python/
# Instructions
# - Download and install National Instruments VISA software (https://www.ni.com/visa/)
# - Download and install PyVISA (eg. "pip install -U pyvisa" from command line)

from visa import *
import time

__all__ = ['DP832']


_delay = 0.01  # in seconds


class DP832:
    def __init__(self, address):
        try:
            self.rm = ResourceManager()
            # self.instrument_list = self.rm.list_resources()

            self.address = address

            if self.address.__len__() == 0:
                self.status = "Not Connected"
                # print("Could not connect to device")
            else:
                # self.address = self.address[0]
                self.device = self.rm.open_resource(self.address)
                # print("Connected to " + self.address)
                self.status = "Connected"
                self.connected_with = 'USB'

        except VisaIOError:
            self.status = "Not Connected"
            # print("PyVISA is not able to find any devices")

    def select_output(self, chan):
        # define a CHANNEL SELECT function
        command = ':INST:NSEL %s' % chan
        self.device.write(command)
        time.sleep(_delay)

    def toggle_output(self, chan, state):
        # define a TOGGLE OUTPUT function
        if state in ['ON', 'OFF']:
            command = ':OUTP CH%s,%s' % (chan, state)
            self.device.write(command)
            time.sleep(_delay)
        else:
            print("invalid command. Must be: ['ON','OFF']")

    def set_voltage(self, chan, val):
        # define a SET VOLTAGE function
        command = ':INST:NSEL %s' % chan
        self.device.write(command)
        time.sleep(_delay)
        command = ':VOLT %s' % val
        self.device.write(command)
        time.sleep(_delay)

    def set_current(self, chan, val):
        # define a SET CURRENT function
        command = ':INST:NSEL %s' % chan
        self.device.write(command)
        time.sleep(_delay)
        command = ':CURR %s' % val
        self.device.write(command)
        time.sleep(_delay)

    def set_ovp(self, chan, val):
        # define a SET VOLT PROTECTION function
        command = ':INST:NSEL %s' % chan
        self.device.write(command)
        time.sleep(_delay)
        command = ':VOLT:PROT %s' % val
        self.device.write(command)
        time.sleep(_delay)

    def toggle_ovp(self, state):
        # define a TOGGLE VOLTAGE PROTECTION function
        command = ':VOLT:PROT:STAT %s' % state
        self.device.write(command)
        time.sleep(_delay)

    def set_ocp(self, chan, val):
        # define a SET CURRENT PROTECTION function
        command = ':INST:NSEL %s' % chan
        self.device.write(command)
        time.sleep(_delay)
        command = ':CURR:PROT %s' % val
        self.device.write(command)
        time.sleep(_delay)

    def toggle_ocp(self, state):
        # define a TOGGLE CURRENT PROTECTION function
        command = ':CURR:PROT:STAT %s' % state
        self.device.write(command)
        time.sleep(_delay)

    def measure_voltage(self, chan):
        # define a MEASURE VOLTAGE function
        command = ':MEAS:VOLT? CH%s' % chan
        volt = self.device.query(command)
        volt = float(volt)
        time.sleep(_delay)
        return volt

    def measure_current(self, chan):
        # define a MEASURE CURRENT function
        command = ':MEAS:CURR? CH%s' % chan
        curr = self.device.query(command)
        curr = float(curr)
        time.sleep(_delay)
        return curr

    def measure_power(self, chan):
        # define a MEASURE POWER function
        command = ':MEAS:POWE? CH%s' % chan
        power = self.device.query(command)
        power = float(power)
        time.sleep(_delay)
        return power