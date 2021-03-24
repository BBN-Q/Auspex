import time
import usbtmc
from visa import *

__all__ = ['DP832']


_delay = 0.01  # in seconds


class DP832:
    def __init__(self, address):
        self.rm = ResourceManager()
        self.address = address
        self.status = "Not Connected"

    def connect(self):
#        try:
        self.device = self.rm.open_resource(self.address)
        self.status = "Connected"
        self.connected_with = "USB"
#        except:
#            print("Failed to connect")
    
    def disconnect(self):
        try: 
            self.device.close()
            return True
        except:
            print("Failed to disconnect")
            return False
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

    def toggle_all(self, state):
    # define a TOGGLE OUTPUT function
        if state in ['ON', 'OFF']:
            self.toggle_output(1,state)
            self.toggle_output(2,state)
            self.toggle_output(3,state)
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

    def check_output(self, chan):
        command = f":OUTP? CH{chan}"
        val = self.device.query(command)
        time.sleep(_delay)
        return val

    def check_ocp(self, chan=None):
        if chan is not None:
            self.select_output(chan)
        command = ':CURR:PROT:TRIP?'
        val = self.device.query(command)
        time.sleep(_delay)
        return val

    def check_ovp(self, chan=None):
        if chan is not None:
            self.select_output(chan)
        command = ':VOLT:PROT:TRIP?'
        val = self.device.query(command)
        time.sleep(_delay)
        return val

    def test_fan(self):
        command = ':SYST:SELF:TEST:FAN?'
        val = self.device.query(command)
        time.sleep(_delay)
        return val

    def test_temp(self):
        command = ':SYST:SELF:TEST:TEMP?'
        val = self.device.query(command)
        time.sleep(_delay)
        return val

