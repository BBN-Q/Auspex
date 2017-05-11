# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

__all__ = ['LakeShore370', 'LakeShore335']

from .instrument import SCPIInstrument, StringCommand, FloatCommand, IntCommand, Command
import numpy as np

def indexed_map(values):
    return {v: '%d' % i for i, v in enumerate(values)}

class LakeShore370(SCPIInstrument):
    """Lakeshore 370 AC Resistance Bridge"""

# Allowed values
    SENS_CHAN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    RAMP_MIN = 0.001
    RAMP_MAX = 10
    P_MIN = 0.001
    P_MAX = 1000
    I_MAX = 10000
    D_MAX = 2500
    MAX_TEMP = 0.3 #for safety, since this is most likely connected to a DR, do not let user set a very high temperature setpoint
    HEATER_VALUES = [0, 31.6e-6, 100e-6, 316e-6, 1e-3, 3.16e-3, 10e-3, 31.6e-3, 100e-3] #amperes
    HEATER_RES_MIN = 1 #ohms
    HEATER_RES_MAX = 100000 #ohms
    
    HEATER_RANGE_MAP = indexed_map(HEATER_VALUES)
    HEATER_UNITS_MAP = {'Kelvin': 1, 'Ohms': 2}
    HEATER_DISPLAY_MAP = {'Current': 1, 'Power': 2}
    
# Generic init and connect methods
    def __init__(self, resource_name=None, *args, **kwargs):
        super(LakeShore370, self).__init__(resource_name, *args, **kwargs)
        self.name = "LakeShore 370 Resistance Bridge"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        super(LakeShore370, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\r\n"

# Input validation
    def check_channel(self, chan):
        if chan not in self.SENS_CHAN:
            raise Exception("Channel {} is not in the allowed channel list.".format(chan))
    
    def check_ramp_rate(self, rate):
        if rate < self.RAMP_MIN or rate > self.RAMP_MAX:
            raise Exception("Ramp rate {} is outside of allowed range: [{}, {}]".format(rate, self.RAMP_MIN, self.RAMP_MAX))
    
    def check_pid(self, P, I, D):
        if abs(P) < self.P_MIN or abs(P) > self.P_MAX:
            raise Exception("PID P value {} is outside of allowed range.".format(P))
        if abs(I) > self.I_MAX:
            raise Exception("PID I value {} is outside of allowed range.".format(I))
        if abs(D) > self.D_MAX:
            raise Exception("PID D value {} is outside of allowed range.".format(D)) 
    
    def check_setp(self, T):
        if T > self.MAX_TEMP:
            raise Exception("Setpoint temperature is greater than allowed maximum. Change MAX_TEMP if you really want to do this.")
    
    def check_resistance(self, R):
        if R < self.HEATER_RES_MIN or R > self.HEATER_RES_MAX:
            raise Exception('Heater resistance is outside of allowed range: (%d, %d)'%(HEATER_RES_MIN, HEATER_RES_MAX))
    
# Commands

    heater_range = Command(get_string="HTRRNG?", set_string="HTRRNG {:s}", value_map=HEATER_RANGE_MAP)
    heater_status = IntCommand(get_string="HTRST?")
    heater_output = FloatCommand(get_string="HTR?")
    control_mode = StringCommand(get_string="CMODE?", set_string="CMODE {:s}",
        value_map={"PID": '1', "Zone": '2', "OpenLoop": '3', "Off": '4'})
    heater_setting = FloatCommand(scpi_string="MOUT")

    def heater_off(self):
        """Turn off heater.
        Args:
            None.
        Returns:
            None.
        """
        self.heater_range = 0

    
    def temp(self, chan):
        """Get Lakshore temperature reading for a specific channel.
        
        Args:
            chan: Channel to be queried.
        Returns:
            temp: Channel temperature in Kelvin.
        """
        self.check_channel(chan)
        return float(self.interface.query("RDGK? %d"%chan))
        
    def resistance(self, chan):
        """Get Lakshore resistance reading for a specific channel.
        
        Args:
            chan: Channel to be queried.
        Returns:
            temp: Channel resistance in Ohms.
        """
        self.check_channel(chan)
        return float(self.interface.query("RDGR? %d"%chan))
    
    @property 
    def setpoint(self):
        """Get current temperature setpoint."""
        return float(self.interface.query("SETP?"))
    
    @setpoint.setter
    def setpoint(self, T):
        """Set temperature control setpoint."""
        self.check_setp(T)
        self.interface.write("SETP %f"%T)
    
    @property
    def ramp_state(self):
        """Check if ramping to temperature setpoint."""
        return bool(int(self.interface.query("RAMPST?")))
    
    @property
    def ramp(self):
        """Get setpoint ramp status or value. Returns 0 if ramping is off."""
        ans = self.interface.query("RAMP?").split(",")
        if int(ans[0]) == 0:
            return 0
        else:
            return float(ans[1])
    
    @ramp.setter
    def ramp(self, rate):
        """Set temperature setpoint ramp rate.
        
        Args:
            rate: Setpoint ramp rate, in Kelvin/minute. Set to 0 to turn off ramping.
        """
        if rate == 0:
            self.interface.write("RAMP 0, %f"%self.RAMP_MIN)
        else:
            self.check_ramp_rate(rate)
            self.interface.write("RAMP 1, %f"%rate)
    
    @property

    def pid(self):
        """Get PID values. Returns tuple (P,I,D)."""
        return tuple(map(float, self.interface.query("PID?").split(",")))
    
    def set_pid(self, P, I, D):
        """Set PID parameters."""
        self.check_pid(P, I, D)
        self.interface.write("PID %f, %f, %f"%(P, I, D))
     
    @property
    def control_setup(self):
        """Get current temperature control setup.
        Args:
            None.
        Returns:
            setup: (Channel, Filtered or Unfiltered, Setpoint units, 
                autoscan delay (seconds), display units, current limit, resistance)
        """
        
        ans = self.interface.query("CSET?").split(',')
        channel = int(ans[0])
        filter = bool(ans[1])
        units = {v: k for k, v in self.HEATER_UNITS_MAP.items()}[int(ans[2])]
        delay = int(ans[3])
        display = {v: k for k, v in self.HEATER_DISPLAY_MAP.items()}[int(ans[4])]
        limit = {v: k for k, v in self.HEATER_RANGE_MAP.items()}[ans[5]]
        resistance = float(ans[6])
        return (channel, filter, units, delay, display, limit, resistance)
    
    def set_control_setup(self, channel, limit, resistance, units='Kelvin', delay=10., 
        filter=True, display='Power'):
        """Set up the heater control parameters.
        
        Args:
            channel: Channel to control, 1-16.
            limit: Heater output current limit.
            resistance: Heater resistance.
            units: Heater setpoint units. Kelvin or Ohms.
            delay: Delay in seconds for setpoint change during autoscanning: 1-255 seconds.
            filter: Control on filtered or unfilitered readings.
            display: Heater output display. Power or Current.
        Returns:
            None.
        """       
        self.check_channel(channel)
        self.check_resistance(resistance)
        if limit not in self.HEATER_RANGE_MAP.keys():
            raise Exception('Allowed heater limit currents: %s.'%list(HEATER_RANGE_MAP.keys()))
        if units not in self.HEATER_UNITS_MAP.keys():
            raise Exception('Allowed output units: Kelvin or Ohms.')
        if display not in self.HEATER_DISPLAY_MAP.keys():
            raise Exception('Allowed heater scale units: Power or Current.')
        if delay < 1 or delay > 255:
            raise Exception('Setpoint autoscan delay must be tween 1-255 seconds.')
            
        self.interface.write("CSET %d, %d, %d, %d, %d, %d, %3.3f"%(
            channel,
            int(filter),
            self.HEATER_UNITS_MAP[units],
            delay,
            self.HEATER_DISPLAY_MAP[display],
            int(self.HEATER_RANGE_MAP[limit]),
            resistance))
            
        
        
    

class LakeShore335(SCPIInstrument):
    """LakeShore 335 Temperature Controller"""

# Allowed value arrays

    T_VALS          = ['A', 'B']
    SENTYPE_VALS    = [0, 1, 2, 3, 4]
    ZO_VALS         = [0, 1]
    R_VALS          = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    UNIT_VALS       = [1, 2, 3]
    HTR_VALS        = [0, 1, 2, 3, 4, 5]
    HTR_CTR_VALS    = [0, 1, 2]
    HTR_RNG_VALS    = [0, 1, 2, 3]


# Generic init and connect methods

    def __init__(self, resource_name=None, *args, **kwargs):
        super(LakeShore335, self).__init__(resource_name, *args, **kwargs)
        self.name = "LakeShore 335 Temperature Controller"

    def connect(self, resource_name=None, interface_type=None):
        if resource_name is not None:
            self.resource_name = resource_name
        super(LakeShore335, self).connect(resource_name=self.resource_name, interface_type=interface_type)
        self.interface._resource.read_termination = u"\r\n"

# Message checkers
    def check_sense_msg(self, vals):
        if len(vals) != 5: 
            raise Exception("Invalid number of parameters. Must specify: Sensor Type, Auto Range Option, Range, Compensation Option, Units")
        if vals[0] not in self.SENTYPE_VALS: 
            raise Exception("Invalid sensor type:\n0 = Disabled\n1 = Diode\n2 = Platinum RTD\n 3 = NTC RTD\n 4 = Thermocouple")
        if vals[1] not in self.ZO_VALS:
            raise Exception("Invalid autoranging option:\n0 = OFF\n1 = ON")
        if vals[2] not in self.R_VALS: 
            raise Exception("Invalid range specificed. See Lake Shore 335 Manual")
        if vals[3] not in self.ZO_VALS:
            raise Exception("Invalid compenstation option:\n0 = OFF\n1 = ON")
        if vals[4] not in self.UNIT_VALS:
            raise Exception("Invalid units specified:\n1 = Kelvin\n2 = Celsius\n3 = Sensor (Ohms or Volts)")

    def check_htr_msg(self, vals):
        if len(vals) != 3: 
            raise Exception("Invalid number of parameters. Must specify: Control Mode, Input, Powerup Enable")
        if vals[0] not in self.HTR_VALS:
            raise Exception("Invalid Control mode:\n0 = OFF\n1 = PID Loop\n2 = Zone\n3 = Open Loop\n4 = Monitor Out\n5 = Warmup Supply")
        if vals[1] not in self.HTR_CTR_VALS:
            raise Exception("Invalid Control input:\n0 = None\n1 = A\n2 = B")
        if vals[2] not in self.ZO_VALS:
            raise Exception("Invalid Powerup Enable mode:\n0 = OFF\n1 = ON")

    def check_hconf_msg(self, vals):
        if len(vals) != 5: 
            raise Exception("Invalid number of parameters. Must specify: Output Type, Heater Resistance, Max Heater current, Max User Current, Output Display")
        if vals[0] not in self.ZO_VALS:
            raise Exception("Invalid Output type:\n0 = Current\n1 = Voltage")
        if vals[1] not in self.UNIT_VALS or vals[1]==3:
            raise Exception("Invalid Heater resistance:\n0 = 25 Ohm\n1 = 50 Ohm")
        if vals[2] not in self.SENTYPE_VALS:
            raise Exception("Invalid Max current:\n0 = User Specified\n1 = 0.707 A\n2 = 1 A\n3 = 1.141 A\n 4 = 1.732 A")
        if 1<vals[3] and vals[2]==2:
            raise Exception("Max current is 1 A for 50 Ohm Heater")
        if 1.414<vals[3] and vals[0]==0:
            raise Exception("Max current is 1.414 A for Current output")
        if 1.732<vals[3] and vals[0]==1:
            raise Exception("Max current is 1.732 A for Voltage output")
        if vals[4] not in self.UNIT_VALS or vals[4]==3:
            raise Exception("Invalid Output Display:\n1 = Current\n2 = Power")


    def check_pid_msg(self, vals):
        if len(vals) != 3: 
            raise Exception("Invalid number of parameters. Must specify: P, I, D")
        if vals[0]<0.1 or 1000<vals[0]:
            raise Exception("Invalid Proportional (gain) range. 0.1 < P < 1000")
        if vals[1]<0.1 or 1000<vals[1]:
            raise Exception("Invalid Integral (reset) range. 0.1 < I < 1000")
        if vals[2]<0 or 200<vals[2]:
            raise Exception("Invalid Derivative (rate) range. 0 < D < 200")

    def check_range_msg(self, val):
        if val not in self.HTR_RNG_VALS:
            raise Exception("Invalid Range setting:\n0 = Off\n1 = Low\n2 = Medium\n3 = High")

# Read Temperature

    def Temp(self,sense='A'):
        if sense not in self.T_VALS: 
            raise Exception("Must read sensor A or B")
        else: 
            return float(self.interface.query("KRDG? {}".format(sense)))

# Configure T senses

    @property
    def  config_sense_A(self):
        return self.interface.query_ascii_values("INTYPE? A",converter=u'd')
       

    @config_sense_A.setter
    def config_sense_A(self,vals):
        self.check_sense_msg(vals)
        self.interface.write(("INTYPE A,"+','.join(['{:d}']*len(vals))).format(*vals))

    @property
    def  config_sense_B(self):
        return self.interface.query_ascii_values("INTYPE? B",converter=u'd')
       

    @config_sense_B.setter
    def config_sense_B(self, vals):
        self.check_sense_msg(vals)
        self.interface.write(("INTYPE B,"+','.join(['{:d}']*len(vals))).format(*vals))

# Heater and PID Control

    @property
    def  config_htr_1(self):
        return self.interface.query_ascii_values("HTRSET? 1",converter=u'e')
       

    @config_htr_1.setter
    def config_htr_1(self, vals):
        self.check_hconf_msg(vals)
        self.interface.write("HTRSET 1,{:d},{:d},{:d},{:.3f},{:d}".format(int(vals[0]),int(vals[1]),int(vals[2]),vals[3],int(vals[4])))

    @property
    def  config_htr_2(self):
        return self.interface.query_ascii_values("HTRSET? 2",converter=u'e')
       

    @config_htr_2.setter
    def config_htr_2(self, vals):
        self.check_hconf_msg(vals)
        self.interface.write("HTRSET 2,{:d},{:d},{:d},{:.3f},{:d}".format(int(vals[0]),int(vals[1]),int(vals[2]),vals[3],int(vals[4])))
    
    @property
    def  control_htr_1(self):
        return self.interface.query_ascii_values("OUTMODE? 1",converter=u'd')
       

    @control_htr_1.setter
    def control_htr_1(self, vals):
        self.check_htr_msg(vals)
        self.interface.write(("OUTMODE 1,"+','.join(['{:d}']*len(vals))).format(*vals))

    @property
    def  control_htr_2(self):
        return self.interface.query_ascii_values("OUTMODE? 2",converter=u'd')
       

    @control_htr_2.setter
    def control_htr_2(self, vals):
        self.check_htr_msg(vals)
        self.interface.write(("OUTMODE 2,"+','.join(['{:d}']*len(vals))).format(*vals))

    @property
    def  pid_htr_1(self):
        return self.interface.query_ascii_values("PID? 1",converter=u'e')
       

    @pid_htr_1.setter
    def pid_htr_1(self, vals):
        self.check_pid_msg(vals)
        self.interface.write(("PID 1,"+','.join(['{:.1f}']*len(vals))).format(*vals))

    @property
    def  pid_htr_2(self):
        return self.interface.query_ascii_values("PID? 2",converter=u'e')
       

    @pid_htr_2.setter
    def pid_htr_2(self, vals):
        self.check_pid_msg(vals)
        self.interface.write(("PID 2,"+','.join(['{:.1f}']*len(vals))).format(*vals))

    @property
    def  temp_htr_1(self):
        return self.interface.query_ascii_values("SETP? 1",converter=u'e')
       
    @temp_htr_1.setter
    def temp_htr_1(self, val):
        self.interface.write("SETP 1,{:.3f}".format(val)) 

    @property
    def  temp_htr_2(self):
        return self.interface.query_ascii_values("SETP? 2",converter=u'e')
       

    @temp_htr_2.setter
    def temp_htr_2(self, val):
        self.interface.write("SETP 2,{:.3f}".format(val)) 

    @property
    def range_htr_1(self):
        return self.interface.query_ascii_values("RANGE? 1",converter=u'd')

    @range_htr_1.setter
    def range_htr_1(self,val):
        self.check_range_msg(val)
        self.interface.write("RANGE 1,{:d}".format(val))

    @property
    def range_htr_2(self):
        return self.interface.query_ascii_values("RANGE? 2",converter=u'd')

    @range_htr_2.setter
    def range_htr_2(self,val):
        self.check_range_msg(val)
        self.interface.write("RANGE 2,{:d}".format(val))

    @property
    def mout_htr_1(self):
        return self.interface.query_ascii_values("MOUT? 1",converter=u'e')

    @mout_htr_1.setter
    def mout_htr_1(self,val):
        if val<0 or 100<val: 
            raise Exception("Manual Heater output must be 0 - 100 %")
        self.interface.write("MOUT 1,{:.2f}".format(val))

    @property
    def mout_htr_2(self):
        return self.interface.query_ascii_values("MOUT? 2",converter=u'e')

    @mout_htr_2.setter
    def mout_htr_2(self,val):
        if val<0 or 100<val: 
            raise Exception("Manual Heater output must be 0 - 100 %")
        self.interface.write("MOUT 2,{:.2f}".format(val))

