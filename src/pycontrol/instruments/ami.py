from .instrument import Instrument, Command, FloatCommand, IntCommand


class AMI430(Instrument):
    """AMI430 Power Supply Programmer"""

    SUPPLY_TYPES = \
    ["AMI 12100PS", "AMI 12200PS", "AMI 4Q05100PS", "AMI 4Q06125PS", "AMI 4Q06250PS",
    "AMI 4Q12125PS", "AMI 10100PS", "AMI 10200PS", "HP 6260B", "Kepco BOP 20-5M",
    "Kepco BOP 20-10M", "Xantrex XFR 7.5-140", "Custom", "AMI Model 05100PS-430-601",
    "AMI Model 05200PS-430-601", "AMI Model 05300PS-430-601", "AMI Model 05400PS-430-601", "AMI Model 05500PS-430-601"]

    RAMPING_STATES = \
    ["RAMPING to target field/current", "HOLDING at the target field/current", "PAUSED",
    "Ramping in MANUAL UP mode", "Ramping in MANUAL DOWN mode", "ZEROING CURRENT (in progress)",
    "Quench detected", "At ZERO current", "Heating persistent switch", "Cooling persistent switch"]

    #Configure commands
    supply_type = Command("Supply type", get_string="SUPPly:TYPE?",
        value_map={v:str(ct) for ct,v in enumerate(SUPPLY_TYPES)})
    voltage_min    = FloatCommand("Minimum supply voltage", get_string="SUPPly:VOLTage:MINimum?")
    voltage_max    = FloatCommand("Maximum supply voltage", get_string="SUPPly:VOLTage:MAXimum")
    current_min    = FloatCommand("Minimum supply current", get_string="SUPPly:CURRent:MINimum?")
    current_max    = FloatCommand("Maximum supply current", get_string="SUPPly:CURRent:MAXimum")
    current_limit  = FloatCommand("Maximum magnitude of current (A)", set_string="CONFigure:CURRent:LIMit {:f}", get_string="CURRent:LIMit?")
    current_rating = FloatCommand("Magnet current rating (A)", set_string="CONFigure:CURRent:RATING {f}", get_string="CURRent:RATING?")
    stability      = FloatCommand("Stability setting in percent", set_string="CONFigure:STABility {:f}", get_string="STABility?", value_range=(0,100))
    coil_const     = FloatCommand("Field-to-current ratio (kG/A or T/A)", set_string="CONFigure:COILconst {:f}",get_string="COILconst?")
    persistent_switch = Command("Persistent switch installed (bool)", set_string="CONFigure:PSwitch {}",
        get_string="PSwitch:INSTalled?", value_map={False:"0", True:"1"})
    absorber = Command("Absorber installed (bool)", set_string="CONFigure:ABsorber {}", get_string="ABsorber?",
        value_map={False:"0", True:"1"})
    field_units = Command("Preferred field units (kG/T)", set_string="CONFigure:FIELD:UNITS {}",
        get_string="FIELD:UNITS?", value_map={"kG":"0", "T":"1"})

    #Ramp commands
    voltage_limit = FloatCommand("Ramping voltage limit (V)", set_string="CONFigure:VOLTage:LIMit {:f}", get_string="VOLTage:LIMit?")
    current_target = FloatCommand("Target current (A)", set_string="CONFigure:CURRent:TARGet {:f}",
        get_string="CURRent:TARGet?", value_range=(0,44.2))
    field_target = FloatCommand("Field target (kG/T)", set_string="CONFigure:FIELD:TARGet {:f}",
        get_string="FIELD:TARGet?", value_range=(0,44.2))
    ramp_num_segments = IntCommand("Number of segments for ramp", set_string="CONFigure:RAMP:RATE:SEGments {:d}",
        get_string="RAMP:RATE:SEGments?", value_range=(1,10))
    ramp_rate_units = Command("Ramp rate time unit (seconds/minutes)", set_string="CONFigure:RAMP:RATE:UNITS {}",
    get_string="RAMP:RATE:UNITS?", value_map={"seconds":"0", "minutes":"1"})

    #Current operating conditions
    voltage = FloatCommand("Voltage at supply (V)", get_string="VOLTage:SUPPly?")
    current_magnet = FloatCommand("Current at magnet", get_string="CURRent:MAGnet?")
    current_supply = FloatCommand("Current at supply", get_string="CURRent:SUPPly?")
    field = FloatCommand("Calculated magnet field", get_string="FIELD:MAGnet?")
    inductance = FloatCommand("Measured inductance (H)", get_string="INDuctance?")
    ramping_state = Command("Current ramping state", get_string="STATE?", value_map={v:str(ct+1) for ct,v in enumerate(RAMPING_STATES)})

    def __init__(self, name, resource_name, *args, **kwargs):
        resource_name += "::7180::SOCKET"
        super(AMI430, self).__init__(name, resource_name, *args, **kwargs)
        self.interface._resource.read_termination = u"\r\n"
        #device responds with 'American Magnetics Model 430 IP Interface\r\nHello\r\n' on connect
        connect_response = self.interface.read()
        assert connect_response == "American Magnetics Model 430 IP Interface"
        connect_response = self.interface.read()
        assert connect_response == "Hello."

        #Default to T field units
        self.field_units = "T"

    #TODO when we want more than one segment
    # def set_ramp_rate_current(self, segment, current, ) = FloatCommand("Ramp rate for specified segement (A/sec or A/min)", set_string="CONFigure:RAMP:RATE:CURRent {segment:d},{:f}",
    #     get_string="RAMP:RATE:CURRent:{segment:d}?", additional_args=["segment"])


    #Ramping state controls
    def ramp(self):
        """
        Places the Model 430 Programmer in automatic ramping mode. The Model
        430 will continue to ramp at the configured ramp rate(s) until the target
        field/current is achieved.
        """
        self.interface.write("RAMP")

    def pause(self):
        """
        Pauses the Model 430 Programmer at the present operating field/current.
        """
        self.interface.write("PAUSE")

    def ramp_up(self):
        """
        Places the Model 430 Programmer in the MANUAL UP ramping mode.
        Ramping continues at the ramp rate until the Current Limit is achieved.
        """
        self.interface.write("INCR")

    def ramp_down(self):
        """
        Places the Model 430 Programmer in the MANUAL DOWN ramping
        mode. Ramping continues at the ramp rate until the Current Limit is
        achieved (or zero current is achieved for unipolar power supplies).
        """
        self.interface.write("DECR")

    def zero(self):
        """
        Places the Model 430 Programmer in ZEROING CURRENT mode.
        Ramping automatically initiates and continues at the ramp rate until the
        power supply output current is less than 0.1% of Imax, at which point the
        AT ZERO status becomes active.
        """
        self.interface.write("ZERO")
