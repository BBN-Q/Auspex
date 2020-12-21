Instrument Documentation
========================

Instruments
************

The *Instrument* class is designed to dramatically reduce the amount of boilerplate code required for defining device drivers. The following (from *picosecond.py*) amounts to the majority of the driver definition::

    class Picosecond10070A(SCPIInstrument):
        """Picosecond 10070A Pulser"""
        amplitude      = FloatCommand(scpi_string="amplitude")
        delay          = FloatCommand(scpi_string="delay")
        duration       = FloatCommand(scpi_string="duration")
        trigger_level  = FloatCommand(scpi_string="level")
        period         = FloatCommand(scpi_string="period")
        frequency      = FloatCommand(scpi_string="frequency",
                          aliases=['freq'])
        offset         = FloatCommand(scpi_string="offset")
        trigger_source = StringCommand(scpi_string="trigger",
                          allowed_values=["INT", "EXT", "GPIB"])

        def trigger(self):
            self.interface.write("*TRG")

Each of the Commands is converted into the relevant driver code as detailed below. 

Commands
########

The *Command* class variables are parsed by the *MetaInstrument* metaclass, and automatically expanded into setters and getters (as appropriate) and a *property* that gives convenient access to commands. For example, the following *Command*::

    frequency = FloatCommand(scpi_string='frequency')

will be expanded into the following equivalent set of class methods::

    def get_frequency(self):
        return float(self.interface.query('frequency?'))
    def set_frequency(self, value):
        self.interface.write('frequency {:E}'.format(value))
    @property
    def frequency(self):
        return self.get_frequency()
    @frequency.setter
    def frequency(self, value):
        self.set_frequency(value)

Instruments with consistent command syntax (which number fewer than one might hope) lend themselves to extremely concise drivers. Using additional keyword arguments such as ``allowed_values``, ``aliases``, and ``value_map`` allows for more advanced commands to specified without the usual driver fluff. Full documentation can be found in the API reference. 

Property Access
###############

Property access gives us a convenient way of interacting with instrument values. In the following example we construct an instance of the ``Picosecond10070A`` class and fire off a number of pulses::

    pspl = Picosecond10070A("GPIB0::24::INSTR")

    pspl.amplitude = 0.944                  # Using setter
    print("Trigger delay is: ", pspl.delay) # Using getter

    for dur in 1e-9*np.arange(1, 11, 0.5):
        pspl.duration = dur
        pspl.trigger()
        time.sleep(0.05)

Properties present certain risks alongside their convenience: running ``instr.falter_slop = 18.0`` will produce no errors (since it's perfectly reasonable Python) despite the user having intended to set the ``filter_slope`` value. As such, we actually lock the class dictionary after parsing and intilization, and will produce errors informing you of your spelling creativities. 
