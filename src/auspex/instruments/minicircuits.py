"""
  Driver for Mini-Circuits instruments.

  Include:
    RC-2SP4T-A18 MUX switching box
"""

from math import log2
from auspex.log import logger
from .instrument import Instrument
import http.client

class HTTPInstrument(Instrument):
    def __init__(self,resource_name=None,name="HTTP Instrument",instrument_type="HTTP"):
        """ Control for HTTP Instruments

        resource_name: Network IP address and port, e.g. 100.100.10.10:80
        """
        self.resource_name = resource_name
        self.name = name
        self.instrument_type = instrument_type
        self.interface_type = "HTTP"
        self._resource = None

    def connect(self,resource_name=None):
        """ Connect to the MUX via Ethernet HTTP

        resource_name: Network IP address and port, e.g. 100.100.10.10:80
        """
        if resource_name is not None:
            self.resource_name = resource_name
        if self.resource_name is None:
            self._resource = None
            logger.error("Failed setting up connection to %s. resource_name is not provided." %self.name)
            return False
        try:
            logger.debug("HTTP connect to %s at %s" %(self.name,self.resource_name))
            self._resource = http.client.HTTPConnection(self.resource_name)
            # Test connection
            logger.debug("Test connection to %s" %self.resource_name)
            self._resource.request("GET","/")
            res = self._resource.getresponse()
            if res.status == 200:
                logger.info("Successfully set up connection to %s" %self.resource_name)
                data = res.read()
                return True
            else:
                self._resource = None
                logger.error("For some reason, failed setting up connection to %s" %self.resource_name)
                return False
        except Exception as ex:
            self._resource = None
            logger.error("Failed setting up connection to %s. Exception: %s" %(self.resource_name,ex))
            return False

    def disconnect(self):
        if self._resource is not None:
            self._resource.close()
            logger.info("Disconnected %s from %s" %(self.name,self.resource_name))
        else:
            logger.warning("No connection is established. Do nothing.")

    def request(self,method,command):
        """ Send a request via HTTP and retrieve response """
        if self._resource is None:
            logger.error("No connection established for %s. Query returns None." %self.name)
            return None
        logger.debug("Send request to %s: %s" %(self.name,command))
        self._resource.request(method,command)
        res = self._resource.getresponse()
        if res.status == 200:
            logger.debug("Successfully made request to %s. Status: %s - %s" %(self.name,res.status,res.reason))
        else:
            logger.warning("Issue making request to %s. Status: %s - %s" %(self.name,res.status,res.reason))
        data = res.read()
        logger.debug("Response from %s: %s" %(self.name,data))
        return data

    def query(self,command):
        return self.request("GET",'/'+str(command))

    def write(self,command):
        """ Post data or instruction via HTTP """
        return self.request("POST",'/'+str(command))

    def close(self):
        self.disconnect()

class MUX_2SP4T(HTTPInstrument):
    def __init__(self,resource_name=None,name="Mini-Circuits RC-2SP4T-A18 MUX",instrument_type="MUX"):
        """ Control for Mini-Circuits RC-2SP4T Switching Matrix

        resource_name: Network IP address and port, e.g. 100.100.10.10:80
        """
        super(MUX_2SP4T,self).__init__(resource_name,name,instrument_type)

    def set_one(self,params):
        """ Set the connection of 1 channel at a time without affecting other channel(s).
        params: dictionary defining which channel to be set to which terminal

        Examples: set_one({'A':1}) sets channel A to 1, leaves channel B and the others unchanged.
                  set_one({'A':2,'B':4}) sets channel A to 2, channel B to 4, leaves the others unchanged.
        """
        logger.debug("Set some connection(s) for %s" %self.name)
        res = []
        for k,v in params.items():
            r = self.write("SP4T"+k+":STATE:"+str(v))
            res.append(r)
        return res

    def set_all(self,params,digit=4):
        """ Set the connections of all channels
        params: number, string or dictionary defining which channel to be set to which terminal
        digit: number of digits per channel. Default 4 (channels, bits)

        Examples: set_all({'A':2}) or set_all(2) or set_all('10') sets channel A to 2, disconnects channel B
        """
        logger.debug("Set all connections for %s" %self.name)
        if isinstance(params,int):
            val = params
        elif isinstance(params,str):
            val = int(params,2)
        elif isinstance(params,dict):
            val = 0
            for k,v in params.items():
                shift = (ord(k.upper()) - 65)*digit
                shift += v-1
                val += 1 << shift
        else:
            logger.error("Function set_all is not yet implemented for this type of argument: %s" %type(params))
            return None
        res = self.write("SETP="+str(val))
        return res

    @property
    def status(self,digit=4):
        """ Get the status of the switching matrix
        digit: number of digits per channel. Default 4 (channels, bits)
        """
        logger.debug("Retrieve connection status of %s" %self.name)
        res = self.query("SWPORT?")
        res = int(res)
        # Need to convert the return value to dictionary of channels
        chunk = 1 << digit
        ch = 65
        result = {}
        while res > 0:
            mod = res % chunk
            if mod > 0:
                result[chr(ch)] = int(log2(mod)) + 1
            ch += 1
            res = res >> digit
        return result
