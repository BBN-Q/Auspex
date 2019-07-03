import pkgutil
import importlib
from .instrument import Command
import json
import socket
import types, functools, inspect
import random
import zmq
import time

instrument_map = {}
for loader, name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module('auspex.instruments.' + name)
    if hasattr(module, "__all__"):
        globals().update((name, getattr(module, name)) for name in module.__all__)
        for name in module.__all__:
            instrument_map.update({name:getattr(module,name)})

client_classes = {}
server_classes = instrument_map

# def lookup_instr(address, server=False):
#     classes = server_classes if server else client_classes
#     available_addresses = [e['address'] for e in available_instruments_local]
#     inst = [inst for inst in available_instruments_local if inst['address'] == address]
#     if len(inst) == 0:
#         return None
#     inst = inst[0]
#     instr_class = [v for k,v in classes.items() if inst['model'] in k][0]
#     return instr_class

def get_servers_instruments(address):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{address}:7777")
    token = f"{random.getrandbits(64):016x}"
    socket.send_pyobj({'msg_type': "server_control", 'cmd_name': "list_instruments_local", 'token': token})
    message = socket.recv_pyobj()
    if message['token'] != token:
        raise Exception("Received wrong token in reply!")
    if not message['success']:
        raise Exception("Command Failed")
    return message['return_value']

def find_remote_instruments(address=None, broadcast='255.255.255.255', port=7777):
    local_addrs = socket.gethostbyname_ex(socket.gethostname())[-1]
    if address is None:
        for addr in local_addrs:
            if not addr.startswith('127'):
                address = addr
    # Create UDP socket
    handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Ask operating system to let us do broadcasts from socket
    handle.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Register poller
    poller = zmq.Poller()
    poller.register(handle, zmq.POLLIN)

    ping_at = time.time()
    servers = []

    handle.sendto('!'.encode(), 0, (broadcast, port))
    while time.time() - ping_at < 1.0:
        try:
            events = dict(poller.poll(1000))
        except KeyboardInterrupt:
            print("interrupted")
            break
        if handle.fileno() in events:
            buf, addrinfo = handle.recvfrom(8192)
            servers.append(addrinfo[0])

    instruments = {}
    for server in servers:
        server_instrs = get_servers_instruments(server)
        for i in server_instrs:
            i['server'] = server
        instruments[server] = server_instrs

    return servers, instruments

class AuspexInstrumentHub():
    def __init__(self):
        self.find_instruments()
    
    def find_instruments(self):
        self.servers, instruments_by_server = find_remote_instruments()
        self.instruments = [instr for instrs in instruments_by_server.values() for instr in instrs]

    def instr_from_serial(self, serial, connect=True):
        instrs = [i for i in self.instruments if i['serial'] == serial]
        if len(instrs) == 0:
            raise Exception("Could not find instrument with that serial number")
        if len(instrs) > 1:
            instrs = [i for i in instrs if 'SOCKET' in i['address']]
        instr = instrs[0]

        instance = [v for k,v in client_classes.items() if instr['model'] in k][0]()
        # instance = lookup_instr(instr['address'])()
        if connect:
            instance.connect(instr['server'], instr['address'])
        return instance

    def update_instruments(self):
        for server in self.servers:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{server}:{7777}")
            token = f"{random.getrandbits(64):016x}"
            msg = {'msg_type': "server_control",
                   'cmd_name': "update_instruments",
                   'token': token}
            socket.send_pyobj(msg)
            rep = socket.recv_pyobj()
            if rep['token'] != token:
                raise Exception("Received wrong token in reply!")
            if not rep['success']:
                raise Exception("Command Failed")

class SimpleInstrument():
    def connect(self, server_address, instrument_address, port=7777):
        """Connect to the instrument server that is responsible for this instrument."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{server_address}:{port}")
        self.instrument_address = instrument_address

    def disconnect(self):
        self.socket.close()
        self.context.term()

    def configure_from_database_object(self, proxy):
        """Accept a database instrument object and configure this instrument using the columns
        found in that instance."""
        self.configure_with_dict(dict((col, getattr(proxy, col)) for col in proxy.__table__.columns.keys()))

    def configure_with_dict(self, settings_dict):
        """Accept a settings dictionary and attempt to set all of the instrument
        parameters using the key/value pairs."""
        for name, value in settings_dict.items():
            if name not in ["id", "label", "model", "address", "channel_db_id", "standalone"]:
                if "_id" in name:
                    continue
                # Python is insane, and attempts to run a property's getter
                # when queried by hasattr. Avoid this behavior with the
                # "ask for forgiveness" paradigm.
                try:
                    setattr(self, name, value)
                except (AttributeError, TypeError) as e:
                    logger.info("Instrument {} property: {} could not be set to {}.".format(self.name,name,value))
                    pass

    def send(self, msg):
        msg['token'] = f"{random.getrandbits(64):016x}"
        msg['instrument_address'] = self.instrument_address
        self._snd(msg)
        return self._rcv(msg['token'])

    def _snd(self, msg):
        self.socket.send_pyobj(msg)

    def _rcv(self, token):
        message = self.socket.recv_pyobj()
        if message['token'] != token:
            raise Exception("Received wrong token in reply!")
        if not message['success']:
            raise Exception("Command Failed")
        return message['return_value']

for name, inst in instrument_map.items():
    dct = inst.__dict__
    clsdict = {}
    properties = []
    for k,v in dct.items():
        cmd = { 'msg_type': 'instr_control',
                'inst_type': inst.__name__, 
                'cmd_name': k,
                'mode': [],
                'args': (),
                'kwargs': {},
                'expose': True}
        if isinstance(v, Command):
            cmd['args'] = v.args
            if 'get_string' in v.kwargs:
                cmd['mode'].append('get')
            if 'set_string' in v.kwargs:
                cmd['mode'].append('set')
            if 'scpi_string' in v.kwargs:
                cmd['mode'] = ['get', 'set']
            cmd['src'] = 'Command'
            cmd['args'] = ('value',)
        elif isinstance(v, property):
            if v.fget:
                cmd['mode'].append('get')
            if v.fset:
                cmd['mode'].append('set')
            cmd['args'] = ('value',)
            cmd['src'] = 'Property'
        elif hasattr(v, "_is_io_method"):
            if v._has_get_method:
                cmd['mode'].append('get')
            if v._has_set_method:
                cmd['mode'].append('set')
            if not v._expose:
                cmd['expose'] = v._expose
                cmd['mode']= ['custom']
            cmd['src'] = 'Function'

            params = inspect.signature(v).parameters

            cmd['args']   = tuple((ak for ak, av in params.items() if av.default == inspect._empty and ak != "self"))
            cmd['kwargs'] = {ak: av.default for ak, av in params.items() if av.default != inspect._empty }
        else:
            continue

        kwargs = cmd.pop('kwargs')
        args = cmd.pop('args')
        mode = cmd.pop('mode')
        expose = cmd.pop('expose')

        arg_code     = ", ".join([f"{c}" for c in args])
        arg_code_get = ", ".join([f"{c}" for c in args if c != 'value'])
        kwarg_code   = ", ".join([f"{k}='{v}'" if isinstance(v,str) else f"{k}='{v}'" for k,v in kwargs.items()])
        
        # kwarg_vals     = "(" + arg_code + ")"
        if arg_code != "":
            arg_code = ", "+arg_code+", "
        if arg_code_get != "":
            arg_code_get = ", "+arg_code_get+", "

        if 'set' in mode:
            cmd['mode'] = 'set'
            arg_vals     = "{" + ", ".join([f"'{n}': {n}" for n in list(args)+list(kwargs.keys())])  + "}"
            exec(f"""def fset(self{arg_code}{kwarg_code}):
                        cmd = {cmd}
                        cmd['arg_vals'] = {arg_vals}
                        return self.send(cmd)""")
            if k[:4] == 'set_':
                clsdict[k] = fset
            clsdict['set_'+k] = fset

        if 'get' in mode:
            cmd['mode'] = 'get'
            get_args = [a for a in args if a != 'value']
            arg_vals     = "{" + ", ".join([f"'{n}': {n}" for n in list(args)+list(kwargs.keys()) if n != 'value'])  + "}"
            exec(f"""def fget(self{arg_code_get}{kwarg_code}):
                        cmd = {cmd}
                        cmd['arg_vals'] = {arg_vals}
                        return self.send(cmd)""")
            if k[:4] == 'get_':
                clsdict[k] = fget
            clsdict['get_'+k] = fget

        if "custom" in mode:
            cmd['mode'] = 'custom'
            arg_vals     = "{" + ", ".join([f"'{n}': {n}" for n in list(args)+list(kwargs.keys())])  + "}"
            exec(f"""def func(self{arg_code}{kwarg_code}):
                        cmd = {cmd}
                        cmd['arg_vals'] = {arg_vals}
                        return self.send(cmd)""")
            clsdict[k] = func

        if expose:
            prop = property(fget if 'get' in mode else None, fset if 'set' in mode else None, None, f"Property for setting/getting {name} {k}")
            clsdict[k] = prop
            properties.append({'name':k, 'value': prop})
    
    clsdict['_properties'] = properties
    clsdict['__doc__'] = f"Auspex stub class for {name} dervied by parsing instrument class."
    client_classes[name] = type(name, (SimpleInstrument,), clsdict)
