# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import multiprocessing as mp
import json
import zmq
import queue
import time
import numpy as np
from auspex.log import logger

class PlotDescServerProcess(mp.Process):

    def __init__(self, plot_desc={}, port = 7771):
        super(PlotDescServerProcess, self).__init__()
        self.plot_desc = plot_desc
        self.port = port

        # Event for killing the server properly
        self.exit = mp.Event()

    def run(self):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.ROUTER)
        self.sock.bind("tcp://*:%s" % self.port)
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

        # Loop and accept messages
        while not self.exit.is_set():
            socks = dict(self.poller.poll(100))
            if socks.get(self.sock) == zmq.POLLIN:
                ident, msg = self.sock.recv_multipart()
                if msg == b"WHATSUP":
                    self.sock.send_multipart([ident, b"HI!", json.dumps(self.plot_desc).encode('utf8')])
        self.sock.close()
        self.context.destroy()

    def shutdown(self):
        self.exit.set()
        self.join()

class PlotDataServerProcess(mp.Process):

    def __init__(self, data_queue, port = 7772):
        super(PlotDataServerProcess, self).__init__()
        self.data_queue = data_queue
        self.port = port
        self.daemon = True

        # Event for killing the filter properly
        self.exit = mp.Event()

    def run(self):
        self.context = zmq.Context()
        self.sock = self.context.socket(zmq.PUB)
        self.sock.bind("tcp://*:%s" % self.port)
        
        # Loop and accept messages
        while not self.exit.is_set():
            try:
                message = self.data_queue.get(True, 0.02)
                self.send(message)
            except queue.Empty as e:
                continue
        self.sock.close()
        self.context.destroy()

    def send(self, message):
        data = message['data']
        msg  = message['msg']
        name = message['name']

        msg_contents = [msg.encode(), name.encode()]

        # We might be sending multiple axes, series, etc.
        # Just add them succesively to a multipart message.
        for dat in data:
            md = dict(
                dtype = str(dat.dtype),
                shape = dat.shape,
            )
            msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
        self.sock.send_multipart(msg_contents)

    def shutdown(self):
        self.exit.set()
        self.join()

