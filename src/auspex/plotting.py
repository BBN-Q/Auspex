# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0


import os
import sys
import subprocess
import psutil
import tempfile

if sys.platform == 'win32' or 'NOFORKING' in os.environ:
    from threading import Thread as Process
    from threading import Event
else:
    from multiprocessing import Process
    from multiprocessing import Event

import json
import zmq
import queue
import time
import numpy as np
from auspex.log import logger

class BokehServerProcess(object):
    def __init__(self, notebook=False):
        super(BokehServerProcess, self).__init__()
        self.run_in_notebook = notebook
        self.pid_filename = os.path.join(tempfile.gettempdir(), "auspex_bokeh.pid")

    def run(self):
        # start a Bokeh server if one is not already running
        pid = self.read_session_pid()
        if pid:
            self.p = psutil.Process(pid)
            logger.info("Using existing Bokeh server")
            return
        logger.info("Starting Bokeh server")
        args = ["bokeh", "serve", "--port", "5006"]
        if self.run_in_notebook:
            args.append("--allow-websocket-origin=localhost:8888")
        self.p = subprocess.Popen(args, env=os.environ.copy())
        self.write_session_pid()
        # sleep to give the Bokeh server a chance to start
        # TODO replace this with some bokeh client API call that
        # verifies that the server is running
        time.sleep(3)

    def terminate(self):
        if self.p:
            print("Killing bokeh server process {}".format(self.p.pid))
            try:
                for child_proc in psutil.Process(self.p.pid).children():
                    print("Killing child process {}".format(child_proc.pid))
                    child_proc.terminate()
            except:
                print("Couldn't kill child processes.")
            self.p.terminate()
            self.p = None
            os.remove(self.pid_filename)

    def write_session_pid(self):
        with open(self.pid_filename, "w") as f:
            f.write("{}\n".format(self.p.pid))

    def read_session_pid(self):
        # check if pid file exists
        if not os.path.isfile(self.pid_filename):
            return None
        with open(self.pid_filename) as f:
            pid = int(f.readline())
        if psutil.pid_exists(pid):
            return pid
        else:
            return None

class PlotDescServerProcess(Process):

    def __init__(self, plot_desc={}, port = 7771):
        super(PlotDescServerProcess, self).__init__()
        self.plot_desc = plot_desc
        self.port = port

        # Event for killing the server properly
        self.exit = Event()

    def run(self):
        try:
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
        except Exception as e:
            logger.warning("PlotDescServerProcess failed with exception {}".format(e))

    def shutdown(self):
        self.exit.set()
        self.join()

class PlotDataServerProcess(Process):

    def __init__(self, data_queue, port = 7772):
        super(PlotDataServerProcess, self).__init__()
        self.data_queue = data_queue
        self.port = port
        self.daemon = True

        # Event for killing the filter properly
        self.exit = Event()

    def run(self):
        try:
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
        except Exception as e:
            logger.warning("PlotDataServerProcess failed with exception {}".format(e))

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

