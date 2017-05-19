# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from threading import Thread
import subprocess
import psutil
import os
import json
import sys
import tempfile
import time
import asyncio
import zmq
import zmq.asyncio
from auspex.log import logger

class MatplotServerThread(Thread):

    def __init__(self, plot_desc={}, status_port = 7771, data_port = 7772):
        super(MatplotServerThread, self).__init__()
        self.plot_desc = plot_desc
        self.status_port = status_port
        self.data_port = data_port
        self.daemon = True
        self.stopped = False
        self.start()

    async def poll_sockets(self):
        while not self.stopped:
            evts = dict(await self.poller.poll(1000))
            if self.status_sock in evts and evts[self.status_sock] == zmq.POLLIN:
                ident, msg = await self.status_sock.recv_multipart()
                print("Got {} from {}".format(msg.decode(), ident.decode()))
                if msg == b"WHATSUP":
                    await self.status_sock.send_multipart([ident, b"HI!", json.dumps(self.plot_desc).encode('utf8')])
            await asyncio.sleep(0)

    async def _send(self, name, data, msg):
        md = dict(
            dtype = str(data.dtype),
            shape = data.shape,
        )
        await self.data_sock.send_multipart([msg.encode(), name.encode(), json.dumps(md).encode(), data])

    def send(self, name, data, msg="data"):
        self._loop.create_task(self._send(name, data, msg=msg))

    def stop(self):
        self.stopped = True
        pending = asyncio.Task.all_tasks(loop=self._loop)
        self._loop.stop()
        time.sleep(1)
        for task in pending:
            task.cancel()
            try:
                self._loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
        self._loop.close()

    def run(self):
            self._loop = zmq.asyncio.ZMQEventLoop()
            asyncio.set_event_loop(self._loop)
            self.context = zmq.asyncio.Context()
            self.status_sock = self.context.socket(zmq.ROUTER)
            self.data_sock = self.context.socket(zmq.PUB)
            self.status_sock.bind("tcp://*:%s" % self.status_port)
            self.data_sock.bind("tcp://*:%s" % self.data_port)
            self.poller = zmq.asyncio.Poller()
            self.poller.register(self.status_sock, zmq.POLLIN)

            self._loop.create_task(self.poll_sockets())
            try:
                self._loop.run_forever()
            finally:
                self.stop()

# class BokehServerProcess(object):
#     def __init__(self, notebook=False):
#         super(BokehServerProcess, self).__init__()
#         self.run_in_notebook = notebook
#         self.pid_filename = os.path.join(tempfile.gettempdir(), "auspex_bokeh.pid")

#     def run(self):
#         # start a Bokeh server if one is not already running
#         pid = self.read_session_pid()
#         if pid:
#             self.p = psutil.Process(pid)
#             logger.info("Using existing Bokeh server")
#             return
#         logger.info("Starting Bokeh server")
#         args = ["bokeh", "serve", "--port", "5006", "--allow-websocket-origin=localhost:8888", "--allow-websocket-origin=localhost:8889", "--allow-websocket-origin=localhost:8890"]
#         self.p = subprocess.Popen(args, env=os.environ.copy())
#         self.write_session_pid()
#         # sleep to give the Bokeh server a chance to start
#         # TODO replace this with some bokeh client API call that
#         # verifies that the server is running
#         time.sleep(3)

#     def terminate(self):
#         if self.p:
#             print("Killing bokeh server process {}".format(self.p.pid))
#             try:
#                 for child_proc in psutil.Process(self.p.pid).children():
#                     print("Killing child process {}".format(child_proc.pid))
#                     child_proc.terminate()
#             except:
#                 print("Couldn't kill child processes.")
#             self.p.terminate()
#             self.p = None
#             os.remove(self.pid_filename)

#     def write_session_pid(self):
#         with open(self.pid_filename, "w") as f:
#             f.write("{}\n".format(self.p.pid))

#     def read_session_pid(self):
#         # check if pid file exists
#         if not os.path.isfile(self.pid_filename):
#             return None
#         with open(self.pid_filename) as f:
#             pid = int(f.readline())
#         # check that a process is running on that PID
#         if not psutil.pid_exists(pid):
#             return None
#         # check that the process is a Bokeh server
#         cmd = psutil.Process(pid).cmdline()
#         if any('bokeh' in item for item in cmd):
#             return pid
#         return None
