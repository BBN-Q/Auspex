# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

# from threading import Thread
import multiprocessing as mp
import subprocess
import psutil
import queue
import os
import json
import sys
import tempfile
import time
import asyncio
import zmq
import zmq.asyncio
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
            socks = dict(self.poller.poll())
            if socks.get(self.sock) == zmq.POLLIN:
                ident, msg = self.sock.recv_multipart()
                if msg == b"WHATSUP":
                    self.sock.send_multipart([ident, b"HI!", json.dumps(self.plot_desc).encode('utf8')])

    def shutdown(self):
        self.exit.set()
        self.sock.close()
        self.context.destroy()

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

    def send(self, message):
        print(message.keys())
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
        print("_send from plot server", name, id(self))
        self.sock.send_multipart(msg_contents)

    def shutdown(self):
        self.exit.set()
        self.sock.close()
        self.context.destroy()

        # # self._loop.create_task(self.listen_for_connections())
        # # self._loop.create_task(self.send_data_messages())
        # try:
        #     self._loop.run_forever()
        # finally:
        #     self.status_sock.close()
        #     self.data_sock.close()
        #     self.context.destroy()

    # async def listen_for_connections(self):
    #     while not self.exit.is_set():
    #         evts = dict(await self.poller.poll(50))
    #         if self.status_sock in evts and evts[self.status_sock] == zmq.POLLIN:
    #             ident, msg = await self.status_sock.recv_multipart()
    #             if msg == b"WHATSUP":
    #                 await self.status_sock.send_multipart([ident, b"HI!", json.dumps(self.plot_desc).encode('utf8')])
    #         await asyncio.sleep(0.020)

    # async def send_data_messages(self):
    #     while not self.exit.is_set():
    #         try:
    #             msg_dict = self.data_queue.get(False)
    #             print("data in queue")
    #             await self._send(msg_dict["name"], msg_dict["data"], msg_dict["msg"])
    #         except:
    #             print("no data in queue")
    #             pass

    #         await asyncio.sleep(0.020)

    # async def _send(self, name, data, msg="data"):
    #     msg_contents = [msg.encode(), name.encode()]
    #     # We might be sending multiple axes, series, etc.
    #     # Just add them succesively to a multipart message.
    #     for dat in data:
    #         md = dict(
    #             dtype = str(dat.dtype),
    #             shape = dat.shape,
    #         )
    #         msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
    #     print("_send from plot server", name, id(self))
    #     await self.data_sock.send_multipart(msg_contents)

    # async def put(self, stuff):
    #     self.data_queue.put_no_wait(stuff)

    # def send(self, name, *data, msg="data"):
    #     print("send from plot server", name, id(self))
    #     if not self.stopped:
    #         print("creating _send task")
    #         self._loop.create_task(self.put({"name": name, "data": data, "msg":msg}))

    # def shutdown(self):
    #     self.send("irrelevant", np.array([]), msg="done")
    #     self.exit.set()
    #     pending = asyncio.Task.all_tasks(loop=self._loop)
    #     self._loop.stop()
    #     time.sleep(1)
    #     for task in pending:
    #         task.cancel()
    #         try:
    #             self._loop.run_until_complete(task)
    #         except asyncio.CancelledError:
    #             pass
    #     self._loop.close()

    # def run(self):
    #     self._loop = zmq.asyncio.ZMQEventLoop()
    #     asyncio.set_event_loop(self._loop)
    #     self.context = zmq.asyncio.Context()
    #     self.status_sock = self.context.socket(zmq.ROUTER)
    #     self.data_sock = self.context.socket(zmq.PUB)
    #     self.status_sock.bind("tcp://*:%s" % self.status_port)
    #     self.data_sock.bind("tcp://*:%s" % self.data_port)
    #     self.poller = zmq.asyncio.Poller()
    #     self.poller.register(self.status_sock, zmq.POLLIN)

    #     self._loop.create_task(self.listen_for_connections())
    #     self._loop.create_task(self.send_data_messages())
    #     try:
    #         self._loop.run_forever()
    #     finally:
    #         self.status_sock.close()
    #         self.data_sock.close()
    #         self.context.destroy()
