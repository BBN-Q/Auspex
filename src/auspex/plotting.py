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
import numpy as np
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
            evts = dict(await self.poller.poll(50))
            if self.status_sock in evts and evts[self.status_sock] == zmq.POLLIN:
                ident, msg = await self.status_sock.recv_multipart()
                if msg == b"WHATSUP":
                    await self.status_sock.send_multipart([ident, b"HI!", json.dumps(self.plot_desc).encode('utf8')])
            await asyncio.sleep(0.010)

    async def _send(self, name, data, msg="data"):
        msg_contents = [msg.encode(), name.encode()]
        # We might be sending multiple axes, series, etc.
        # Just add them succesively to a multipart message.
        for dat in data:
            md = dict(
                dtype = str(dat.dtype),
                shape = dat.shape,
            )
            msg_contents.extend([json.dumps(md).encode(), np.ascontiguousarray(dat)])
        await self.data_sock.send_multipart(msg_contents)

    def send(self, name, *data, msg="data"):
        self._loop.create_task(self._send(name, data, msg=msg))

    def stop(self):
        self.send("irrelevant", np.array([]), msg="done")
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
                self.status_sock.close()
                self.data_sock.close()
                self.context.destroy()
