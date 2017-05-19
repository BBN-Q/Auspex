import asyncio
import zmq.asyncio
from random import randrange
import time
import numpy as np
import json
from threading import Thread

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class AsyncMatplotServer(Thread):

    def __init__(self, plot_desc={}, status_port = 7771, data_port = 7772):
        super(AsyncMatplotServer, self).__init__()
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

    async def _send(self, data):
        await self.data_sock.send(("server %d"%data).encode())

    def send(self, data):
        self._loop.create_task(self._send(data))

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

            print("starting...")

            self._loop.create_task(self.poll_sockets())
            try:
                self._loop.run_forever()
            finally:
                self.stop()


if __name__ == "__main__":
    plot_desc_1 = {
        'Population': {
            'plot_mode': 'real',
            'plot_dims': 1,
            'xlabel': 'Rabbits',
            'ylabel': 'Foxes',
            },
        'Junk': {
            'plot_mode': 'imag',
            'plot_dims': 1,
            'xlabel': 'Length of Curve',
            'ylabel': 'Height of Curve',
            },
        'Image': {
            'plot_mode': 'real',
            'plot_dims': 2,
            'xlabel': 'Bottom Axis',
            'ylabel': 'Side Axis',
            },
    }

    s = AsyncMatplotServer(plot_desc_1)
    while True:
        time.sleep(1)
    # time.sleep(1)
    # for j in range(3):
    #     s.send(randrange(1,10))
    #     time.sleep(1)
