import asyncio
import zmq.asyncio
from random import randrange
import time
import numpy as np
from threading import Thread

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

context2 = zmq.Context()
socket_sessions = context2.socket(zmq.PUB)
socket_sessions.bind("tcp://*:5557")

class AsyncMatplotServer(Thread):

    def __init__(self, status_port = 7771, data_port = 7772):
        super(AsyncMatplotServer, self).__init__()
        self.status_port = status_port
        self.data_port = data_port
        self.daemon = True
        self.start()

    async def poll_sockets(self):
        while True:
            evts = dict(await self.poller.poll(1000))
            if self.status_sock in evts and evts[self.status_sock] == zmq.POLLIN:
                ident, msg = await self.status_sock.recv_multipart()
                print("Got {} from {}".format(msg.decode(), ident.decode()))
                if msg == b"WHATSUP":
                    await self.status_sock.send_multipart([ident, b"HI!"])
            await asyncio.sleep(0)

    async def _send(self, data):
        await self.data_sock.send(("server %d"%data).encode())

    def send(self, data):
        self._loop.create_task(self._send(data))

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
                self._loop.stop()
                pending = asyncio.Task.all_tasks()
                self._loop.run_until_complete(asyncio.gather(*pending))
                self._loop.close()

if __name__ == "__main__":
    s = AsyncMatplotServer()
    time.sleep(1)
    for j in range(20):
        s.send(randrange(1,10))
        time.sleep(1)
