"""
Messing around with asyncio for streaming data flow.

Requires Python 3.5
"""

import asyncio
import datetime
import logging
logging.basicConfig(format="%(levelname)s: %(asctime)s %(message)s", level=logging.INFO)

class DataProducer(object):

    async def take_data(self, loop, timeout):
        end_time = loop.time() + timeout
        while True:
            timeStamp = str(datetime.datetime.now())
            logging.info("Acquired data and putting in queue")
            await self.queue.put("Queue has data at " + timeStamp )
            if (loop.time() + 1.0) >= end_time:
                loop.stop()
                break
            await asyncio.sleep(1)

class DataCruncher(object):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def wait_for_data(self):
        while True:
            logging.info("DataCruncher waiting for data")
            data = await self.queue.get()
            logging.info("DataCruncher got data")
            self.process_data(data)

    def process_data(self, data):
        logging.info("DataCruncher processed data from queue: " + data)

class DataInterconnect(object):
    def __init__(self):
        self.input_queues = {}
        self.output_queues = {}
        self.crossbar = {}

    def register_input(self, name):
        self.input_queues[name] = asyncio.Queue()

    def register_output(self, name, q, src):
        self.output_queues[name] = q
        self.crossbar.setdefault(src, []).append(name)

    @staticmethod
    async def connect_input_output(inQ, outQs):
        while True:
            logging.info("DataInterconnect waiting for data")
            data = await inQ.get()
            for q in outQs:
                logging.info("DataInterconnect sending data")
                await q.put(data)

    def run(self, loop):
        for inQ_name, inQ in self.input_queues.items():
            loop.create_task(DataInterconnect().connect_input_output(inQ,
                [self.output_queues[outQ] for outQ in self.crossbar[inQ_name]]) )

if __name__ == '__main__':
    fakeDigitizer = DataProducer()
    fakeDSP = DataCruncher()

    fakeInterconnect = DataInterconnect()

    fakeInterconnect.register_input("fakeDigitizer")
    fakeInterconnect.register_output("fakeDSP", fakeDSP.queue, "fakeDigitizer")

    fakeDigitizer.queue = fakeInterconnect.input_queues["fakeDigitizer"]

    loop = asyncio.get_event_loop()
    fakeInterconnect.run(loop)
    tasks = [
        asyncio.ensure_future(fakeDSP.wait_for_data()),
        asyncio.ensure_future(fakeDigitizer.take_data(loop, 5))]
    try:
        loop.run_until_complete(asyncio.wait(tasks))
    finally:
        logging.info("Cancelled!")
        for t in tasks:
            t.cancel()
        loop.close()
