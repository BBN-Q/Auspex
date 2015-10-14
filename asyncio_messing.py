"""
Messing around with asyncio for streaming data flow.

Requires Python 3.5
"""

import abc
import asyncio
import datetime
import numpy as np
from scipy.signal import lfilter
import pandas as pd

import logging
logging.basicConfig(format="%(levelname)s: %(asctime)s %(message)s", level=logging.INFO)

def shutdown(loop):
    pending = asyncio.Task.all_tasks()
    for t in pending:
        t.cancel()
    loop.stop()

class DataProducer(object):
    def __init__(self, name):
        self.name = name
        self.out_queue = asyncio.Queue()

class ADC(DataProducer):

    async def take_data(self, loop, timeout):
        end_time = loop.time() + timeout
        while True:
            data = np.sin(2*np.pi*2*0.1*np.arange(100)) + 0.1*np.random.randn(100)
            timeStamp = str(datetime.datetime.now())
            logging.info("Acquired data and putting in queue")
            await self.out_queue.put(data)
            if (loop.time() + 1.0) >= end_time:
                shutdown(loop)
                break
            await asyncio.sleep(1)

class DataCruncher(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self.name = name
        self.out_queue = asyncio.Queue()
        self.in_queue = asyncio.Queue()

    async def wait_for_data(self):
        while True:
            data = await self.in_queue.get()
            logging.info("DataCruncher got data")
            await self.process_data(data)

    @abc.abstractmethod
    async def process_data(self, data):
        "Process the latest data"
        return

class FIR(DataCruncher):
    def __init__(self, name, b, a):
        super().__init__(name)
        self.b = b
        self.a = a

    async def process_data(self, data):
        await self.out_queue.put(lfilter(self.b, self.a, data))
        logging.info("DataCruncher processed data")

class DataWriter(object):
    def __init__(self):
        self.in_queues = {}

    def add_stream(self, name):
        self.in_queues[name] = asyncio.Queue()

    async def run(self):
        store = pd.HDFStore("silly.h5")
        while True:
            results, _ = await asyncio.wait([q.get() for q in self.in_queues.values()])
            logging.info("DataWriter got data")
            data = [r.result() for r in results]
            df = pd.DataFrame({n:d for n,d in zip(self.in_queues.keys(), data)})
            store.append("data", df, index=False)

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
    async def broadcast(inQ, outQs):
        while True:
            logging.info("DataInterconnect waiting for data")
            data = await inQ.get()
            for q in outQs:
                logging.info("DataInterconnect sending data")
                await q.put(data)

    def run(self, loop):
        for inQ_name, inQ in self.input_queues.items():
            loop.create_task(DataInterconnect().broadcast(inQ,
                [self.output_queues[outQ] for outQ in self.crossbar[inQ_name]]) )

if __name__ == '__main__':
    fakeADC = ADC("ADC")
    fakeDSP = FIR("FIR1", 1/5*np.ones(5), 1)
    myWriter = DataWriter()

    fakeInterconnect = DataInterconnect()

    fakeInterconnect.register_input("fakeADC")
    fakeInterconnect.register_output("fakeDSP", fakeDSP.in_queue, "fakeADC")
    myWriter.add_stream("fakeADC")
    fakeInterconnect.register_output("myWriter", myWriter.in_queues["fakeADC"], "fakeADC")

    fakeADC.out_queue = fakeInterconnect.input_queues["fakeADC"]

    loop = asyncio.get_event_loop()
    fakeInterconnect.run(loop)
    tasks = [
        asyncio.ensure_future(fakeDSP.wait_for_data()),
        asyncio.ensure_future(fakeADC.take_data(loop, 5)),
        asyncio.ensure_future(myWriter.run())]
    try:
        loop.run_until_complete(asyncio.wait(tasks))
    except asyncio.CancelledError:
        logging.info("Caught CancelledError")
    finally:
        logging.info("Cancelled!")
        loop.close()

    logging.info("Made it out of asyncio land!")
