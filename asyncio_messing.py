"""
Messing around with asyncio for streaming data flow.

Requires Python 3.5
"""

import asyncio
import datetime

class DataProducer(object):

    queues = []

    async def take_data(self, loop, timeout):
        end_time = loop.time() + timeout
        while True:
            timeStamp = str(datetime.datetime.now())
            print("Have data at " + timeStamp)
            for q in self.queues:
                print("Putting data in queue")
                await q.put("Queue has data at " + timeStamp )
            if (loop.time() + 1.0) >= end_time:
                break
            await asyncio.sleep(1)

class DataCruncher(object):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def wait_for_data(self):
        while True:
            print("Waiting for data...")
            data = await self.queue.get()
            print("Got data from queue...")
            self.process_data(data)

    def process_data(self, data):
        print("Processed data from queue: " + data)


if __name__ == '__main__':
    fakeDigitizer = DataProducer()
    fakeDSP = DataCruncher()

    fakeDigitizer.queues.append(fakeDSP.queue)

    loop = asyncio.get_event_loop()
    tasks = [
        asyncio.ensure_future(fakeDSP.wait_for_data()),
        asyncio.ensure_future(fakeDigitizer.take_data(loop, 5))]
    loop.run_until_complete(asyncio.wait(tasks))
