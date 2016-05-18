import asyncio
import numpy as np
from .stream import ProcessingNode, DataStreamDescriptor

class Averager(ProcessingNode):
    """Takes data and collapses along the specified axis."""
    def __init__(self, axis=0, **kwargs):
        super(Averager, self).__init__(**kwargs)
        self.axis = axis # Can be string on numerical index

    def update_descriptors(self):
        descriptor_in = self.input_streams[0].descriptor

        # Convert named axes to an index
        if isinstance(self.axis, str):
            names = [a.label for a in descriptor_in.axes]
            if self.axis not in names:
                raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis, self.descriptor_in))
            self.axis = names.index(self.axis)
            # print("** Converted string to axis {} **:".format(self.axis))

        new_axes = descriptor_in.axes[:]
        new_axes.pop(self.axis)
        # print("** New Axes {} **".format(new_axes))
        descriptor_out = DataStreamDescriptor()
        descriptor_out.axes = new_axes
        # print("** with averaged axis subtracted **: {}".format(descriptor_out))

        for os in self.output_streams:
            os.descriptor = descriptor_out
            # print("** in loop **: {}".format(os.descriptor))

    async def run(self):
        idx = 0
        self.data = np.empty(self.input_streams[0].num_points())
        while True:
            if self.input_streams[0].done():
                # We've stopped receiving new input, make sure we've flushed the output streams
                if len(self.output_streams) > 0:
                    if False not in [os.done() for os in self.output_streams]:
                        print("Cruncher finished crunching (clearing outputs).")
                        break
                else:
                    print("Cruncher finished crunching.")
                    break

            new_data = await self.input_streams[0].queue.get()
            print("{} got data".format(self.label))

            new_data = 2*new_data

            self.data[idx:idx+len(new_data)] = new_data
            for output_stream in self.output_streams:
                await output_stream.push(new_data)
            idx += len(new_data)