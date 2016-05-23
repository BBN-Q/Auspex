import asyncio
import numpy as np
from pycontrol.stream import DataStreamDescriptor
from .filter import Filter, InputConnector, OutputConnector

class Average(Filter):
    """Takes data and collapses along the specified axis."""

    data = InputConnector()
    partial_average = OutputConnector()
    final_average = OutputConnector()

    def __init__(self, axis=0, **kwargs):
        super(Average, self).__init__(**kwargs)
        self.axis = axis # Can be string on numerical index
        self.points_before_averaging = None

    def update_descriptors(self):
        descriptor_in = self.input_streams[0].descriptor

        # Convert named axes to an index
        if isinstance(self.axis, str):
            names = [a.label for a in descriptor_in.axes]
            if self.axis not in names:
                raise ValueError("Could not find axis {} within the DataStreamDescriptor {}".format(self.axis, self.descriptor_in))
            self.axis = names.index(self.axis)

        new_axes = descriptor_in.axes[:]
        new_axes.pop(self.axis)
        descriptor_out = DataStreamDescriptor()
        descriptor_out.axes = new_axes

        self.points_before_averaging = descriptor_in.num_points_through_axis(self.axis)
        print("Average thinks it needs {} points before averaging a chunk.".format(self.points_before_averaging))
        print("Average has axes in {}.".format(descriptor_in.axes))
        print("Average has axes out {}.".format(descriptor_out.axes))
        self.data_dims = descriptor_in.data_dims(fortran=True)
        self.chunk_dims = list(reversed(self.data_dims[0:self.axis]))

        for os in self.output_streams:
            os.descriptor = descriptor_out
            # print("** in loop **: {}".format(os.descriptor))

    async def run(self):
        if self.points_before_averaging is None:
            raise Exception("Average has not been initialized. Run 'update_descriptors'")

        idx = 0

        # We only need to accumulate up to the averaging axis
        # BUT we may get something longer at any given time!
        self.data = np.empty(self.input_streams[0].descriptor.num_points())
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
            print("{} got data {}".format(self.label, new_data))

            if len(new_data.shape) > 1:  # We are getting unflattened data
                new_data = new_data.flatten()
            self.data[idx:idx+len(new_data)] = new_data
            idx += len(new_data)

            # Have we amassed enough data to average?
            while idx >= self.points_before_averaging:

                # import ipdb; ipdb.set_trace()
                chunk_to_avg = self.data[0:self.points_before_averaging].reshape(self.chunk_dims)
                print("About to average {}".format(chunk_to_avg))
                avg = np.mean(chunk_to_avg, axis=0)
                print("Average {}".format(avg))
                # Shift remaining data back
                self.data[0:idx] = self.data[self.points_before_averaging:idx+self.points_before_averaging]
                idx -= self.points_before_averaging

                for output_stream in self.output_streams:
                    await output_stream.push(avg)
            
