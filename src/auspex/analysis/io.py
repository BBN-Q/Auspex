# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
from auspex.stream import DataStream, DataAxis, SweepAxis, DataStreamDescriptor, InputConnector, OutputConnector
import h5py
import numpy as np

def load_from_HDF5(filename, groupname='main'):
    with h5py.File(filename, 'r') as f:
        # Reconstruct the descriptor
        descriptor = DataStreamDescriptor()
        g = f[groupname]
        axis_refs = g['descriptor']
        for ref in reversed(axis_refs):
            ax = g[ref]
            if len(ax.dtype) > 1:
                # Unstructured

                names = [k for k in ax.dtype.fields.keys()]
                units = [ax.attrs['unit_'+name] for name in names]
                points = ax[:]
                points = points.view(np.float32).reshape(points.shape + (-1,))
                descriptor.add_axis(DataAxis(names, points=points, unit=units))
            else:
                # Structured
                name = ax.attrs['NAME'].decode('UTF-8')
                unit = ax.attrs['unit']
                points = ax[:]
                descriptor.add_axis(DataAxis(name, points=points, unit=unit))

        for attr_name in axis_refs.attrs.keys():
            descriptor.metadata[attr_name] = axis_refs.attrs[attr_name]

        data = g['data'][:]
        return data, descriptor

if __name__ == '__main__':
    filename = "test_writehdf5_adaptive_unstructured-0000.h5"
    data, desc = load_from_HDF5(filename)
    for a in desc.axes:
        print(a)
    print(data)
    print([k for k in data.dtype.fields.keys()])
