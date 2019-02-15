# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import auspex.analysis.switching as sw
# from auspex.analysis.io import load_from_HDF5
try:
    from adapt import refine
except:
    print("Could not import the adapt package.")
import numpy as np
import time

def delaunay_refine_from_file(writer, x_name, y_name, z_name, max_points=500, criterion="integral", threshold = "one_sigma", plotter=None):
    raise Exception("Refinement needs to be updated for new data format")
    def refine_func(sweep_axis, experiment):
        data, desc = load_from_HDF5(writer.filename.value, reshape=False)
        groupname = writer.groupname.value
        zs = data[groupname][z_name]
        ys = data[groupname][y_name]
        xs = data[groupname][x_name]

        points     = np.array([xs, ys]).transpose()
        new_points = refine.refine_scalar_field(points, zs, all_points=False,
                                    criterion=criterion, threshold=threshold)
        if len(points) + len(new_points) > max_points:
            print("Reached maximum points ({}).".format(max_points))
            return False
        print("Reached {} points.".format(len(points) + len(new_points)))
        sweep_axis.add_points(new_points)

        if plotter:
            data = np.array([xs, ys, zs]).transpose()            
            experiment.push_to_plot(plotter, data)

        # time.sleep(0.02)
        return True
    return refine_func
