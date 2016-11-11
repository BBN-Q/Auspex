# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import auspex.analysis.switching as sw
from adapt import refine
import numpy as np
import time

def delaunay_refine_from_file(writer, x_name, y_name, z_name, max_points=500, criterion="integral", threshold = "one_sigma", plotter=None):
    async def refine_func(sweep_axis):
        zs = writer.data.value[z_name]
        ys = writer.data.value[y_name]
        xs = writer.data.value[x_name]

        points     = np.array([xs, ys]).transpose()
        new_points = refine.refine_scalar_field(points, zs, all_points=False,
                                    criterion=criterion, threshold=threshold)
        if len(points) + len(new_points) > max_points:
            print("Reached maximum points ({}).".format(max_points))
            return False
        print("Reached {} points.".format(len(points) + len(new_points)))
        sweep_axis.add_points(new_points)

        if plotter:
            exp = plotter.experiment

            mesh, scale_factors = sw.scaled_Delaunay(points)
            xs     = xs[mesh.simplices]
            ys     = ys[mesh.simplices]
            avg_zs = [np.mean(row) for row in zs[mesh.simplices]]
            await exp.push_to_plot(plotter, [xs,ys,avg_zs])

        time.sleep(0.1)
        return True
    return refine_func
