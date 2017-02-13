# Copyright 2016 Raytheon BBN Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from auspex.log import logger

def load_switching_data(filename, start_state=None, failure=False,
                        threshold=None,
                        voltage_scale_factor=1.0, duration_scale_factor=1.0):
    with h5py.File(filename, 'r') as f:
        # Regular axes
        states = f['state'][:]
        reps   = f['attempt'][:]
        # Unstructured axes
        dat = f['data'][:].reshape((-1, reps.size, states.size))
        Vs  = dat['voltage']
        # Get the point tuples
        durs   = dat['pulse_duration'][:,0,0]
        amps   = dat['pulse_voltage'][:,0,0]
        points = np.array([durs, amps]).transpose()

    if failure:
        return points, reset_failure(Vs, start_state=start_state)
    else:
        return points, switching_phase(Vs, start_state=start_state, threshold=threshold)

def switching_phase(data, **kwargs):
    counts, start_stt = count_matrices(data, **kwargs)
    switched_stt = int(1 - start_stt)

    beta_as = 1 + counts[:,start_stt,switched_stt]
    beta_bs = 1 + counts[:,start_stt,start_stt]

    return np.vectorize(beta.mean)(beta_as, beta_bs)

def clusterer(data, num_clusters=2):
    all_vals = data.flatten()
    all_vals.resize((all_vals.size,1))
    logger.info("Number of clusters: %d" % num_clusters)

    init_guess = np.linspace(np.min(all_vals), np.max(all_vals), num_clusters)
    init_guess[[1,-1]] = init_guess[[-1,1]]
    init_guess.resize((num_clusters,1))

    clust = KMeans(init=init_guess, n_clusters=num_clusters)
    state = clust.fit_predict(all_vals)

    for ct in range(num_clusters):
        logger.info("Cluster {}: {} +/- {}".format(ct, all_vals[state==ct].mean(), all_vals[state==ct].std()))
    return clust

# def average_data(data, avg_points):
#     return np.array([np.mean(d.reshape(avg_points, -1, order="F"), axis=0) for d in data])
    # if display:
    #     plt.figure()
    #     for ct in range(num_clusters):
    #         import seaborn as sns
    #         sns.distplot(all_vals[state == ct], kde=False, norm_hist=False)

def count_matrices(data, start_state=None, threshold=None, display=None):
    num_clusters = 2
    if threshold is None:
        clust = clusterer(data)
        state = clust.fit_predict(data.reshape(-1, 1)).reshape(data.shape)
    else:
        logger.debug("Cluster data based on threshold = {}".format(threshold))
        state = data > threshold

    init_state  = state[:,:,0]
    final_state = state[:,:,1]
    switched    = np.logical_xor(init_state, final_state)

    init_state_frac = [np.mean(init_state == ct) for ct in range(num_clusters)]
    for ct, fraction in enumerate(init_state_frac):
        logger.info("Initial fraction of state %d: %f" %(ct, fraction))

    if start_state is not None and start_state in range(num_clusters):
        start_stt = start_state
    else:
        start_stt = np.argmax(init_state_frac)
    logger.info("Start state set to state: {}".format(start_stt))
    logger.info("Switched state is state: {}".format(1-start_stt))

    # This array contains a 2x2 count_matrix for each coordinate tuple
    count_mat = np.zeros((init_state.shape[0], 2, 2))

    # count_mat      = np.zeros((2,2), dtype=np.int)
    count_mat[:,0,0] = np.logical_and(init_state == 0, np.logical_not(switched)).sum(axis=-1)
    count_mat[:,0,1] = np.logical_and(init_state == 0, switched).sum(axis=-1)
    count_mat[:,1,0] = np.logical_and(init_state == 1, switched).sum(axis=-1)
    count_mat[:,1,1] = np.logical_and(init_state == 1, np.logical_not(switched)).sum(axis=-1)

    return count_mat, start_stt

def reset_failure(data, **kwargs):
    counts, start_stt = count_matrices(data, **kwargs)
    switched_stt = int(1 - start_stt)
    num_failed = np.array([c[switched_stt,switched_stt] + c[switched_stt, start_stt] for c in counts])
    return num_failed

def switching_BER(data, **kwargs):
    """ Process data for BER experiment. """
    counts, start_stt = count_matrices(data, **kwargs)
    count_mat = counts[0]
    switched_stt = int(1 - start_stt)
    mean = beta.mean(1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    limit = beta.mean(1+count_mat[start_stt,switched_stt]+count_mat[start_stt,start_stt], 1)
    ci68 = beta.interval(0.68, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    ci95 = beta.interval(0.95, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    return mean, limit, ci68, ci95

def plot_BER(volts, multidata, **kwargs):
    ber_dat = [switching_BER(data, **kwargs) for data in multidata]
    mean = []; limit = []; ci68 = []; ci95 = []
    for datum in ber_dat:
        mean.append(datum[0])
        limit.append(datum[1])
        ci68.append(datum[2])
        ci95.append(datum[3])
    mean = np.array(mean)
    limit = np.array(limit)
    fig = plt.figure()
    plt.semilogy(volts, 1-mean, '-o')
    plt.semilogy(volts, 1-limit, linestyle="--")
    plt.fill_between(volts, [1-ci[0] for ci in ci68], [1-ci[1] for ci in ci68],  alpha=0.2, edgecolor="none")
    plt.fill_between(volts, [1-ci[0] for ci in ci95], [1-ci[1] for ci in ci95],  alpha=0.2, edgecolor="none")
    plt.ylabel("Switching Error Rate", size=14)
    plt.xlabel("Pulse Voltage (V)", size=14)
    plt.title("Bit Error Rate", size=16)
    return fig

def phase_diagram_grid(x_vals, y_vals, data,
                            title="Phase diagram",
                            xlabel="Pulse Duration (s)",
                            ylabel="Pulse Amplitude (V)"):
    fig = plt.figure()
    data = data.reshape(len(y_vals), len(x_vals), order='F')
    plt.pcolormesh(x_vals, y_vals, data, cmap="RdGy")
    plt.colorbar()
    plt.title(title, size=16)
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    return fig

def scaled_Delaunay(points):
    """ Return a scaled Delaunay mesh and scale factors """
    scale_factors = []
    points = np.array(points)
    for i in range(points.shape[1]):
    	scale_factors.append(1.0/np.mean(points[:,i]))
    	points[:,i] = points[:,i]*scale_factors[-1]
    mesh = Delaunay(points)
    return mesh, scale_factors

def phase_diagram_mesh(points, values,
                                title="Phase diagram",
                                xlabel="Pulse Duration (s)",
                                ylabel="Pulse Amplitude (V)",
                                shading="flat",
                                voronoi=False, **kwargs):
    fig = plt.figure()
    if voronoi:
        from scipy.spatial import Voronoi, voronoi_plot_2d
        points[:,0] *= 1e9
        vor = Voronoi(points)
        cmap = mpl.cm.get_cmap('RdGy')
        # colorize
        for pr, v in zip(vor.point_region, values):
            region = vor.regions[pr]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                plt.fill(*zip(*polygon), color=cmap(v))
    else:
        mesh, scale_factors = scaled_Delaunay(points)
        xs = mesh.points[:,0]/scale_factors[0]
        ys = mesh.points[:,1]/scale_factors[1]
        plt.tripcolor(xs,ys,mesh.simplices.copy(),values, cmap="RdGy",shading=shading,**kwargs)
    plt.xlim(min(xs),max(xs))
    plt.ylim(min(ys),max(ys))
    plt.title(title, size=18)
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    plt.colorbar()
    return fig

def crossover_pairs(points, values, threshold):
    """ Find all pairs of points whose values are on the two sides of threshold """
    mesh, scale_factors = scaled_Delaunay(points)
    nb_indices, indptr = mesh.vertex_neighbor_vertices
    pairs = []
    for k, value in enumerate(values):
        nbs = indptr[nb_indices[k]:nb_indices[k+1]]
        for nb in nbs:
            if (value-threshold)*(values[nb]-threshold) < 0:
                pairs.append([k,nb])
    return np.array(pairs)

def load_refined_switching_data(filename, start_state=None, failure=False,
                        threshold=None, display=False,
                        voltage_scale_factor=1.0, duration_scale_factor=1.0):
    with h5py.File(filename, 'r') as f:
        logger.debug("Read data from file: %s" % filename)
        durations = duration_scale_factor*np.array([f['axes'][k].value for k in f['axes'].keys() if "duration-data" in k])
        voltages  = voltage_scale_factor*np.array([f['axes'][k].value for k in f['axes'].keys() if "voltage-data" in k])
        dsets     = np.array([f[k].value for k in f.keys() if "data" in k])
        data      = np.concatenate(dsets, axis=0)
        voltages  = np.concatenate(voltages, axis=0)
        durations = np.concatenate(durations, axis=0)
    data_mean = np.mean(data, axis=-1)
    points = np.array([durations, voltages]).transpose()
    if failure:
        return points, reset_failure(data_mean,start_state=start_state, display=display)
    else:
        return points, switching_phase(data_mean,start_state=start_state,threshold=threshold, display=display)

def load_BER_data(filename):
    with h5py.File(filename, 'r') as f:
        dsets = [f[k] for k in f.keys() if "data" in k]
        data_mean = [np.mean(dset.value, axis=-1) for dset in dsets]
        volts = [float(dset.attrs['pulse_voltage']) for dset in dsets]
    return volts, data_mean
