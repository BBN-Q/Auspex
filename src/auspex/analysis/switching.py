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

from auspex.analysis.io import load_from_HDF5
from auspex.log import logger

def load_switching_data(filename_or_fileobject, start_state=None, group="main", failure=False, threshold=None,
                        voltage_scale_factor=1.0, duration_scale_factor=1.0, data_name='voltage', data_filter=None):
    data, desc = load_from_HDF5(filename_or_fileobject, reshape=False)
    # Regular axes
    states = desc[group].axis("state").points
    reps   = desc[group].axis("attempt").points
    # Main data array, possibly unstructured
    dat = data[group][:].reshape((-1, reps.size, states.size))
    # Filter data if desired
    # e.g. filter_func = lambda dat: np.logical_and(dat['field'] == 0.04, dat['temperature'] == 4.0)
    if data_filter:
        dat = dat[np.where(data_filter(dat))]

    Vs     = dat[data_name]
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
    logger.debug("Number of clusters: %d" % num_clusters)

    init_guess = np.linspace(np.min(all_vals), np.max(all_vals), num_clusters)
    init_guess[[1,-1]] = init_guess[[-1,1]]
    init_guess.resize((num_clusters,1))

    clust = KMeans(init=init_guess, n_clusters=num_clusters)
    state = clust.fit_predict(all_vals)

    for ct in range(num_clusters):
        logger.debug("Cluster {}: {} +/- {}".format(ct, all_vals[state==ct].mean(), all_vals[state==ct].std()))
    return clust

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
        logger.debug("Initial fraction of state %d: %f" %(ct, fraction))

    if start_state is not None and start_state in range(num_clusters):
        start_stt = start_state
    else:
        start_stt = np.argmax(init_state_frac)
    logger.debug("Start state set to state: {}".format(start_stt))
    logger.debug("Switched state is state: {}".format(1-start_stt))

    # This array contains a 2x2 count_matrix for each coordinate tuple
    count_mat = np.zeros((init_state.shape[0], 2, 2))

    # count_mat      = np.zeros((2,2), dtype=np.int)
    count_mat[:,0,0] = np.logical_and(init_state == 0, np.logical_not(switched)).sum(axis=-1)
    count_mat[:,0,1] = np.logical_and(init_state == 0, switched).sum(axis=-1)
    count_mat[:,1,0] = np.logical_and(init_state == 1, switched).sum(axis=-1)
    count_mat[:,1,1] = np.logical_and(init_state == 1, np.logical_not(switched)).sum(axis=-1)

    return count_mat, start_stt

def count_matrices_ber(data, start_state=None, threshold=None, display=None):
    num_clusters = 2
    if threshold is None:
        clust = clusterer(data)
        state = clust.fit_predict(data.reshape(-1, 1)).reshape((-1,2))
    else:
        logger.debug("Cluster data based on threshold = {}".format(threshold))
        state = data > threshold
        state = state.reshape((-1,2))

    init_state  = state[:,0]
    final_state = state[:,1]
    switched    = np.logical_xor(init_state, final_state)

    init_state_frac = [np.mean(init_state == ct) for ct in range(num_clusters)]
    for ct, fraction in enumerate(init_state_frac):
        logger.debug("Initial fraction of state %d: %f" %(ct, fraction))

    if start_state is not None and start_state in range(num_clusters):
        start_stt = start_state
    else:
        start_stt = np.argmax(init_state_frac)
    logger.debug("Start state set to state: {}".format(start_stt))
    logger.debug("Switched state is state: {}".format(1-start_stt))

    # This array contains a 2x2 count_matrix for each coordinate tuple
    count_mat = np.zeros((2, 2))

    # count_mat      = np.zeros((2,2), dtype=np.int)
    count_mat[0,0] = np.logical_and(init_state == 0, np.logical_not(switched)).sum()
    count_mat[0,1] = np.logical_and(init_state == 0, switched).sum()
    count_mat[1,0] = np.logical_and(init_state == 1, switched).sum()
    count_mat[1,1] = np.logical_and(init_state == 1, np.logical_not(switched)).sum()

    return count_mat, start_stt

def reset_failure(data, **kwargs):
    counts, start_stt = count_matrices(data, **kwargs)
    switched_stt = int(1 - start_stt)
    num_failed = np.array([c[switched_stt,switched_stt] + c[switched_stt, start_stt] for c in counts])
    return num_failed

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
    for i in range(points.shape[1]):
        mesh.points[:,i] = mesh.points[:,i]/scale_factors[i]
    return mesh

def phase_diagram_mesh(points, values,
                                title="Phase diagram",
                                xlabel="Pulse Duration (s)",
                                ylabel="Pulse Amplitude (V)",
                                shading="flat",
                                voronoi=False, **kwargs):
    # fig = plt.figure()
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
        mesh = scaled_Delaunay(points)
        xs = mesh.points[:,0]
        ys = mesh.points[:,1]
        plt.tripcolor(xs,ys,mesh.simplices.copy(),values, cmap="RdGy",shading=shading,**kwargs)
    plt.xlim(min(xs),max(xs))
    plt.ylim(min(ys),max(ys))
    plt.title(title, size=18)
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    plt.colorbar()
    # return fig

def crossover_pairs(points, values, threshold):
    """ Find all pairs of points whose values are on the two sides of threshold """
    mesh = scaled_Delaunay(points)
    nb_indices, indptr = mesh.vertex_neighbor_vertices
    pairs = []
    for k, value in enumerate(values):
        nbs = indptr[nb_indices[k]:nb_indices[k+1]]
        for nb in nbs:
            if (value-threshold)*(values[nb]-threshold) < 0:
                pairs.append([k,nb])
    return np.array(pairs)

# Functions for finding boundary points
def find_cross(point1, point2, cut = 0):
    """ Estimate by interpolation to find intersection between \
    y=cut line and the line connecting two given points
    """
    return point1[0] + (point2[0]-point1[0])*(cut-point1[1])/(point2[1]-point1[1])

def find_boundary(mesh,vals,threshold=0.5):
    """ Find boundary points on the phase diagram where the switching probability = threshold """
    boundary_points = []
    durs = mesh.points[:,0]
    volts = mesh.points[:,1]
    indices, indptr = mesh.vertex_neighbor_vertices
    for k in range(len(vals)):
        for k_nb in indptr[indices[k]:indices[k+1]]:
            if (vals[k]-threshold)*(vals[k_nb]-threshold)<0:
                x0 = find_cross([durs[k],vals[k]],[durs[k_nb],vals[k_nb]],cut=threshold)
                y0 = find_cross([volts[k],vals[k]],[volts[k_nb],vals[k_nb]],cut=threshold)
                boundary_points.append([x0,y0])

    boundary_points = np.array(boundary_points)
    if len(boundary_points) > 0:
        b = np.ascontiguousarray(boundary_points).view(np.dtype((np.void,
                            boundary_points.dtype.itemsize * boundary_points.shape[1])))
        _, idx = np.unique(b, return_index=True)
        boundary_points = boundary_points[idx]
        # Sort the boundary_points by x-axis
        boundary_points = sorted(boundary_points, key=itemgetter(0))
    return np.array(boundary_points)

def f_macrospin(t_data, t0, v0):
    return v0*(1.0 + t0/t_data)

def fit_macrospin(points, p0=(0.0,0.0)):
    """ Fit to the macrospin model """
    # For best results, first need to normalize the data points
    points = np.array(points)
    xscale = np.max(points[:,0])
    yscale = np.max(points[:,1])
    xnorm = points[:,0]/xscale
    ynorm = points[:,1]/yscale
    params, pcov = curve_fit(f_macrospin, xnorm, ynorm, p0=p0)
    perrs = np.sqrt(np.diag(pcov))
    # Rescale back to original scale
    params[0] = params[0]*xscale
    params[1] = params[1]*yscale
    perrs[0] = perrs[0]*xscale
    perrs[1] = perrs[1]*yscale
    return params, perrs

def find_closest(t, v, t0, v0):
    """ Find the closest point on the curve f = a + b/x
    to the given point (t,v)
    """
    a = v0
    b = v0*t0
    # Solve for intersection points
    eqn_coefs = [1/b, -t/b, 0, v-a, -b]
    tis = np.roots(eqn_coefs)
    tis = tis[abs(tis.imag/tis.real)<0.01].real # We care only real solutions
    tis = tis[tis>0] # and positive ones
    # Choose the shortest among solutions
    ds = abs(tis-t)*np.sqrt(1 + np.power(tis,4)/(b*b)) # Distance from solutions to given point (t,v)
    idx = np.argmin(ds)
    ti = tis[idx]
    vi = a + b/ti
    return ti, vi
def dX(x,b):
    """ dX = sqrt(1 + df^2)*dx for f = a + b/x"""
    return np.sqrt(1 + b**2/x**4)
def x2X(x,xo,b):
    """Convert from original x to new X
    xo: The coordinates of the new origin (xo,yo).
        Here we need xo only
    """
    return integrate.quad(dX,xo,x,args=(b))[0]
def y2Y(y0,yi,a,b):
    """Convert from original y0 to new Y
    yi: The y-coordinate of the point (xi,yi) on the f=a+b/x curve
    that is closest to the given point (x0,y0)"""
    return (y0-yi)*np.sqrt(1+np.power(yi-a,4)/(b*b))

def perp_fit(ts, vs):

    def lsq_macrospin(p, ts, vs):
        t0 = p[0]
        v0 = p[1]
        a = v0
        b = t0*v0
        to = 1
        vo = a + b/to

        # Here is what we expect
        vs_ideal = v0*(1.0 + t0/ts)
        Xs = []
        Ys = []
        for t,v in zip(ts,vs):
            ti,vi = find_closest(t,v,t0,v0)
            Xs.append(x2X(ti,to,b))
            Ys.append(y2Y(v,vi,a,b))
        return np.power(Ys,2)
    p0 = [0.2, 100]
    p, flag = leastsq(lsq_macrospin, p0, args=(ts, vs))
    return p

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

def switching_BER(data, **kwargs):
    """ Process data for BER experiment. """
    count_mat, start_stt = count_matrices_ber(data, **kwargs)
    switched_stt = int(1 - start_stt)
    mean = beta.mean(1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    limit = beta.mean(1+count_mat[start_stt,switched_stt]+count_mat[start_stt,start_stt], 1)
    ci68 = beta.interval(0.68, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    ci95 = beta.interval(0.95, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    return mean, limit, ci68, ci95

def load_BER_data(filename_or_fileobject, start_state=None, group="main",
                  threshold=None, voltage=None, data_filter=None, SBER=False):
    # Load data from HDF5 object or filename
    data, desc = load_from_HDF5(filename_or_fileobject)
    # Regular data axes
    states = desc[group].axis("state").points
    reps   = desc[group].axis("attempt").points

    # Select group
    dat = data[group]

    # Filter data if desired
    # e.g. filter_func = lambda dat: np.logical_and(dat['field'] == 0.04, dat['temperature'] == 4.0)
    if data_filter:
        dat = dat[np.where(data_filter(dat))]

    # Either return data for a specific voltage or all voltages
    if voltage:
        dat = dat[np.where(dat['pulse_voltage'] == voltage)]
        voltages = np.array([voltage])
    else:
        # Get unique voltage values
        voltages = np.unique(dat['pulse_voltage'])

    # Select and process the relevant data
    means = []
    limits = []
    ci68s = []
    ci95s = []
    for v in voltages:
        # logger.info(f"There are {dat.shape} observations for voltage {v}")
        d = dat[np.where(dat['pulse_voltage'] == v)].reshape((-1, reps.size, states.size))
        mean, limit, ci68, ci95 = switching_BER(d['Data'], start_state=start_state, threshold=threshold)
        if SBER:
            means.append(mean)
            limits.append(limit)
            ci68s.append(np.array(ci68))
            ci95s.append(np.array(ci95))
        else:
            means.append(1-mean)
            limits.append(1-limit)
            ci68s.append(1-np.array(ci68))
            ci95s.append(1-np.array(ci95))

    return voltages, means, limits, ci68s, ci95s

def nTron_IQ(data, desc, rotate=True):
    iq_vals = data['Integrated']['Data']
    iq_vals -= iq_vals.mean()
    if rotate:
        slope, offset = np.polyfit(iq_vals.real, iq_vals.imag, 1)
        angle = np.arctan(slope)
        iq_vals *= np.exp(-1j*angle)
    return iq_vals

def nTron_IQ_plot(iq_vals, desc, threshold=0.0):
    iq_vals = iq_vals.real < threshold
    iqr = iq_vals.reshape(desc['Integrated'].dims(), order='C')
    iqrm = np.mean(iqr, axis=0)
    extent = (0.18, 10, 0.14, 0.40)
    aspect = 9.84/0.34
    plt.imshow(iqrm, origin='lower', cmap='RdGy', extent=extent, aspect=aspect)


# def plot_BER(volts, multidata, **kwargs):
#     ber_dat = [switching_BER(data, **kwargs) for data in multidata]
#     mean = []; limit = []; ci68 = []; ci95 = []
#     for datum in ber_dat:
#         mean.append(datum[0])
#         limit.append(datum[1])
#         ci68.append(datum[2])
#         ci95.append(datum[3])
#     mean = np.array(mean)
#     limit = np.array(limit)
#     fig = plt.figure()
#     plt.semilogy(volts, 1-mean, '-o')
#     plt.semilogy(volts, 1-limit, linestyle="--")
#     plt.fill_between(volts, [1-ci[0] for ci in ci68], [1-ci[1] for ci in ci68],  alpha=0.2, edgecolor="none")
#     plt.fill_between(volts, [1-ci[0] for ci in ci95], [1-ci[1] for ci in ci95],  alpha=0.2, edgecolor="none")
#     plt.ylabel("Switching Error Rate", size=14)
#     plt.xlabel("Pulse Voltage (V)", size=14)
#     plt.title("Bit Error Rate", size=16)
#     return fig

# def load_BER_data_legacy(filename):
#     with h5py.File(filename, 'r') as f:
#         dsets = [f[k] for k in f.keys() if "data" in k]
#         data_mean = [np.mean(dset.value, axis=-1) for dset in dsets]
#         volts = [float(dset.attrs['pulse_voltage']) for dset in dsets]
#     return volts, data_mean
