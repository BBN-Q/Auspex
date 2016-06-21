import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

def cluster(data, num_clusters=2):
    all_vals = data.flatten()
    all_vals.resize((all_vals.size,1))
    init_guess = np.linspace(np.min(all_vals), np.max(all_vals), num_clusters)
    init_guess[[1,-1]] = init_guess[[-1,1]]
    init_guess.resize((num_clusters,1))
    clusterer = KMeans(init=init_guess, n_clusters=num_clusters)
    return clusterer

def average_data(data, avg_points):
    return np.array([np.mean(d.reshape(avg_points, -1, order="F"), axis=0) for d in data])

def switching_phase(data):
    num_clusters = 2
    clusterer = cluster(data)
    all_vals = data.flatten()
    all_vals.resize((all_vals.size,1))
    state = clusterer.fit_predict(all_vals)

    init_state  = state[::2]
    initial_state_fractions = [np.sum(init_state == ct)/len(init_state) for ct in range(num_clusters)]
    start_stt = np.argmax(initial_state_fractions)
    switched_stt = 1 - start_stt
    print("Most frequenctly occuring initial state: {} (with {}% probability)".format(start_stt,
                                                            initial_state_fractions[start_stt]))

    counts =[]
    for buf in data:
        state = clusterer.predict(buf.reshape((buf.size,1)))
        init_state = state[::2]
        final_state = state[1::2]
        switched = np.logical_xor(init_state, final_state)

        count_mat = np.zeros((2,2), dtype=np.int)

        count_mat[0,0] = np.sum(np.logical_and(init_state == 0, np.logical_not(switched) ))
        count_mat[0,1] = np.sum(np.logical_and(init_state == 0, switched ))
        count_mat[1,0] = np.sum(np.logical_and(init_state == 1, switched ))
        count_mat[1,1] = np.sum(np.logical_and(init_state == 1, np.logical_not(switched) ))

        counts.append(count_mat)

    mean = np.array([beta.mean(1+c[start_stt,switched_stt],
                               1+c[start_stt,start_stt]) for c in counts])
    return mean

def switching_BER(data):
    """ Process data for BER experiment. """
    num_clusters = 2
    clusterer = cluster(data)
    all_vals = data.flatten()
    all_vals.resize((all_vals.size,1))
    state = clusterer.fit_predict(all_vals)

    init_state  = state[::2]
    initial_state_fractions = [np.sum(init_state == ct)/len(init_state) for ct in range(num_clusters)]
    start_stt = np.argmax(initial_state_fractions)
    switched_stt = 1 - start_stt
    state = clusterer.predict(data.reshape((data.size,1)))
    init_state = state[::2]
    final_state = state[1::2]
    switched = np.logical_xor(init_state, final_state)

    count_mat = np.zeros((2,2), dtype=np.int)
    count_mat[0,0] = np.sum(np.logical_and(init_state == 0, np.logical_not(switched) ))
    count_mat[0,1] = np.sum(np.logical_and(init_state == 0, switched ))
    count_mat[1,0] = np.sum(np.logical_and(init_state == 1, switched ))
    count_mat[1,1] = np.sum(np.logical_and(init_state == 1, np.logical_not(switched) ))

    mean = beta.mean(1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    limit = beta.mean(1+count_mat[start_stt,switched_stt]+count_mat[start_stt,start_stt], 1)
    ci68 = beta.interval(0.68, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    ci95 = beta.interval(0.95, 1+count_mat[start_stt,switched_stt],1+count_mat[start_stt,start_stt])
    return mean, limit, ci68, ci95

def plot_BER(volts, multidata):
    ber_dat = [switching_BER(data) for data in multidata]
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
    plt.title(title, size=16)
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

def load_switching_data(filename):
    with h5py.File(filename, 'r') as f:
        durations = np.array([f['axes'][k].value for k in f['axes'].keys() if "pulse_duration-data" in k])
        voltages = np.array([f['axes'][k].value for k in f['axes'].keys() if "pulse_voltage-data" in k])
        dsets = np.array([f[k].value for k in f.keys() if "data" in k])
        data = np.concatenate(dsets, axis=0)
        voltages = np.concatenate(voltages, axis=0)
        durations = np.concatenate(durations, axis=0)
    data_mean = np.mean(data, axis=-1)
    points = np.array([durations, voltages]).transpose()
    return points, switching_phase(data_mean)

def load_BER_data(filename):
    with h5py.File(filename, 'r') as f:
        dsets = [f[k].value for k in f.keys() if "data" in k]
    data_mean = [np.mean(data, axis=-1) for data in dsets]
    return data_mean
