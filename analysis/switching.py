import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
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
    starting_state = np.argmax(initial_state_fractions)
    switched_state = 1 - starting_state
    print("Most frequenctly occuring initial state: {} (with {}% probability)".format(starting_state,
                                                            initial_state_fractions[starting_state]))

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

    mean = np.array([beta.mean(1+c[starting_state,switched_state],
                               1+c[starting_state,starting_state]) for c in counts])
    return mean

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
                                ylabel="Pulse Amplitude (V)",**kwargs):
    mesh, scale_factors = scaled_Delaunay(points)
    xs = mesh.points[:,0]/scale_factors[0]
    ys = mesh.points[:,1]/scale_factors[1]
    fig = plt.figure()
    plt.tripcolor(xs,ys,mesh.simplices.copy(),values,cmap="RdGy",shading="flat",**kwargs)
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
