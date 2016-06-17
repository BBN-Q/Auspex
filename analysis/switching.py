import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

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
                            xlabel="Pulse Duration (ns)",
                            ylabel="Pulse Amplitude (V)"):
    fig = plt.figure()
    data = data.reshape(len(y_vals), len(x_vals), order='F')
    plt.pcolormesh(x_vals, y_vals, data, cmap="RdGy")
    plt.colorbar()
    plt.title(title, size=16)
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    return fig

def phase_diagram_mesh(points, values,
                                title="Phase diagram",
                                xlabel="Pulse Duration (ns)",
                                ylabel="Pulse Amplitude (V)",**kwargs):
    mesh = Delaunay(points)
    fig = plt.figure()
    fig.tripcolor(mesh.points[:,0],mesh.points[:,1],mesh.simplices.copy(),values,**kwargs)
    plt.title(title, size=16)
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    return fig
