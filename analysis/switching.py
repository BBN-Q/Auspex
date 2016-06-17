import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import beta

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

def switching_phase_diagram(data):
    num_clusters = 2
    data = average_data()
    clusterer = cluster(data)
    state = clusterer.fit_predict(all_vals)

    init_state  = state[::2]

    initial_state_fractions = [np.sum(init_state == ct)/len(init_state) for ct in range(num_clusters)]
    starting_state = np.argmax(initial_state_fractions)
    switched_state = 1 - init_state
    print("Most frequenctly occuring initial state: {} (with {}% probability)".format(starting_state,
                                                            initial_state_fractions[starting_state]))
    for buf in buffers:
        state = clusterer.predict(buf.reshape((len(buf),1)))
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

def render_phase_diagram(x_values, y_values, x_title="", y_title=""):
