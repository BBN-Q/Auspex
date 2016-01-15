import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import beta

def switching_plots(buffers, axis1, num_clusters=2):
    #Get an idea of SNR
    #Cluster all the data into three based with starting point based on edges
    all_vals = buffers.flatten()
    all_vals.resize((all_vals.size,1))
    init_guess = np.linspace(np.min(all_vals), np.max(all_vals), num_clusters)
    init_guess[[1,-1]] = init_guess[[-1,1]]
    init_guess.resize((num_clusters,1))
    clusterer = KMeans(init=init_guess, n_clusters=num_clusters)
    state = clusterer.fit_predict(all_vals)

    #Report initial state distributions
    print("Total initial state distribution:")
    init_state = state[::2]
    for ct in range(num_clusters):
        print("\tState {}: {:.2f}%".format(ct, 100*np.sum(init_state == ct)/len(init_state)))

    #Approximate SNR from centre distance and variance
    std0 = np.std(all_vals[state == 0])
    std1 = np.std(all_vals[state == 1])
    mean_std = 0.5*(std0 + std1)
    centre0 = clusterer.cluster_centers_[0,0]
    centre1 = clusterer.cluster_centers_[1,0]
    centre_dist = centre1 - centre0
    print("Centre distance = {:.3f} with widths = {:.4f} / {:.4f} gives SNR ratio {:.3}".format(centre_dist, std0, std1, centre_dist/mean_std))

    #Have a look at the distributions
    plt.figure()
    for ct in range(num_clusters):
        sns.distplot(all_vals[state == ct], kde=False, norm_hist=False)

    #calculate some switching matrices for each amplitude
    # 0->0 0->1
    # 1->0 1->1
    counts = []
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

    mean_PtoAP = np.array([beta.mean(1+c[0,1], 1+c[0,0]) for c in counts])
    limit_PtoAP = np.array([beta.mean(1+c[0,1]+c[0,0], 1) for c in counts])
    mean_APtoP = np.array([beta.mean(1+c[1,0], 1+c[1,1]) for c in counts])
    limit_APtoP = np.array([beta.mean(1+c[1,0]+c[1,1], 1) for c in counts])
    ci68_PtoAP = np.array([beta.interval(0.68, 1+c[0,1], 1+c[0,0]) for c in counts])
    ci68_APtoP = np.array([beta.interval(0.68, 1+c[1,0], 1+c[1,1]) for c in counts])
    ci95_PtoAP = np.array([beta.interval(0.95, 1+c[0,1], 1+c[0,0]) for c in counts])
    ci95_APtoP = np.array([beta.interval(0.95, 1+c[1,0], 1+c[1,1]) for c in counts])

    # import h5py
    # FID = h5py.File("data/PSPL-HighStatistics-vs-Dur-2.h5", "w")
    # FID.create_dataset("/buffer", data=buffers, compression="lzf")
    # FID.create_dataset("/durations", data=durations, compression="lzf")
    # FID.close()

    plt.figure()
    # volts = 7.5*np.power(10, (-5+attens)/20)
    current_palette = sns.color_palette()
    plt.plot(axis1, mean_PtoAP)
    plt.fill_between(axis1, [ci[0] for ci in ci68_PtoAP], [ci[1] for ci in ci68_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.fill_between(axis1, [ci[0] for ci in ci95_PtoAP], [ci[1] for ci in ci95_PtoAP], color=current_palette[0], alpha=0.2, edgecolor="none")
    plt.plot(axis1, mean_APtoP)
    plt.fill_between(axis1, [ci[0] for ci in ci68_APtoP], [ci[1] for ci in ci68_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")
    plt.fill_between(axis1, [ci[0] for ci in ci95_APtoP], [ci[1] for ci in ci95_APtoP], color=current_palette[1], alpha=0.2, edgecolor="none")
    # plt.xlabel("Pulse Amp (V)")
    plt.xlabel("Pulse Duration (ns)")
    plt.ylabel("Estimated Switching Probability")
    # plt.title("P to AP")
    # means_diagram_PtoAP = mean_PtoAP.reshape(len(attens), len(durations), order='F')
    # plt.pcolormesh(axis1, volts, means_diagram_PtoAP, cmap="RdGy")
    # plt.colorbar()
    # plt.figure()
    # plt.title("AP to P")
    # means_diagram_APtoP = mean_APtoP.reshape(len(attens), len(durations), order='F')
    # plt.pcolormesh(axis1, volts, means_diagram_APtoP, cmap="RdGy")
    # plt.colorbar()

    plt.figure()
    plt.semilogy(axis1, 1-mean_PtoAP)
    plt.semilogy(axis1, 1-limit_PtoAP, color=current_palette[0], linestyle="--")
    plt.semilogy(axis1, 1-mean_APtoP)
    plt.semilogy(axis1, 1-limit_APtoP, color=current_palette[1], linestyle="--")
    plt.ylabel("Switching Error Rate")
    plt.xlabel("Pulse Duration (ns)")

    plt.show()
