"""
Created by Philippenko, 24th July 2020.

"""
import random
from copy import deepcopy

import torch
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from src.utils.data.DataPreparation import add_bias_term

dim_tnse_fig = (12, 9)


def palette(nb_cluster: int = 10):
    return sns.color_palette("bright", nb_cluster)


def tsne(data):
    """Compute the TSNE representation of a dataset."""
    np.random.seed(25)
    tsne = TSNE()
    X_embedded = tsne.fit_transform(scale(data))
    fig, ax = plt.subplots(figsize=dim_tnse_fig)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax).set_title("TSNE - 2D representation of data")
    return X_embedded


def dirichlet_sampling(data, target_column_name: int, nb_devices: int, beta: int = 1):
    """Returns for each data point, the index of the devices it will belong to."""
    print("Beta parameters of the dirichlet distribution: ", beta)
    Y_data = data.loc[:, data.columns == target_column_name].values
    labels = np.unique(Y_data)

    proportions = [np.random.dirichlet(np.repeat(beta, nb_devices)) for i in labels]
    indices_split_by_label = [np.where(np.array(Y_data) == i)[0] for i in labels]
    predicted_cluster = list(range(len(Y_data)))
    for idx_label in range(len(labels)):
        indices_by_label = indices_split_by_label[idx_label]
        proportion = proportions[idx_label]
        random.shuffle(indices_by_label)
        start, end = 0, 0
        for i in range(nb_devices):
            end += int(proportion[i] * len(indices_by_label))
            if i == nb_devices-1:
                end = len(indices_by_label)
            for indices in indices_by_label[start:end]:
                predicted_cluster[indices] = i
            start = end

    # TODO : à supprimer ?
    print([len(np.where(np.array(predicted_cluster) == i)[0]) for i in range(20)])
    return np.array(predicted_cluster)


def find_cluster(embedded_data, tsne_cluster_file, nb_cluster: int = 10):
    """Find cluster in a dataset."""
    np.random.seed(25)
    clustering = GaussianMixture(n_components=nb_cluster, random_state=0, tol=1e-6, n_init=4, max_iter=2000)\
        .fit(embedded_data)
    predicted_cluster = clustering.predict(embedded_data)

    fig, ax = plt.subplots(figsize=dim_tnse_fig)
    sns.scatterplot(embedded_data[:, 0], embedded_data[:, 1], ax=ax, hue=predicted_cluster, legend='full',
                    palette=palette(nb_cluster))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    # plt.title("Gaussian Mixture - Finding clusters in the TSNE", fontsize=20)
    ax.get_legend().remove()

    plt.savefig('{0}.eps'.format(tsne_cluster_file), format='eps')

    return predicted_cluster


def clustering_data(data, clustered_indices, target_column_name: str, nb_cluster: int = 10):
    """
    Split a dataset in clusters (and add a column for bias in each cluster of data).

    :param data:
    :param clustered_indices:
    :param target_column_name:
    :param nb_cluster:
    :return:
    """

    # Separing features and labels
    Y_data = data.loc[:, data.columns == target_column_name].values
    X_data = data.loc[:, data.columns != target_column_name].to_numpy()

    X_data = scale(X_data)
    if not np.sort(np.unique(Y_data)).tolist() == [-1.0, 1.0]:
        Y_data = scale(Y_data)

    X_data = scale(X_data)
    if not np.sort(np.unique(Y_data)).tolist() == [-1.0, 1.0]:
        Y_data = scale(Y_data)

    X, Y = [], []
    for i in range(nb_cluster):
        indices = np.where(np.array(clustered_indices) == i)

        X_sub_data = X_data[indices]
        Y_sub_data = Y_data[indices]

        X.append(torch.tensor(X_sub_data, dtype=torch.float64))
        Y.append(torch.stack([y[0] for y in torch.tensor(Y_sub_data, dtype=torch.float64)]))

    nb_devices = len(X)
    print("There is {0} devices.".format(nb_devices))

    # Adding a columns of "1" to take into account a potential bias.
    X = add_bias_term(X)
    return X, Y


def rebalancing_clusters(X_origin, Y_origin):
    """If the clusters are too unbalanced w.r.t. the number of elements, rebalance clusters."""
    MAX_RATIO = 10
    X, Y = deepcopy(X_origin), deepcopy(Y_origin)
    do = True
    cpt = 0
    while do:
        do = False
        lenghts = [len(y) for y in Y]
        min_lenght = min(lenghts), np.argmin(lenghts)
        max_lenght = max(lenghts), np.argmax(lenghts)
        if min_lenght[0] * MAX_RATIO < max_lenght[0]:
            print("Changing : ")
            print(min_lenght)
            print(max_lenght)
            do = True
            cpt += 1
            #Merging the smallest cluster with half of the biggest.
            X[min_lenght[1]] = torch.cat((X[min_lenght[1]], X[max_lenght[1]][:int(max_lenght[0] / 2)]))
            Y[min_lenght[1]] = torch.cat((Y[min_lenght[1]], Y[max_lenght[1]][:int(max_lenght[0] / 2)]))

            # Keeping in the largest cluster the second half of its data.
            X[max_lenght[1]] = X[max_lenght[1]][int(max_lenght[0] / 2):]
            Y[max_lenght[1]] = Y[max_lenght[1]][int(max_lenght[0] / 2):]
    return X, Y



def check_data_clusterisation(X, Y, nb_devices:int = 10):
    # Rebuilding data : removing columns of 1, merging all states, unsqueezing and finaly, merging features and states.
    rebuild_data = torch.cat([torch.tensor(torch.cat(Y), dtype=torch.float64).unsqueeze(1), torch.cat(X)[:, 1:]], dim=1)
    label = [i for i in range(nb_devices) for j in range(len(X[i]))]

    # Running TSNE to find cluster.
    tsne = TSNE()
    X_embedded_check = tsne.fit_transform(rebuild_data)

    # Plotting TSNE with labels.
    fig, ax = plt.subplots(figsize=dim_tnse_fig)
    sns.scatterplot(X_embedded_check[:, 0], X_embedded_check[:, 1], ax=ax, hue=label, legend='full',
                    palette=palette(nb_devices)).set_title("Checking that data clusterisation on each device in correct")

    ax.get_legend().remove()