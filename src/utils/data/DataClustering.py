"""
Created by Philippenko, 24th July 2020.

"""
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
    tsne = TSNE()
    X_embedded = tsne.fit_transform(scale(data))
    fig, ax = plt.subplots(figsize=dim_tnse_fig)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], ax=ax).set_title("TSNE - 2D representation of data")
    return X_embedded


def find_cluster(embedded_data, nb_cluster: int = 10):
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

    return predicted_cluster


def clustering_data(data, predicted_cluster, column_name: str, nb_cluster: int = 10):
    """
    Split a dataset in clusters (and add a column for bias in each cluster of data).

    :param data:
    :param predicted_cluster:
    :param column_name:
    :param nb_cluster:
    :return:
    """

    # Separing features and labels
    Y_data = data.loc[:, data.columns == column_name].values
    X_data = data.loc[:, data.columns != column_name].to_numpy()

    X, Y = [], []
    for i in range(nb_cluster):
        indices = np.where(np.array(predicted_cluster) == i)

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
    X, Y = deepcopy(X_origin), deepcopy(Y_origin)
    do = True
    cpt = 0
    while do:
        do = False
        lenghts = [len(y) for y in Y]
        min_lenght = min(lenghts), np.argmin(lenghts)
        max_lenght = max(lenghts), np.argmax(lenghts)
        if min_lenght[0] * 10 < max_lenght[0]:
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
    rebuild_data = torch.cat([torch.tensor(scale(torch.cat(Y)), dtype=torch.float64).unsqueeze(1), torch.cat(X)[:, 1:]],
                             dim=1)
    label = [i for i in range(nb_devices) for j in range(len(X[i]))]

    # Running TSNE to find cluster.
    tsne = TSNE()
    X_embedded_check = tsne.fit_transform(rebuild_data)

    # Plotting TSNE with labels.
    fig, ax = plt.subplots(figsize=dim_tnse_fig)
    sns.scatterplot(X_embedded_check[:, 0], X_embedded_check[:, 1], ax=ax, hue=label, legend='full',
                    palette=palette(nb_devices)).set_title("Checking that data clusterisation on each device in correct")

    ax.get_legend().remove()