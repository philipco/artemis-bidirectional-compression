"""
Created by Philippenko, 10 January 2020.

This class prepare the real dataset for usage.
"""
import pandas as pd
import os

import torch
from sklearn.preprocessing import scale

from src.utils.Utilities import pickle_loader, pickle_saver, file_exist
from src.utils.data.DataClustering import find_cluster, clustering_data, tsne, check_data_clusterisation
from src.utils.data.DataPreparation import add_bias_term


def get_project_root() -> str:
    import pathlib
    path = str(pathlib.Path().absolute())
    if not path.find("artemis"):
        raise ValueError("Current directory looks to be higher than root of the project: {}".format(path))
    split = path.split("artemis")
    return split[0] + "artemis"

def prepare_dataset_by_device(X_merged, Y_merged, nb_devices: int):

    number_of_items = len(X_merged)
    number_of_items_by_devices = number_of_items // nb_devices
    print("Number of points by devices: ", number_of_items_by_devices)

    X, Y = [], []
    for i in range(nb_devices):
        X.append(X_merged[number_of_items_by_devices * i:number_of_items_by_devices * (i + 1)])
        Y_temp = Y_merged[number_of_items_by_devices * i:number_of_items_by_devices * (i + 1)]
        Y.append(torch.stack([y[0] for y in Y_temp]))

    assert len(X) == nb_devices, \
        "The number of groups in the dataset is not equal to the number of device."

    # Adding a columns of "1" to take into account a potential bias.
    X = add_bias_term(X)
    return X, Y

def prepare_noniid_dataset(data, pivot_label: str, filename: str, nb_cluster: int, double_check: bool =False):

    path = "{0}/notebook".format(get_project_root())
    tsne_file = "{0}-tsne".format(filename)
    if not file_exist("{0}.pkl".format(tsne_file), "{0}/pickle".format(path)):
        # Running TNSE to obtain a 2D representation of data
        print("The TSNE ({0}) representation doesn't exist at this location : {1}"
              .format(tsne_file, "{0}/pickle".format(path)))
        embedded_data = tsne(data)
        pickle_saver(embedded_data, "{0}-tsne".format(filename), path)


    embedded_data = pickle_loader("{0}-tsne".format(filename), path)
    # Finding clusters in the TNSE
    print("Finding non-iid clusters in the TNSE represesentation: {0}.pkl".format(tsne_file))
    predicted_cluster = find_cluster(embedded_data, nb_cluster)
    # With the found clusters, splitting data.
    X, Y = clustering_data(data, predicted_cluster, pivot_label, nb_cluster)

    if double_check:
        print("Checking data cluserization, wait until completion before seeing the plots.")
        # Checking that splitting data by cluster is valid.
        check_data_clusterisation(X, Y, nb_cluster)

    return X, Y

def prepare_superconduct(nb_devices: int, iid: bool = True, double_check: bool =False):
    data = pd.read_csv('{0}/dataset/superconduct/train.csv'.format(get_project_root()), sep=",")
    if data.isnull().values.any():
        print("There is missing value.")
    else:
        print("No missing value. Great !")
    print("Scaling data.")
    scaled_data = scale(data)
    data = pd.DataFrame(data=scaled_data, columns = data.columns)
    X_data = data.loc[:, data.columns != "critical_temp"]
    Y_data = data.loc[:, data.columns == "critical_temp"]
    dim = len(X_data.columns)
    print("There is " + str(dim) + " dimensions.")

    print("Head of the dataset:")
    print(data.head())

    if iid:
        X_merged = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(data, "critical_temp", "superconduct", nb_devices, double_check)
    return X, Y, dim

def prepare_quantum(nb_devices: int, iid: bool = True, double_check: bool =False):
    data = pd.read_csv('{0}/dataset/quantum/phy_train.csv'.format(get_project_root()), sep="\t", header=None)

    # Looking for missing values.
    columns_with_missing_values = []
    for col in range(1, len(data.columns)):
        if (not data[data[col] == 999].empty) or (not data[data[col] == 9999].empty):
            columns_with_missing_values.append(col)
    print("Following columns has missing values:", columns_with_missing_values)
    data.drop(data.columns[columns_with_missing_values], axis=1, inplace=True)
    print("The columns with empty values have been removed.")
    data = data.rename(columns={0: "ID", 1: "state", 80: "nothing"})
    data = data.drop(['ID', 'nothing'], axis=1)
    data.head()

    # Looking for empty columns (with null std).
    small_std = []
    std_data = data.std()
    for i in range(len(data.columns)):
        if std_data.iloc[i] < 1e-5:
            small_std.append(i)
    print("This columns are empty: {0}".format(small_std))
    data.iloc[:, small_std].describe()

    # Removing columns with null std
    data = data.loc[:, (data.std() > 1e-6)]
    dim = len(data.columns) - 1
    print("Now, there is " + str(dim) + " dimensions.")

    data = data.replace({'state': {0: -1}})

    print("Head of the dataset (columns has not been re-indexed).")
    print(data.head())

    print("Labels repartition:")
    print(data['state'].value_counts())

    print("Scaling data.")
    scaled_data = scale(data.loc[:, data.columns != "state"])
    X_data = pd.DataFrame(data=scaled_data, columns=data.loc[:, data.columns != "state"].columns)
    Y_data = data.loc[:, data.columns == "state"] # We do not scale labels (+/-1).
    # Merging dataset in one :
    data = pd.concat([X_data, Y_data], axis=1, sort=False)

    if iid:
        # Transforming into torch.FloatTensor
        X_merged = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(data, "state", "quantum", nb_devices, double_check)

    return X, Y, dim
