"""
Created by Philippenko, 10 January 2020.

This class prepares a9a, phishing, quantum, superconduct, w8a, ...
"""
import numpy as np
import pandas as pd
import logging

import torch
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import scale, LabelEncoder

from src.utils.PathDataset import get_path_to_datasets
from src.utils.PickletHandler import pickle_loader, pickle_saver
from src.utils.Utilities import file_exist
from src.utils.data.DataClustering import find_cluster, clustering_data, tsne, check_data_clusterisation, \
    rebalancing_clusters, dirichlet_sampling
from src.utils.data.DataPreparation import add_bias_term


def get_preparation_operator_of_dataset(dataset):
    # Select the correct dataset
    if dataset == "a9a":
        return prepare_a9a

    if dataset == "abalone":
        return prepare_abalone

    if dataset == "covtype":
        return prepare_covtype

    if dataset == "gisette":
        return prepare_gisette

    if dataset == "madelon":
        return prepare_madelon

    if dataset == "mushroom":
        return prepare_mushroom

    if dataset == "quantum":
        return prepare_quantum

    if dataset == "phishing":
        return prepare_phishing

    elif dataset == "superconduct":
        return prepare_superconduct

    if dataset == "w8a":
        return prepare_w8a


def prepare_dataset_by_device(X_merged, Y_merged, nb_devices: int):

    number_of_items = len(X_merged)
    number_of_items_by_devices = number_of_items // nb_devices
    logging.debug("Number of points by devices: ", number_of_items_by_devices)

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


def prepare_noniid_dataset(data, target_label: str, data_path: str, pickle_path: str, nb_cluster: int,
                           dirichlet: int = None, double_check: bool =False):
    if dirichlet is not None:
        X, Y = dirichlet_preparation(data, target_label, pickle_path, nb_cluster, dirichlet)
    else:
        X, Y = TSNE_prepration(data, target_label, data_path, pickle_path, nb_cluster, double_check)

    for y in Y:
        print("Nb of points:", len(y))

    return X, Y


def dirichlet_preparation(data, target_column_name: str, pickle_path: str, nb_devices: int, dirichlet: int):
    print("Dirichlet-based client distribution.")
    dirichlet_file = "{0}-dirichlet-{1}".format(pickle_path, dirichlet)
    if not file_exist("{0}.pkl".format(dirichlet_file)):
        # Sampling according to Dirichlet distribution.
        logging.debug("The TSNE representation ({0}) doesn't exist."
                      .format(dirichlet_file))

        clustered_indices = dirichlet_sampling(data, target_column_name=target_column_name, nb_devices=nb_devices,
                                               beta=dirichlet)
        pickle_saver(clustered_indices, dirichlet_file)

    clustered_indices = pickle_loader("{0}".format(dirichlet_file))

    # With the found clusters, splitting data.
    X, Y = clustering_data(data, clustered_indices, target_column_name, nb_devices)
    return X, Y


def TSNE_prepration(data, target_label: str, data_path: str, pickle_path: str, nb_cluster: int,
                    double_check: bool =False):
    print("TSNE-based client distribution.")
    # The TSNE representation is independent of the number of devices.
    tsne_file = "{0}-tsne".format(data_path)
    if not file_exist("{0}.pkl".format(tsne_file)):
        # Running TNSE to obtain a 2D representation of data
        logging.debug("The TSNE representation ({0}) doesn't exist."
              .format(tsne_file))
        embedded_data = tsne(data)
        pickle_saver(embedded_data, tsne_file)

    tsne_cluster_file = "{0}/tsne-cluster".format(pickle_path)
    if not file_exist("{0}.pkl".format(tsne_cluster_file)):
        # Finding clusters in the TNSE.
        logging.debug("Finding non-iid clusters in the TNSE represesentation: {0}.pkl".format(tsne_file))
        embedded_data = pickle_loader("{0}".format(tsne_file))
        logging.debug("Saving found clusters : {0}.pkl".format(tsne_cluster_file))
        predicted_cluster = find_cluster(embedded_data, tsne_cluster_file, nb_cluster)
        pickle_saver(predicted_cluster, "{0}".format(tsne_cluster_file))

    predicted_cluster = pickle_loader("{0}".format(tsne_cluster_file))

    # With the found clusters, splitting data.
    X, Y = clustering_data(data, predicted_cluster, target_label, nb_cluster)

    if double_check:
        logging.debug("Checking data cluserization, wait until completion before seeing the plots.")
        # Checking that splitting data by cluster is valid.
        check_data_clusterisation(X, Y, nb_cluster)

    return X, Y

def prepare_superconduct(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                         double_check: bool = False):
    raw_data = pd.read_csv('{0}/dataset/superconduct/train.csv'.format(get_path_to_datasets()), sep=",")
    if raw_data.isnull().values.any():
        logging.warning("There is missing value.")
    else:
        logging.debug("No missing value. Great !")
    logging.debug("Scaling data.")

    X_data = raw_data.loc[:, raw_data.columns != "critical_temp"]
    Y_data = raw_data.loc[:, raw_data.columns == "critical_temp"]
    dim = len(X_data.columns)
    logging.debug("There is " + str(dim) + " dimensions.")

    logging.debug("Head of the dataset:")
    logging.debug(raw_data.head())

    if iid:
        X_tensor = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "critical_temp", data_path + "/superconduct", pickle_path, nb_devices,
                                      dirichlet, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_quantum(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):

    raw_data= pd.read_csv('{0}/dataset/quantum/phy_train.csv'.format(get_path_to_datasets()), sep="\t", header=None)

    # Looking for missing values.
    columns_with_missing_values = []
    for col in range(1, len(raw_data.columns)):
        if (not raw_data[raw_data[col] == 999].empty) or (not raw_data[raw_data[col] == 9999].empty):
            columns_with_missing_values.append(col)
    logging.debug("Following columns has missing values:", columns_with_missing_values)
    raw_data.drop(raw_data.columns[columns_with_missing_values], axis=1, inplace=True)
    logging.debug("The columns with empty values have been removed.")
    raw_data = raw_data.rename(columns={0: "ID", 1: "state", 80: "nothing"})
    raw_data = raw_data.drop(['ID', 'nothing'], axis=1)
    raw_data.head()

    # Looking for empty columns (with null std).
    small_std = []
    std_data = raw_data.std()
    for i in range(len(raw_data.columns)):
        if std_data.iloc[i] < 1e-5:
            small_std.append(i)
    logging.debug("This columns are empty: {0}".format(small_std))
    raw_data.iloc[:, small_std].describe()

    # Removing columns with null std
    raw_data = raw_data.loc[:, (raw_data.std() > 1e-6)]
    dim = len(raw_data.columns) - 1 # The dataset still contains the label
    logging.debug("Now, there is " + str(dim) + " dimensions.")

    raw_data = raw_data.replace({'state': {0: -1}})

    logging.debug("Head of the dataset (columns has not been re-indexed).")
    logging.debug(raw_data.head())

    logging.debug("Labels repartition:")
    logging.debug(raw_data['state'].value_counts())

    X_data = raw_data.loc[:, raw_data.columns != "state"]
    Y_data = raw_data.loc[:, raw_data.columns == "state"]  # We do not scale labels (+/-1).

    if iid:
        # Transforming into torch.FloatTensor
        X_tensor = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "state", data_path + "/quantum", pickle_path, nb_devices, dirichlet, double_check)

    return X, Y, dim + 1 # Because we added one column for the bias

def prepare_mushroom(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                     double_check: bool =False):
    raw_data = pd.read_csv('{0}/dataset/mushroom/mushrooms.csv'.format(get_path_to_datasets()))

    # The data is categorial so I convert it with LabelEncoder to transfer to ordinal.
    labelencoder = LabelEncoder()
    for column in raw_data.columns:
        raw_data[column] = labelencoder.fit_transform(raw_data[column])

    # It can be seen that the column "veil-type" is 0 and not contributing to the data so I remove it.
    raw_data = raw_data.drop(["veil-type"], axis=1)
    raw_data = raw_data.replace({'class': {0: -1}})

    Y_data = raw_data.loc[:, raw_data.columns == "class"]
    X_data = raw_data.loc[:, raw_data.columns != "class"]

    dim = len(X_data.columns)

    if iid:
        X_merged = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "class", data_path + "/mushroom", pickle_path, nb_devices,
                                      dirichlet, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_phishing(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                     double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/phishing/phishing.txt".format(get_path_to_datasets()))

    for i in range(len(raw_Y)):
        if raw_Y[i] == 0:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X.todense())
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/phishing", pickle_path, nb_devices, dirichlet, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_a9a(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                double_check: bool =False, test: bool = False):

    if not test:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/a9a/a9a.txt".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
    else:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/a9a/a9a_test.txt".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
        raw_X = np.c_[raw_X, np.zeros((len(raw_Y)))]


    for i in range(len(raw_Y)):
        if raw_Y[i] == 0:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:

        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/a9a", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_abalone(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):
    raw_data = pd.read_csv('{0}/dataset/abalone/abalone.csv'.format(get_path_to_datasets()), sep=",", header = None)

    raw_data = raw_data.rename(columns={ 0: "gender", 1: "Length", 2: "Diameter", 3: "Height", 8: "rings"})
    labelencoder = LabelEncoder()
    raw_data["gender"] = labelencoder.fit_transform(raw_data["gender"])

    X_data = raw_data.loc[:, raw_data.columns != "rings"]
    Y_data = raw_data.loc[:, raw_data.columns == "rings"]

    dim = len(X_data.columns)

    if iid:
        X_merged = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "rings", data_path + "/abalone", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_covtype(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/covtype/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    for i in range(len(raw_Y)):
        if raw_Y[i] == 2:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/covtype", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_madelon(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):

    X_data = pd.read_csv('{0}/dataset/madelon/madelon_train.data'.format(get_path_to_datasets()), sep=" ", header=None)
    X_data.drop(X_data.columns[len(X_data.columns) - 1], axis=1, inplace=True)

    dim = len(X_data.columns)

    Y_data = pd.read_csv('{0}/dataset/madelon/madelon_train.labels'.format(get_path_to_datasets()), header=None)

    raw_data = X_data.copy()
    raw_data["target"] = Y_data.values

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/madelon", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_covtype(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                    double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/covtype/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    for i in range(len(raw_Y)):
        if raw_Y[i] == 2:
            raw_Y[i] = -1

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/covtype", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_gisette(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None, double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/gisette/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/gisette", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1  # Because we added one column for the bias


def prepare_w8a(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, dirichlet: int = None,
                double_check: bool = False, test: bool = False):

    if not test:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/w8a/w8a".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
    else:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/w8a/w8a.t".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
        raw_X = np.c_[raw_X, np.zeros((len(raw_Y)))]

    raw_data = pd.DataFrame(data=raw_X)
    raw_data["target"] = raw_Y
    dim = len(raw_data.columns) - 1

    X_data = raw_data.loc[:, raw_data.columns != "target"]
    Y_data = raw_data.loc[:, raw_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(X_data, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(raw_data, "target", data_path + "/w8a", pickle_path, nb_devices, dirichlet,
                                      double_check)
    return X, Y, dim + 1 # Because we added one column for the bias