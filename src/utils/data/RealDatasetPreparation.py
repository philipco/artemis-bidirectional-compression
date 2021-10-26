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
from src.utils.Utilities import pickle_loader, pickle_saver, file_exist, get_project_root
from src.utils.data.DataClustering import find_cluster, clustering_data, tsne, check_data_clusterisation, \
    rebalancing_clusters
from src.utils.data.DataPreparation import add_bias_term

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

def prepare_noniid_dataset(data, pivot_label: str, data_path: str, pickle_path: str, nb_cluster: int, double_check: bool =False):

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
    X, Y = clustering_data(data, predicted_cluster, pivot_label, nb_cluster)

    if double_check:
        logging.debug("Checking data cluserization, wait until completion before seeing the plots.")
        # Checking that splitting data by cluster is valid.
        check_data_clusterisation(X, Y, nb_cluster)

    # Rebalancing cluster: the biggest one must not be more than 10times bigger than the smallest one.
    X_rebalanced, Y_rebalanced = rebalancing_clusters(X, Y)

    for y in Y_rebalanced:
        print("Nb of points:", len(y))

    return X_rebalanced, Y_rebalanced

def prepare_superconduct(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool = False):
    raw_data = pd.read_csv('{0}/dataset/superconduct/train.csv'.format(get_path_to_datasets()), sep=",")
    if raw_data.isnull().values.any():
        logging.warning("There is missing value.")
    else:
        logging.debug("No missing value. Great !")
    logging.debug("Scaling data.")
    scaled_data = scale(raw_data)

    scaled_data = pd.DataFrame(data=scaled_data, columns = raw_data.columns)
    X_data = scaled_data.loc[:, scaled_data.columns != "critical_temp"]
    Y_data = scaled_data.loc[:, scaled_data.columns == "critical_temp"]
    dim = len(X_data.columns)
    logging.debug("There is " + str(dim) + " dimensions.")

    logging.debug("Head of the dataset:")
    logging.debug(raw_data.head())

    if iid:
        X_tensor = torch.tensor(X_data.to_numpy(), dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "critical_temp", data_path + "/superconduct", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_quantum(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

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

    logging.debug("Scaling data.")
    scaled_data = scale(raw_data.loc[:, raw_data.columns != "state"])

    scaled_X = pd.DataFrame(data=scaled_data, columns=raw_data.loc[:, raw_data.columns != "state"].columns)

    # Merging dataset in one :
    scaled_data = pd.concat([scaled_X, Y_data], axis=1, sort=False)

    if iid:
        # Transforming into torch.FloatTensor
        X_tensor = torch.tensor(scaled_X.to_numpy(), dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "state", data_path + "/quantum", pickle_path, nb_devices, double_check)

    return X, Y, dim + 1 # Because we added one column for the bias

def prepare_mushroom(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):
    raw_data = pd.read_csv('{0}/dataset/mushroom/mushrooms.csv'.format(get_path_to_datasets()))

    # The data is categorial so I convert it with LabelEncoder to transfer to ordinal.
    labelencoder = LabelEncoder()
    for column in raw_data.columns:
        raw_data[column] = labelencoder.fit_transform(raw_data[column])

    # It can be seen that the column "veil-type" is 0 and not contributing to the data so I remove it.
    raw_data = raw_data.drop(["veil-type"], axis=1)
    raw_data = raw_data.replace({'class': {0: -1}})

    Y_data = raw_data.loc[:, raw_data.columns == "class"]

    scaled_data = scale(raw_data.loc[:, raw_data.columns != "class"])
    scaled_X = pd.DataFrame(data=scaled_data, columns=raw_data.loc[:, raw_data.columns != "class"].columns)

    # Merging dataset in one :
    scaled_data = pd.concat([scaled_X, Y_data], axis=1, sort=False)
    dim = len(scaled_X.columns)

    if iid:
        X_merged = torch.tensor(scaled_X.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "class", data_path + "/mushroom", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_phishing(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/phishing/phishing.txt".format(get_path_to_datasets()))

    for i in range(len(raw_Y)):
        if raw_Y[i] == 0:
            raw_Y[i] = -1

    scaled_X = scale(np.array(raw_X.todense(), dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/phishing", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_a9a(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False, test: bool = False):

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

    scaled_X = scale(np.array(raw_X, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/a9a", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_abalone(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):
    raw_data = pd.read_csv('{0}/dataset/abalone/abalone.csv'.format(get_path_to_datasets()), sep=",", header = None)

    raw_data = raw_data.rename(columns={ 0: "gender", 1: "Length", 2: "Diameter", 3: "Height", 8: "rings"})
    labelencoder = LabelEncoder()
    raw_data["gender"] = labelencoder.fit_transform(raw_data["gender"])

    Y_data = raw_data.loc[:, raw_data.columns == "rings"]

    scaled_data = scale(raw_data.loc[:, raw_data.columns != "rings"])
    scaled_X = pd.DataFrame(data=scaled_data, columns=raw_data.loc[:, raw_data.columns != "rings"].columns)

    # Merging dataset in one :
    scaled_data = pd.concat([scaled_X, Y_data], axis=1, sort=False)
    dim = len(scaled_X.columns)

    if iid:
        X_merged = torch.tensor(scaled_X.to_numpy(), dtype=torch.float64)
        Y_merged = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_merged, Y_merged, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "rings", data_path + "/abalone", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_covtype(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/covtype/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    for i in range(len(raw_Y)):
        if raw_Y[i] == 2:
            raw_Y[i] = -1

    scaled_X = scale(np.array(raw_X, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/covtype", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_madelon(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

    raw_data = pd.read_csv('{0}/dataset/madelon/madelon_train.data'.format(get_path_to_datasets()), sep=" ", header=None)
    raw_data.drop(raw_data.columns[len(raw_data.columns) - 1], axis=1, inplace=True)
    scaled_X = scale(np.array(raw_data, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    dim = len(scaled_data.columns)

    Y_data = pd.read_csv('{0}/dataset/madelon/madelon_train.labels'.format(get_path_to_datasets()), header=None)

    scaled_data["target"] = Y_data.values

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/madelon", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_covtype(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/covtype/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    for i in range(len(raw_Y)):
        if raw_Y[i] == 2:
            raw_Y[i] = -1

    scaled_X = scale(np.array(raw_X, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/covtype", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias


def prepare_gisette(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool =False):

    raw_X, raw_Y = load_svmlight_file("{0}/dataset/gisette/data".format(get_path_to_datasets()))
    raw_X = raw_X.todense()

    scaled_X = scale(np.array(raw_X, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/gisette", pickle_path, nb_devices,
                                      double_check)
    return X, Y, dim + 1  # Because we added one column for the bias

def prepare_w8a(nb_devices: int, data_path: str, pickle_path: str, iid: bool = True, double_check: bool = False, test: bool = False):

    if not test:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/w8a/w8a".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
    else:
        raw_X, raw_Y = load_svmlight_file("{0}/dataset/w8a/w8a.t".format(get_path_to_datasets()))
        raw_X = raw_X.todense()
        raw_X = np.c_[raw_X, np.zeros((len(raw_Y)))]

    scaled_X = scale(np.array(raw_X, dtype=np.float64))
    scaled_data = pd.DataFrame(data=scaled_X)
    scaled_data["target"] = raw_Y
    dim = len(scaled_data.columns) - 1

    Y_data = scaled_data.loc[:, scaled_data.columns == "target"]

    if iid:
        X_tensor = torch.tensor(scaled_X, dtype=torch.float64)
        Y_tensor = torch.tensor(Y_data.values, dtype=torch.float64)
        X, Y = prepare_dataset_by_device(X_tensor, Y_tensor, nb_devices)
    else:
        X, Y = prepare_noniid_dataset(scaled_data, "target", data_path + "/w8a", pickle_path, nb_devices, double_check)
    return X, Y, dim + 1 # Because we added one column for the bias