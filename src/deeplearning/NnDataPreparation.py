"""
Created by Philippenko, 26th April 2021.

All function required to process data before doing DL.
"""
import random

import numpy as np
from numpy.random import dirichlet
import torch
from sklearn.preprocessing import scale
import logging
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, RandomSampler

from src.deeplearning.Dataset import QuantumDataset, FEMNISTDataset, A9ADataset, PhishingDataset, MushroomDataset
from src.utils.PathDataset import get_path_to_datasets
from src.utils.PickletHandler import pickle_loader, pickle_saver
from src.utils.Utilities import file_exist
from src.utils.data.DataClustering import find_cluster, tsne


def TSNE_non_iid_split(data, data_path: str, nb_cluster: int):
    """
    Split DL dataset based on TSNE.
    """

    X = np.zeros((data.shape[0], 784))

    for i in range(data.shape[0]):
        X[i] = data[i].flatten()

    print("TSNE-based client distribution.")
    # The TSNE representation is independent of the number of devices.
    tsne_file = "{0}-tsne".format(data_path)
    if not file_exist("{0}.pkl".format(tsne_file)):
        # Running TNSE to obtain a 2D representation of data
        logging.debug("The TSNE representation ({0}) doesn't exist.".format(tsne_file))
        embedded_data = tsne(X)
        pickle_saver(embedded_data, tsne_file)

    tsne_cluster_file = "{0}/tsne-cluster".format(data_path)
    if not file_exist("{0}.pkl".format(tsne_cluster_file)):
        # Finding clusters in the TNSE.
        logging.debug("Finding non-iid clusters in the TNSE represesentation: {0}.pkl".format(tsne_file))
        embedded_data = pickle_loader("{0}".format(tsne_file))
        logging.debug("Saving found clusters : {0}.pkl".format(tsne_cluster_file))
        predicted_cluster = find_cluster(embedded_data, tsne_cluster_file, nb_cluster)
        pickle_saver(predicted_cluster, "{0}".format(tsne_cluster_file))
    return [np.array(np.where(np.array(predicted_cluster) == i)) for i in range(20)]


def non_iid_split(train_data, nb_devices):
    """Splits the training data by target values (leads to a highly non-iid data distribution)."""
    unique_values = {}
    targets = train_data.targets
    n = len(targets)
    if not torch.is_tensor(targets):
        targets = torch.Tensor(targets)
    for i in range(n):
        if targets[i].item() in unique_values:
            unique_values[targets[i].item()] = np.append(unique_values[targets[i].item()], [i])
        else:
            unique_values[targets[i].item()] = np.array([i])

    nb_points_by_clients = n // nb_devices
    matrix = (dirichlet([0.5] * 10, size=nb_devices)  * (nb_points_by_clients+2)).astype(int)# 1 line = 1 worker
    ordered_indices = sorted(unique_values.values(), key=len)
    split = []
    for i in range(nb_devices):
        indices_for_client_i = []
        for j in range(10):
            indices_by_label = ordered_indices[j]
            indices_for_client_i += random.sample(list(indices_by_label), matrix[i][j]) # Join lists
        split.append(np.array(indices_for_client_i))

    return split


def create_loaders(dataset: str, iid: str, nb_devices: int, batch_size: int, stochastic: bool, seed: int = 42):
    """Returns dataloader."""

    train_data, test_data = load_data(dataset, iid)

    train_loader_workers = dict()

    size_dataset = len(train_data)

    # preparing iterators for workers and validation set
    np.random.seed(seed)
    indices = np.arange(size_dataset)

    size_dataset = len(indices)
    size_dataset_worker = np.int(np.floor(size_dataset / nb_devices))

    if iid == "iid":
        top_ind = size_dataset_worker * nb_devices
        seq = range(size_dataset_worker, top_ind, size_dataset_worker)
        split = np.split(indices[:top_ind], seq)
    else:
        # If the dataset contains a split attribute, no need to compute a new one based on unique values.
        if hasattr(train_data, 'split'):
            split = train_data.split
        else:
            path_to_dataset = '{0}/dataset/'.format(get_path_to_datasets())
            split = TSNE_non_iid_split(train_data.data, data_path=path_to_dataset,  nb_cluster=nb_devices)

    pin_memory = True if torch.cuda.is_available() else False
    num_workers = 1  # Suggested max number of worker in my GPU's system is 1

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory = pin_memory, num_workers=4)

    b = 0
    for ind in split:
        train_loader_workers_full = DataLoader(train_data, batch_size=5000, num_workers=num_workers,
                                                  shuffle=False, pin_memory = pin_memory)
        rand_sampler = RandomSampler(Subset(train_data, ind), replacement=True)
        train_loader_workers[b] = DataLoader(Subset(train_data, ind), batch_size=batch_size, num_workers=num_workers,
                                             sampler=rand_sampler, pin_memory = pin_memory)
        b = b + 1

    return train_loader_workers, train_loader_workers_full, test_loader


def load_data(dataset: str, iid: str):
    """Loads a dataset.

    :param dataset: Name of the dataset
    :param iid: True if the dataset must not be splitted by target value
    :return: Train dataset, test dataset
    """
    path_to_dataset = '{0}/dataset/'.format(get_path_to_datasets())
    if dataset == "fake":

        transform = transforms.ToTensor()

        train_data = datasets.FakeData(size=200, transform=transform)

        test_data = datasets.FakeData(size=200, transform=transform)

    elif dataset == 'cifar10':

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_data = datasets.CIFAR10(root=path_to_dataset, train=True, download=True, transform=transform_train)

        test_data = datasets.CIFAR10(root=path_to_dataset, train=False, download=True, transform=transform_test)

    elif dataset == 'mnist':

        # Normalization see : https://stackoverflow.com/a/67233938
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(root=path_to_dataset, train=True, download=False, transform=transform)

        test_data = datasets.MNIST(root=path_to_dataset, train=False, download=False, transform=transform)

    elif dataset == "fashion_mnist":

        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download and load the training data
        train_data = datasets.FashionMNIST(path_to_dataset, download=True, train=True, transform=train_transforms)

        # Download and load the test data
        test_data = datasets.FashionMNIST(path_to_dataset, download=True, train=False, transform=val_transforms)

    elif dataset == "femnist":

        transform = transforms.Compose([transforms.ToTensor()])

        train_data = FEMNISTDataset(path_to_dataset, download=True, train=True, transform=transform)

        test_data = FEMNISTDataset(path_to_dataset, download=True, train=False, transform=transform)

    elif dataset == "a9a":

        train_data = A9ADataset(train=True, iid=iid)

        test_data = A9ADataset(train=False, iid=iid)

    elif dataset == "mushroom":

        train_data = MushroomDataset(train=True, iid=iid)

        test_data = MushroomDataset(train=False, iid=iid)

    elif dataset == "phishing":

        train_data = PhishingDataset(train=True, iid=iid)

        test_data = PhishingDataset(train=False, iid=iid)

    elif dataset == "quantum":
        train_data = QuantumDataset(train=True, iid=iid)

        test_data = QuantumDataset(train=False, iid=iid)

    return train_data, test_data
