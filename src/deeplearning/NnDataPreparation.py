"""
Created by Philippenko, 26th April 2021.
"""
from math import floor

import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, RandomSampler

from src.deeplearning.DLParameters import DLParameters
from src.deeplearning.Dataset import QuantumDataset, FEMNISTDataset, A9ADataset, PhishingDataset

def non_iid_split(train_data, nb_devices):
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

    ordered_indices = sorted(unique_values.values(), key=len)
    if len(ordered_indices) < nb_devices:
        while len(ordered_indices) != nb_devices:
            ordered_indices = sorted(ordered_indices[:-1] + [ordered_indices[-1][:floor(len(ordered_indices[-1])/2)]] \
                              + [ordered_indices[-1][floor(len(ordered_indices[-1]) / 2):]], key=len)
    if len(ordered_indices) > nb_devices:
        while len(ordered_indices) != nb_devices:
            ordered_indices = sorted([np.append(ordered_indices[0], ordered_indices[1])] + ordered_indices[2:], key=len)

    return ordered_indices

    return None


def create_loaders(parameters: DLParameters, seed: int = 42):
    train_data, test_data = load_data(parameters)

    train_loader_workers = dict()
    train_loader_workers_full = dict()

    size_dataset = len(train_data)

    # preparing iterators for workers and validation set
    np.random.seed(seed)
    indices = np.arange(size_dataset)

    n_val = np.int(np.floor(0.1 * size_dataset))
    val_data = Subset(train_data, indices=indices[:n_val])

    # indices = indices[n_val:]
    size_dataset = len(indices)
    size_dataset_worker = np.int(np.floor(size_dataset / parameters.nb_devices))
    top_ind = size_dataset_worker * parameters.nb_devices
    seq = range(size_dataset_worker, top_ind, size_dataset_worker)

    if parameters.iid == "iid":
        split = np.split(indices[:top_ind], seq)
    else:
        # If the dataset contains a split attribute, no need to compute a new one based on unique values.
        if hasattr(train_data, 'split'):
            split = train_data.split
        else:
            split = non_iid_split(train_data, parameters.nb_devices)

    test_loader = DataLoader(test_data, batch_size=parameters.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=parameters.batch_size, shuffle=False)

    b = 0
    for ind in split:
        train_loader_workers_full[b] = DataLoader(Subset(train_data, ind), batch_size=size_dataset_worker,
                                                  shuffle=True)
        rand_sampler = RandomSampler(Subset(train_data, ind), replacement=True)
        train_loader_workers[b] = DataLoader(Subset(train_data, ind), batch_size=parameters.batch_size,
                                             sampler=rand_sampler)
        b = b + 1

    if parameters.stochastic:
        return train_loader_workers, train_loader_workers_full, val_loader, test_loader
    else:
        return train_loader_workers_full, train_loader_workers_full, val_loader, test_loader


def load_data(parameters: DLParameters):
    if parameters.dataset == "fake":

        transform = transforms.ToTensor()

        train_data = datasets.FakeData(size=200, transform=transform)

        test_data = datasets.FakeData(size=200, transform=transform)

    elif parameters.dataset == 'cifar10':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_data = datasets.CIFAR10(root='../dataset/', train=True, download=True, transform=transform_train)

        test_data = datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform_test)

    elif parameters.dataset == 'mnist':

        # Normalization see : https://stackoverflow.com/a/67233938
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_data = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)

        test_data = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)

    elif parameters.dataset == "fashion_mnist":

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

        # Download and load the training data
        train_data = datasets.FashionMNIST('../dataset/', download=True, train=True, transform=transform)

        # Download and load the test data
        test_data = datasets.FashionMNIST('../dataset/', download=True, train=False, transform=transform)

    elif parameters.dataset == "femnist":

        transform = transforms.Compose([transforms.ToTensor()])

        train_data = FEMNISTDataset('../dataset/', download=True, train=True, transform=transform)

        test_data = FEMNISTDataset('../dataset/', download=True, train=False, transform=transform)

    elif parameters.dataset == "a9a":

        train_data = A9ADataset(train=True, iid=parameters.iid)

        test_data = A9ADataset(train=False, iid=parameters.iid)

    elif parameters.dataset == "phishing":

        train_data = PhishingDataset(train=True, iid=parameters.iid)

        test_data = PhishingDataset(train=False, iid=parameters.iid)

    elif parameters.dataset == "quantum":
        train_data = QuantumDataset(train=True, iid=parameters.iid)

        test_data = QuantumDataset(train=False, iid=parameters.iid)

    return train_data, test_data
