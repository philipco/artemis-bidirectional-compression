"""
Created by Philippenko, 13rd May 2021.
"""
import itertools
import os

import numpy as np
import torch
from PIL import Image
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import scale
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_and_extract_archive

from src.utils.Utilities import get_project_root, create_folder_if_not_existing
from src.utils.data.RealDatasetPreparation import prepare_quantum

class FEMNISTDataset(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.

    This class is taken from the code of paper Federated Learning on Non-IID Data Silos: An Experimental Study.
    https://github.com/Xtra-Computing/NIID-Bench
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        self.dataidxs = dataidxs

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        create_folder_if_not_existing(self.raw_folder)
        create_folder_if_not_existing(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)

    def __len__(self):
        return len(self.data)

class QuantumDataset(Dataset):

    def __init__(self, train=True):
        root = get_project_root()
        create_folder_if_not_existing("{0}/pickle/quantum-non-iid-N21".format(root))
        X, Y, dim_notebook = prepare_quantum(20, data_path="{0}/pickle/".format(root),
                                             pickle_path="{0}/pickle/quantum-non-iid-N20".format(root),
                                             iid=True, for_dl=True)

        X = scale(X)
        Y = np.array(Y, dtype=np.float64)

        for i in range(len(Y)):
            if Y[i] == -1:
                Y[i] = 0

        n = int(len(X) * 10 / 100)

        test_data = X[:n]
        test_labels = Y[:n]
        X, Y = X[n:], Y[n:]

        self.train = train
        if self.train:
            self.data = X
            self.labels = Y
        else:
            self.data = test_data
            self.labels = test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.labels[index]).type(torch.LongTensor)

class PhishingDataset(Dataset):

    def __init__(self, train=True):
        root = get_project_root()
        create_folder_if_not_existing("{0}/pickle/quantum-non-iid-N21".format(root))
        X, Y = load_svmlight_file("../dataset/phishing/phishing.txt")

        X = scale(np.array(X.todense(), dtype=np.float64))
        Y = np.array(Y, dtype=np.float64)

        n = int(len(X) * 10 / 100)

        test_data = X[:n]
        test_labels = Y[:n]
        X, Y = X[n:], Y[n:]

        self.train = train
        if self.train:
            self.data = X
            self.labels = Y
        else:
            self.data = test_data
            self.labels = test_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.labels[index]).type(torch.LongTensor)

class A9ADataset(Dataset):
    """ `A9A <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a>`_ Dataset.

    This class is taken from the code of Accelerated Stochastic Gradient-free and Projection-free Methods
    https://github.com/TLMichael/Acc-SZOFW/blob/main/app/datasets.py"""

    def __init__(self, path, train=True):
        self.path = path
        X_train, y_train = load_svmlight_file("../dataset/a9a/a9a.txt")
        X_test, y_test = load_svmlight_file("../dataset/a9a/a9a_test.txt")
        X_train = X_train.todense()
        X_test = X_test.todense()
        X_test = np.c_[X_test, np.zeros((len(y_test), X_train.shape[1] - np.size(X_test[0, :])))]

        X_train = np.array(X_train, dtype=np.float64)
        X_test = np.array(X_test, dtype=np.float64)
        y_train = (y_train + 1) / 2
        y_test = (y_test + 1) / 2
        y_train = np.array(y_train, dtype=np.float64)
        y_test = np.array(y_test, dtype=np.float64)

        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        x = torch.tensor(x)#, device=DEVICE)
        y = torch.tensor(y)#, device=DEVICE)
        return x.float(), y.type(torch.LongTensor)

    def __repr__(self):
        head = self.__class__.__name__ + ' ' + self.split
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.path is not None:
            body.append("File location: {}".format(self.path))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


if __name__ == '__main__':
    dataset = QuantumDataset(train=True)
    print(len(dataset))
    print(dataset[100])
    dataset = QuantumDataset(train=False)
    print(len(dataset))
    print(dataset[100])