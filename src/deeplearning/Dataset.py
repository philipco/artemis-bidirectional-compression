"""
Created by Philippenko, 13rd May 2021.
"""
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_and_extract_archive

from src.utils.Utilities import get_project_root, create_folder_if_not_existing
from src.utils.data.RealDatasetPreparation import prepare_quantum, prepare_a9a, prepare_phishing


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

    def __init__(self, train: bool =True, iid: str = "iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False
        create_folder_if_not_existing("{0}/pickle/quantum-{1}-N20".format(root, iid))
        X_train, Y_train, dim_notebook = prepare_quantum(20, data_path="{0}/pickle/".format(root),
                                             pickle_path="{0}/pickle/quantum-{1}-N20".format(root, iid), iid=bool_iid)

        X_train = torch.cat([x for x in X_train])
        Y_train = torch.cat([y for y in Y_train])

        # for i in range(len(Y_train)):
        #     if Y_train[i] == -1:
        #         Y_train[i] = 0

        n = int(len(X_train) * 10 / 100)

        test_idx = random.sample(range(len(X_train)), n)
        train_idx = [e for e in list(range(len(X_train))) if e not in test_idx]

        X_test, Y_test = X_train[test_idx], Y_train[test_idx]
        # X_train, Y_train = X_train[train_idx], Y_train[train_idx]
        
        self.train = train
        if self.train:
            print('Total number of point:', len(X_train))
            self.data = X_train
            self.labels = Y_train
        else:
            self.data = X_test
            self.labels = Y_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index].float(),  self.labels[index].float()

class PhishingDataset(Dataset):

    def __init__(self, train=True, iid: str = "iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False
        create_folder_if_not_existing("{0}/pickle/quantum-{1}-N20".format(root, iid))
        X, Y, dim_notebook = prepare_phishing(20, data_path="{0}/pickle/".format(root),
                                             pickle_path="{0}/pickle/quantum-{1}-N20".format(root, iid),
                                             iid=bool_iid)

        X = torch.cat([x for x in X])
        Y = torch.cat([y for y in Y])

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

class A9ADataset(Dataset):

    def __init__(self, train=True, iid: str ="iid"):
        root = get_project_root()
        bool_iid = True if iid == "iid" else False

        create_folder_if_not_existing("{0}/pickle/quantum-{1}-N20".format(root, iid))
        X_train, y_train, dim_notebook = prepare_a9a(20, data_path="{0}/pickle/".format(root),
                                             pickle_path="{0}/pickle/a9a-{1}-N20".format(root, iid),
                                             iid=bool_iid, test=False)

        X_train = torch.cat([x for x in X_train])
        y_train = torch.cat([y for y in y_train])

        for i in range(len(y_train)):
            if y_train[i] == -1:
                y_train[i] = 0

        X_test, y_test, dim_notebook = prepare_a9a(20, data_path="{0}/pickle/".format(root),
                                                     pickle_path="{0}/pickle/a9a-{1}-N20".format(root, iid),
                                                     iid=bool_iid, test=True)

        X_test = torch.cat([x for x in X_test])
        y_test = torch.cat([y for y in y_test])

        for i in range(len(y_test)):
            if y_test[i] == -1:
                y_test[i] = 0

        if train:
            print('Total number of point:', len(X_train))
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.targets[index]).type(torch.LongTensor)


if __name__ == '__main__':
    dataset = QuantumDataset(train=True)
    print(len(dataset))
    print(dataset[100])
    dataset = QuantumDataset(train=False)
    print(len(dataset))
    print(dataset[100])