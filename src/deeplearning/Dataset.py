"""
Created by Philippenko, 13rd May 2021.
"""
import itertools
import os

import torch
from PIL import Image
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
        X, Y, dim_notebook = prepare_quantum(20, data_path="{0}/pickle/".format(root), pickle_path="{0}/pickle/quantum-non-iid-N20".format(root), iid=False)
        for y in Y:
            for i in range(len(y)):
                if y[i].item() == -1:
                    y[i] = 0
                else:
                    y[i] = 1

        test_data, test_labels = [], []
        eval_data, eval_labels = [], []
        last_index = 0
        split = []
        for i in range(len(X)):
            x, y = X[i], Y[i]
            n = int(len(x) * 10 / 100)
            test_data += x[:n]
            test_labels += y[:n]
            eval_data += x[n:2*n]
            eval_labels += y[n:2*n]
            X[i], Y[i] = X[i][n:], Y[i][n:]
            split.append(list(range(last_index, last_index + len(X[i]))))
            last_index += len(X[i])

        self.train = train
        if self.train:
            self.data = eval_data + list(itertools.chain.from_iterable(X[:20]))
            self.labels = eval_labels + list(itertools.chain.from_iterable(Y[:20]))
            self.ind_val = len(eval_data)
            self.split = [[s[i] + len(eval_data) for i in range(len(s))] for s in split]
        else:
            self.data = test_data
            self.labels = test_labels



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.data[index].float(), self.labels[index].type(torch.LongTensor)


if __name__ == '__main__':
    dataset = QuantumDataset(train=True)
    print(len(dataset))
    print(dataset[100])
    dataset = QuantumDataset(train=False)
    print(len(dataset))
    print(dataset[100])