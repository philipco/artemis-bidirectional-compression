"""
Created by Philippenko, 13rd May 2021.
"""
import itertools

import torch
from torch.utils.data import Dataset

from src.utils.Utilities import get_project_root
from src.utils.data.RealDatasetPreparation import prepare_quantum


class QuantumDataset(Dataset):

    def __init__(self, train=True):
        root = get_project_root()
        X, Y, dim_notebook = prepare_quantum(21, data_path="{0}/pickle/".format(root), pickle_path=None, iid=True)
        for y in Y:
            for i in range(len(y)):
                if y[i].item() == -1:
                    y[i] = 0
                else:
                    y[i] = 1
        self.train = train
        if self.train:
            self.data = list(itertools.chain.from_iterable(X[:20]))
            self.labels = list(itertools.chain.from_iterable(Y[:20]))
        else:
            self.data = X[len(X)-1]
            self.labels = Y[len(X)-1]

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