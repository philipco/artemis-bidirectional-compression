"""
Created by Philippenko, 13rd May 2021.
"""
import itertools

import torch
from torch.utils.data import Dataset

from src.utils.Utilities import get_project_root, create_folder_if_not_existing
from src.utils.data.RealDatasetPreparation import prepare_quantum


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