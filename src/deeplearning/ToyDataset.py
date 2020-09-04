# Creating the artificial dataset
import torch

from torch.utils.data import Dataset

from src.utils.Constants import DIM


class Data1d(Dataset):

    # Constructor
    def __init__(self):
        nb_points = 400
        self.x = torch.zeros(nb_points, 1)
        self.x[:, 0] = torch.arange(-1, 1, 2 / nb_points)

        self.w = torch.tensor([[1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b

        self.y = self.f + 0.001 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting the length
    def __len__(self):
        return self.len  # Instantiation of the class

class Data2d(Dataset):

    # Constructor
    def __init__(self):
        nb_points = 20
        self.x = torch.zeros(nb_points, 2)
        self.x[:, 0] = torch.arange(-1, 1, 2 / nb_points)
        self.x[:, 1] = torch.arange(-1, 1, 2 / nb_points)

        self.w = torch.tensor([[2.0], [2.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b

        self.y = self.f + 0.0001 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Getting the length
    def __len__(self):
        return self.len  # Instantiation of the class
