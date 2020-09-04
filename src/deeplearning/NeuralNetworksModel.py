from abc import abstractmethod

import torch.nn as nn
import torch.nn.functional as F


class ANeuralNetworkModel(nn.Module):

    @abstractmethod
    def forward(self, x):
        pass

    def number_of_param(self):
        return sum(p.numel() for p in self.parameters())


class TwoLayersModel(ANeuralNetworkModel):

    def __init__(self):
        super(TwoLayersModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LinReg(ANeuralNetworkModel):

    def __init__(self):
        super(LinReg, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat


