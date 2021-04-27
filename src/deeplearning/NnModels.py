"""
Created by Philippenko, 26th April 2021.
"""

from torch import nn
import torch.nn.functional as F

import math

class SimplestNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear = nn.Linear(224,224)
        self.relu = nn.ReLU(True)
        self.out_linear = nn.Linear(224 * 224 * 3, 10)

    def forward(self, x):
        # out = self.conv(x)
        out = self.linear(x)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.out_linear(out)
        return out
