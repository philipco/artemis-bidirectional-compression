"""
Created by Philippenko, 6 January 2020.

This python file gather all constants used as default value of this implementation of Artemis.
"""

import multiprocessing as mp
import numpy as np
import torch

W_BOUND = (-7.5, 7.5)  # Bound of the plot when plotting gradient descent.
NB_DEVICES = 10  # Default number of devices.
DIM = 10  # Default dimension.
DIM_OUTPUT = 1  # Default output dimension.
NB_EPOCH = 100  # Number of epoch for one gradient descent.
NB_OF_POINTS_BY_DEVICE = 200  # Default number of points by device.
MAX_LOSS = 1e10 # maximal acceptable loss when considering that gradient descent diverged.

BIAS = 2

CORES = mp.cpu_count()


def generate_param(n_dimensions: int):
    """Simulation of model's parameters"""
    nnz = 20
    idx = np.arange(n_dimensions)
    W = torch.FloatTensor((-1) ** (idx + 1) * np.exp(-idx / 10.)).to(dtype=torch.float64)
    W[nnz:] = 0.
    return W


TRUE_MODEL_PARAM = generate_param(DIM)

DEVICE_RANGE = [1, 3, 10, 16, 20, 40]  # Range of device used in experiments
DIMENSION_RANGE = [1, 4, 10, 16, 20, 160, 320]  # Range of dimension used in experiments

step_formula = [(lambda it, L, omega, N: 40 / (2*L)),
                (lambda it, L, omega, N: 20 / (2*L)),
                (lambda it, L, omega, N: 5 / L),
                (lambda it, L, omega, N: 2 / L),
                (lambda it, L, omega, N: 1 / L),
                (lambda it, L, omega, N: 1 / (2*L)),
                (lambda it, L, omega, N: 1 / (4*L)),
                (lambda it, L, omega, N: 1 / (8*L)),
                (lambda it, L, omega, N: 1 / (16*L)),
                (lambda it, L, omega, N: 1 / (32*L))
                ]

label_step_formula = ["N/L",
                      "N/2L",
                      "5/L",
                      "2/L",
                      "$L^{-1}$",
                      "$2L^{-1}$",
                      "$4L^{-1}$",
                      "$8L^{-1}$",
                      "$16L^{-1}$",
                      "$32L^{-1}$"
                      ]


