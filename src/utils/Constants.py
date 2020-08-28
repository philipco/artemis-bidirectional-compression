"""
Created by Philippenko, 6 January 2020.

This python file gather all constants used as default value of this implementation of Artemis.
"""

import multiprocessing as mp
import numpy as np
import torch

W_BOUND = (-7.5, 7.5)  # Bound of the plot when plotting gradient descent.
NB_WORKERS = 10  # Default number of devices.
DIM = 10  # Default dimension.
DIM_OUTPUT = 1  # Default output dimension.
NB_EPOCH = 3  # Number of epoch for one gradient descent.
NB_OF_POINTS_BY_DEVICE = 200  # Default number of points by device.
MAX_LOSS = 1e10 # maximal acceptable loss when considering that gradient descent diverged.
LR = 1e-3
NB_RUN = 2 # Number of run for a same situation which will then be averaged.
BATCH_SIZE = 64

DEVICE_TYPE = "cpu"

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

step_formula_labels = ["4/L", "2/L", "1/L", "1/(2*L)", "1/(5*L)", "1/(2*L*(1+omega))", "1/(5*L*(1+omega))",
                       "1/(10*L*(1+omega))"]  # range of step formula labels

# range of step formula used in experiments
step_formula = [(lambda it, L, omega, N: 4 / L),
                (lambda it, L, omega, N: 2 / L),
                (lambda it, L, omega, N: 1 / L),
                (lambda it, L, omega, N: 1 / (2 * L)),
                (lambda it, L, omega, N: 1 / (5 * L)),
                (lambda it, L, omega, N: 1 / (2 * L * (1 + omega))),
                (lambda it, L, omega, N: 1 / (5 * L * (1 + omega))),
                (lambda it, L, omega, N: 1 / (10 * L * (1 + omega)))
                ]

DEVICE_RANGE = [1, 3, 10, 16, 20, 40]  # Range of device used in experiments
DIMENSION_RANGE = [1, 4, 10, 16, 20, 160, 320]  # Range of dimension used in experiments
