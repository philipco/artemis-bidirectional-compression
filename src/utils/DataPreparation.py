"""
Created by Philippenko, 10 January 2020.

This class generate data for Logistic and Least-Square regression
"""
from copy import deepcopy

import numpy as np
from numpy.random.mtrand import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
import torch
from math import floor

from src.utils.Constants import DIM, NB_OF_POINTS_BY_DEVICE, BIAS


def add_bias_term(X):
    """Add a bias term in the dataset.

    :param X: dataset
    :return: dataset with an additional columns of 1 at the beginning.
    """
    newX = [torch.cat((torch.ones(len(x), 1).to(dtype=torch.float64), x), 1) for x in X]
    return newX


def add_constant_columns(x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = x.shape[0]
    tx = np.c_[np.ones(num_samples), x]
    return tx


def build_data_logistic(true_model_param: torch.FloatTensor, n_samples=NB_OF_POINTS_BY_DEVICE, n_dimensions=DIM,
               n_devices: int = 1, with_seed: bool = False,
               features_corr=0.6, labels_std=0.4):
    """Build data for logistic regression.

    Args:
        true_model_param: the true parameters of the model.
        n_samples: number of sample by devices.
        n_dimensions: dimension of the problem.
        n_devices: number of devices.
        with_seed: true if we want to initialize the pseudo-random number generator.
        features_corr: correlation coefficient used to generate data points.
        labels_std: standard deviation coefficient of the noises added on labels.

    Returns:
        if more than one device, a list of pytorch tensor, otherwise a single tensor.
    """
    X, Y = [], []
    model_copy = deepcopy(true_model_param)
    for i in range(n_devices):

        # We use two different model to simulate non iid data.
        if i%2==0:
            model_copy[(i+1)%n_dimensions] *= -1
        else:
            model_copy = deepcopy(true_model_param)

        # Construction of a covariance matrix
        cov = toeplitz(features_corr ** np.arange(0, n_dimensions))

        if not with_seed:
            np.random.seed(0)

        sign = np.array([1 for j in range(n_dimensions)])
        if i%2 == 0:
            sign[i%n_dimensions] = -1

        x = torch.from_numpy(sign * multivariate_normal(np.zeros(n_dimensions), cov, size=floor(n_samples)).astype(
            dtype=np.float64))

        # Simulation of the labels
        # NB : Logistic syntethic dataset is used to show how Artemis is used in non-i.i.d. settings.
        # This is why, we don't introduce a bias here.
        y = torch.bernoulli(torch.sigmoid(x.mv(model_copy.T)))
        y[y == 0] = -1
        X.append(x)
        Y.append(y)

    if n_devices == 1:
        return X[0], Y[0]
    return X, Y


def build_data_linear(true_model_param: torch.FloatTensor, n_samples=NB_OF_POINTS_BY_DEVICE, n_dimensions=DIM,
               n_devices: int = 1, with_seed: bool = False, without_noise=False,
               features_corr=0.6, labels_std=0.4):
    """Build data for least-square regression.

    Args:
        true_model_param: the true parameters of the model.
        n_samples: number of sample by devices.
        n_dimensions: dimension of the problem.
        n_devices: number of devices.
        with_seed: true if we want to initialize the pseudo-random number generator.
        features_corr: correlation coefficient used to generate data points.
        labels_std: standard deviation coefficient of the noises added on labels.


    Returns:
        if more than one device, a list of pytorch tensor, otherwise a single tensor.
    """

    X, Y = [], []
    for i in range(n_devices):

        # Construction of a covariance matrix
        cov = toeplitz(features_corr ** np.arange(0, n_dimensions))

        if with_seed:
            np.random.seed(0)
        x = torch.from_numpy(multivariate_normal(np.zeros(n_dimensions), cov, size=floor(n_samples)).astype(dtype=np.float64))

        # Simulation of the labels
        y = x.mv(true_model_param) + BIAS

        # We add or not a noise
        if not without_noise:
            if with_seed:
                y += torch.normal(0, labels_std, size=(floor(n_samples), 1),
                                  generator=torch.manual_seed(0), dtype=torch.float64)[0]
            else:
                y += torch.normal(0, labels_std, size=(floor(n_samples), 1), dtype=torch.float64)[0]

        X.append(x)
        Y.append(y)
    if n_devices == 1:
        return X[0], Y[0]
    return X, Y
