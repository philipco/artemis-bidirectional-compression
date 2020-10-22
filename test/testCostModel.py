"""
Created by Constantin Philippenko, 7th May 2020.

Test to verify that the cost model are correct.
"""

import unittest
import torch
import numpy as np

from src.machinery.Parameters import *
from src.models.CostModel import LogisticModel, RMSEModel
from src.models.RegularizationModel import NoRegularization

nb_samples = 5

zero_tensor = torch.zeros(DIM, dtype=np.float)
ones_tensor = torch.ones(DIM, dtype=np.float)

X = torch.tensor([[-1,  0,  1], [1, 1, 1], [0, 0, 1]], dtype=torch.float64)
w_true = torch.tensor([1,  2, -1], dtype=torch.float64)
Y_reg = X.mv(w_true)
Y_log = torch.tensor([1,  1, -1], dtype=torch.float64)
w = torch.tensor([1,  2, -1], dtype=torch.float64)


class TestRMSECostModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_rmse_set_data(self):
        cost_model = RMSEModel(X, Y_reg)
        self.assertIs(type(cost_model), RMSEModel)
        self.assertIs(type(cost_model.regularization), NoRegularization)
        self.assertTrue(torch.equal(X, cost_model.X))
        self.assertTrue(torch.equal(Y_reg, cost_model.Y))
        print(cost_model.local_L)
        self.assertEqual(cost_model.local_L, 2 * sqrt(18) / 3)


class TestLogisticCostModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_logistic_set_data(self):
        cost_model = LogisticModel(X, Y_log)
        self.assertIs(type(cost_model), LogisticModel)
        self.assertIs(type(cost_model.regularization), NoRegularization)
        self.assertTrue(torch.equal(X, cost_model.X))
        self.assertTrue(torch.equal(Y_log, cost_model.Y))
        self.assertEqual(cost_model.local_L, sqrt(18)/12)
