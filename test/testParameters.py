"""
Created by Constantin Philippenko, 6th May 2020.

Test to verify that predefined parameters are correctly set up.
"""

import unittest

import torch

from src.machinery.Parameters import *
from src.machinery.PredefinedParameters import BiQSGD, Diana, Artemis, Qsgd, SGD_Descent
from src.models.CostModel import RMSEModel
from src.utils.Constants import DIM, NB_EPOCH, NB_DEVICES

X = torch.tensor([[-1,  0,  1], [1, 1, 1], [0, 0, 1]], dtype=torch.float64)
w_true = torch.tensor([1,  2, -1], dtype=torch.float64)
Y_reg = X.mv(w_true)


def step_formula_for_test(it, L, omega, N): return 1 / (L * sqrt(it))


class ParametersTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_parameters_instantiation_without_arguments(self):
        """All arguments of parameters class have default values. Check that this values are correct."""
        params = Parameters(cost_models=RMSEModel(X, Y_reg), step_formula=step_formula_for_test)

        self.assertIs(type(params.cost_models), type(RMSEModel(X, Y_reg)))
        self.assertEqual(params.federated, False)
        self.assertEqual(params.n_dimensions, DIM)
        self.assertEqual(params.nb_devices, NB_DEVICES)
        self.assertEqual(params.batch_size, 1)
        self.assertEqual(
            params.step_formula.__code__.co_code, step_formula_for_test.__code__.co_code)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.regularization_rate, 0)
        self.assertEqual(params.momentum, 0)
        self.assertIsNone(params.quantization_param)
        self.assertEqual(params.up_learning_rate, None)
        self.assertEqual(params.force_learning_rate, False)
        self.assertEqual(params.bidirectional, False)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertEqual(params.down_compress_model, True)
        self.assertEqual(params.use_down_memory, False)
        self.assertEqual(params.use_averaging, False)

    def test_StochasticSingleCompressionWithMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Diana().define(
            cost_models=RMSEModel(X, Y_reg), n_dimensions=n_dimensions, nb_devices=nb_devices,
            quantization_param=quantization_param, step_formula=step_formula_for_test)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code, step_formula_for_test.__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_models), type(RMSEModel(X, Y_reg)))
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, False)

    def test_StochasticSingleCompressionWithoutMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Qsgd().define(
            cost_models=RMSEModel(X, Y_reg), n_dimensions=n_dimensions, nb_devices=nb_devices,
            quantization_param=quantization_param, step_formula=step_formula_for_test)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code, step_formula_for_test.__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.up_learning_rate, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_models), type(RMSEModel(X, Y_reg)))
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, False)

    def test_StochasticDoubleGradientsCompressionWithMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Artemis().define(
            cost_models=RMSEModel(X, Y_reg), n_dimensions=n_dimensions, nb_devices=nb_devices,
            quantization_param=quantization_param, step_formula=step_formula_for_test)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code, step_formula_for_test.__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_models), type(RMSEModel(X, Y_reg)))
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, True)
        self.assertEqual(params.use_down_memory, False)
        self.assertEqual(params.down_compress_model, True)

    def test_StochasticDoubleGradientsCompressionWithoutMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = BiQSGD().define(
            cost_models=RMSEModel(X, Y_reg), n_dimensions=n_dimensions, nb_devices=nb_devices,
            quantization_param=quantization_param, step_formula=step_formula_for_test)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code, step_formula_for_test.__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_models), type(RMSEModel(X, Y_reg)))
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, True)
        self.assertEqual(params.use_down_memory, False)
        self.assertEqual(params.down_compress_model, True)


if __name__ == '__main__':
    unittest.main()