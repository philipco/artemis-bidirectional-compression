"""
Created by Constantin Philippenko, 6th May 2020.

Test to verify that predefined parameters are correctly set up.
"""

import unittest
from src.machinery.Parameters import *
from src.utils.Constants import DIM, NB_EPOCH, NB_WORKERS

class ParametersTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_parameters_instantiation_without_arguments(self):
        """All arguments of parameters class have default values. Check that this values are correct."""
        params = Parameters()

        self.assertIs(type(params.cost_model), RMSEModel)
        self.assertEqual(params.federated, False)
        self.assertEqual(params.n_dimensions, DIM)
        self.assertEqual(params.nb_devices, NB_WORKERS)
        self.assertEqual(params.batch_size, 1)
        self.assertEqual(params.step_formula.__code__.co_code, (lambda it, L, omega, N :1 / (L * sqrt(it))).__code__.co_code)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.regularization_rate, 0)
        self.assertEqual(params.momentum, 0)
        self.assertIsNone(params.quantization_param)
        self.assertEqual(params.learning_rate, None)
        self.assertEqual(params.force_learning_rate, False)
        self.assertEqual(params.bidirectional, False)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertEqual(params.compress_gradients, True)
        self.assertEqual(params.double_use_memory, False)
        self.assertEqual(params.use_averaging, False)

    def test_StochasticSingleCompressionWithMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Diana().define(n_dimensions, nb_devices, quantization_param)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code,
                         (lambda it, L, omega, N: 1 / (L * sqrt(it))).__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_model), RMSEModel)
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, False)

    def test_StochasticSingleCompressionWithoutMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Qsgd().define(n_dimensions, nb_devices, quantization_param)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code,
                         (lambda it, L, omega, N: 1 / (L * sqrt(it))).__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.learning_rate, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_model), RMSEModel)
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, False)

    def test_StochasticDoubleGradientsCompressionWithMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = Artemis().define(n_dimensions, nb_devices, quantization_param)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code,
                         (lambda it, L, omega, N: 1 / (L * sqrt(it))).__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_model), RMSEModel)
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, True)
        self.assertEqual(params.double_use_memory, False)
        self.assertEqual(params.compress_gradients, True)

    def test_StochasticDoubleGradientsCompressionWithoutMemory_iscorrect(self):
        n_dimensions, nb_devices, quantization_param = 10, 10, 1
        params = BiQSGD().define(n_dimensions, nb_devices, quantization_param)
        self.assertEqual(params.n_dimensions, n_dimensions)
        self.assertEqual(params.nb_devices, nb_devices)
        self.assertEqual(params.nb_epoch, NB_EPOCH)
        self.assertEqual(params.step_formula.__code__.co_code,
                         (lambda it, L, omega, N: 1 / (L * sqrt(it))).__code__.co_code)
        self.assertEqual(params.quantization_param, quantization_param)
        self.assertEqual(params.momentum, 0)
        self.assertEqual(params.verbose, False)
        self.assertEqual(params.stochastic, True)
        self.assertIs(type(params.cost_model), RMSEModel)
        self.assertEqual(params.use_averaging, False)
        self.assertEqual(params.bidirectional, True)
        self.assertEqual(params.double_use_memory, False)
        self.assertEqual(params.compress_gradients, True)


if __name__ == '__main__':
    unittest.main()