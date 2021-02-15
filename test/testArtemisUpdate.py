"""
Created by Constantin Philippenko, 7th May 2020.

Test to verify that the Artemis is correct.

Warnings: Due to quantization, self.g may be equal to zero_tensor which will make the tests to fails thinking
that self.g has not be updated. Just relaunch the test.
"""

import unittest
import torch
import numpy as np

from src.machinery.GradientUpdateMethod import ArtemisUpdate
from src.machinery.Parameters import *
from src.machinery.PredefinedParameters import *
from src.machinery.Worker import Worker

nb_samples = 5

zero_tensor = torch.zeros(DIM, dtype=np.float)
ones_tensor = torch.ones(DIM, dtype=np.float)

x = torch.ones(nb_samples, DIM, dtype=np.float)
y = torch.ones(nb_samples, dtype=np.float)
w = 2 * ones_tensor


class TestArtemisUpdate(unittest.TestCase):
    """ Doesn't work. Not update the API in this test."""

    def setUp(self):
        pass

    def test_initialization(self):
        params = Parameters()
        workers = [Worker(0, params)]
        zero_tensor = torch.zeros(params.n_dimensions, dtype=np.float)
        update = ArtemisUpdate(params, workers)
        self.assertTrue(torch.equal(update.g, zero_tensor))
        self.assertTrue(torch.equal(update.h, zero_tensor))
        self.assertTrue(torch.equal(update.v, zero_tensor))
        self.assertTrue(torch.equal(update.H, zero_tensor))
        self.assertTrue(torch.equal(update.value_to_compress, zero_tensor))

    def test_QSGD(self):
        params = Qsgd().define(n_dimensions=DIM, nb_devices=1, quantization_param=10)
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        update.compute(w, 2, 2)
        # Check that gradients have been updated.
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        # Checking that no memory have been updated.
        self.assertTrue(torch.equal(update.h, zero_tensor))
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Checking that for the return nothing has been quantized.
        self.assertTrue(torch.equal(update.value_to_compress, zero_tensor))

    def test_Diana(self):
        params = Diana().define(n_dimensions=DIM, nb_devices=1, quantization_param=10)
        params.up_learning_rate = 0.5
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        new_model_param = update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        self.assertFalse(torch.equal(update.h, zero_tensor))
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Checking that for the return nothing has been quantized.
        # there is a pb, with this test. Pass if ran with Artmis settings.
        self.assertTrue(torch.equal(update.value_to_compress, zero_tensor))

    def test_BiQsgd(self):
        params = BiQSGD().define(n_dimensions=DIM, nb_devices=1, quantization_param=10)
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        # Checking that no memory have been updated.
        self.assertTrue(torch.equal(update.h, zero_tensor))
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Check that l has been updated.
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Check that correct value has been compressed
        self.assertTrue(torch.equal(update.value_to_compress, update.g))

    def test_Artemis(self):
        params = Artemis().define(n_dimensions=DIM, nb_devices=1, quantization_param=10)
        params.up_learning_rate = 0.5
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        self.assertFalse(torch.equal(update.h, zero_tensor))
        # Check that l has been updated.
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Check that correct value has been compressed
        self.assertTrue(torch.equal(update.value_to_compress, update.g))

    def test_doubleGRADIENTcompression_WITH_additional_memory(self):
        params = DoreVariant().define(n_dimensions=DIM, nb_devices=1, quantization_param=10)
        params.up_learning_rate = 0.5
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        artificial_l = ones_tensor.clone().detach()
        # We artificially set different memory to check that it has impact on update computation.
        update.H = artificial_l.clone().detach()
        update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        self.assertFalse(torch.equal(update.h, zero_tensor))
        # Check that l has been updated.
        self.assertFalse(torch.equal(update.H, artificial_l))
        # Check that correct value has been compressed
        self.assertTrue(torch.equal(update.value_to_compress, update.g - artificial_l))

    def test_doubleMODELcompression_without_memory(self):
        params = ModelCompr().define(n_dimensions=DIM, nb_devices=1,
                                                                            quantization_param=10)
        params.up_learning_rate = 0.5
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        new_w = update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        self.assertFalse(torch.equal(update.h, zero_tensor))
        # Check that l has been updated.
        self.assertTrue(torch.equal(update.H, zero_tensor))
        # Check that correct value has been compressed
        self.assertTrue(torch.equal(update.value_to_compress, new_w))

    def test_doubleMODELcompression_WITH_memory(self):
        params = MCM().define(n_dimensions=DIM, nb_devices=1,
                                                                         quantization_param=10)
        params.up_learning_rate = 0.5
        workers = [Worker(0, params)]
        workers[0].set_data(x, y)
        workers[0].cost_model.L = workers[0].cost_model.local_L
        update = ArtemisUpdate(params, workers)
        artificial_l = ones_tensor.clone().detach()
        update.H = artificial_l.clone().detach()
        new_w = update.compute(w, 2, 2)
        # Check that gradients have been updated.
        self.assertFalse(torch.equal(update.g, zero_tensor))
        self.assertFalse(torch.equal(update.v, zero_tensor))
        self.assertFalse(torch.equal(update.h, zero_tensor))
        # Check that l has been updated.
        self.assertFalse(torch.equal(update.H, artificial_l))
        # Check that correct value has been compressed
        self.assertTrue(torch.equal(update.value_to_compress, new_w - artificial_l))