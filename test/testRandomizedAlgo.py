"""
Created by Constantin Philippenko, 19th November 2020.

Test to verify that Randomization is correct.
"""
import random
import unittest
import torch
import numpy as np

from src.machinery.GradientUpdateMethod import ArtemisUpdate, DownCompressModelUpdate
from src.machinery.LocalUpdate import LocalArtemisUpdate
from src.machinery.Parameters import *
from src.machinery.PredefinedParameters import *
from src.machinery.Worker import Worker
from src.models.CostModel import RMSEModel, build_several_cost_model

# We have two workers, each having 3 samples in dimension 2.

nb_samples = 3
dim = 2

zero_tensor = torch.zeros(dim, dtype=np.float)
ones_tensor = torch.ones(dim, dtype=np.float)

X1 = torch.tensor([[1,  1], [2, 2], [0, 1]], dtype=torch.float64)
w_true = torch.tensor([1,  1], dtype=torch.float64)
Y_reg = X1.mv(w_true)

w = torch.tensor([0.1,  0.1], dtype=torch.float64)

X = [X1, X1]
Y = [Y_reg, Y_reg]
number_of_device = 2

def constant_step_size(it, L, omega, N): return 1 / L


class TestRandomizedAlgo(unittest.TestCase):
    """ Doesn't work. Not update the API in this test."""

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestRandomizedAlgo, cls).setUpClass()
        cls.cost_models = build_several_cost_model(RMSEModel, X, Y, number_of_device)
        cls.params = RandMCM().define(n_dimensions=dim,
                                      nb_devices=number_of_device,
                                      up_compression_model=SQuantization(1, dim),
                                      down_compression_model=SQuantization(1, dim),
                                      nb_epoch=1,
                                      cost_models=cls.cost_models,
                                      step_formula=constant_step_size)
        cls.params.down_learning_rate = 1 / cls.params.down_compression_model.omega_c
        cls.params.up_learning_rate = 1
        cls.workers = [Worker(i, cls.params, LocalArtemisUpdate) for i in range(number_of_device)]


    def test_build_randomized_omega(self):
        artemis_update = ArtemisUpdate(self.params, self.workers)
        artemis_update.workers_sub_set = self.workers
        artemis_update.value_to_compress = torch.FloatTensor([i for i in range(0, 100, 10)])
        # We initilize omega_k with two values (as if we are at iteration 2)
        artemis_update.omega_k = [[torch.FloatTensor([i for i in range(0, 100, 10)]),
                                   torch.FloatTensor([i for i in range(10)])],
                                  [torch.FloatTensor([i for i in range(0, 20, 2)]),
                                   torch.FloatTensor([i for i in range(10)])]
                                  ]
        nb_try = 1
        # We want to check that we have two different quantization of the value to compress.
        # But in quantization there is some randomness, and thus vectors can some time be identical.
        # We carry out five try, it after that there are still equal we consider that it is uncorrect.
        artemis_update.build_randomized_omega(self.cost_models)
        self.assertEqual(len(artemis_update.omega), 2,
                         "The number of compressed value kept on central server must be equal to 2.")
        while (nb_try < 5 and torch.all(artemis_update.omega[0].eq(artemis_update.omega[1]))):
            artemis_update.build_randomized_omega(self.cost_models)
            nb_try += 1
        self.assertTrue(nb_try < 5, "After 5 try, the two different quantizations are still identical.")
        self.assertTrue(len(artemis_update.omega_k) == 3)

    def test_update_randomized_model(self):
        artemis_update = ArtemisUpdate(self.params, self.workers)
        artemis_update.workers_sub_set = [(self.workers[i], self.cost_models[i]) for i in range(self.params.nb_devices)]
        artemis_update.H = torch.FloatTensor([-1 for i in range(10)])
        artemis_update.omega = [torch.FloatTensor([i for i in range(0, 100, 10)]),
                                   torch.FloatTensor([i for i in range(0,20, 2)])]
        # Without momentum, should have no impact.
        artemis_update.v = torch.FloatTensor([1 for i in range(10)])
        artemis_update.update_randomized_model()
        print(artemis_update.omega)
        # Valid with the method of averaging
        self.assertTrue(torch.all(artemis_update.v.eq(torch.FloatTensor([6*i - 1 for i in range(10)]))))

    def test_send_back_global_informations_and_update(self):
        artemis_update = ArtemisUpdate(self.params, self.workers)
        self.workers[0].idx_last_update = 1
        self.workers[1].idx_last_update = 1
        artemis_update.workers_sub_set = [(self.workers[i], self.cost_models[i]) for i in range(self.params.nb_devices)]
        artemis_update.omega_k = [[torch.FloatTensor([0, 50]),
                                   torch.FloatTensor([0,10])],
                                  [torch.FloatTensor([2,4]),
                                   torch.FloatTensor([10,20])]
                                  ]
        nb_try = 1
        artemis_update.step = 1/10
        artemis_update.send_back_global_informations_and_update(self.cost_models)
        while (nb_try < 5 and torch.all(artemis_update.workers[0].local_update.model_param
                                                .eq(artemis_update.workers[1].local_update.model_param))):
            self.workers[0].idx_last_update = 1
            self.workers[1].idx_last_update = 1
            artemis_update.send_back_global_informations_and_update(self.cost_models)
            nb_try += 1
        self.assertFalse(torch.all(artemis_update.workers[0].local_update.model_param
                                   .eq(artemis_update.workers[1].local_update.model_param)),
                         "The models on workers are expected to be different.")
        self.assertTrue(self.workers[0].idx_last_update == 2 and self.workers[1].idx_last_update == 2,
                        "Index of last participation of each worker should be updated to 2")


    def test_mechanism_of_randomization(self):

        rmcm_update = DownCompressModelUpdate(self.params, self.workers)
        # First iteration
        global_model_param = rmcm_update.compute(w, self.cost_models, 1, 1)
        self.assertEqual(len(rmcm_update.all_delta_i), number_of_device,
                         "The number of received compressed gradients on the central server should be equal to two.")
        self.assertEqual(len(rmcm_update.omega), number_of_device,
                         "The number of downlink compression kept on the central server should be equal to two. ")
        # To facilitate the assertion on randomization, we set new values of delta_i
        rmcm_update.all_delta_i = [torch.FloatTensor(10,0), torch.FloatTensor(0,1)]
        self.assertTrue(torch.all(global_model_param.eq(w - rmcm_update.step * rmcm_update.g)),
                         "The model param is not correct.")
        self.assertIsNone(rmcm_update.workers[0].local_update.model_param,
                         "The models on workers should not have been yet updated."
                         "They are updated at the begining of each iteration (but not at the first one, "
                         "as there is nothing to update")

