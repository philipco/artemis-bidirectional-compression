"""
Created by Constantin Philippenko, 10th July 2020.

Test to verify performances of implemented gradients descent.
"""

import unittest

import torch
import numpy as np

from src.machinery.GradientDescent import SGD_Descent, ArtemisDescent
from src.machinery.Parameters import Parameters, deacreasing_step_size
from src.models.CompressionModel import SQuantization
from src.models.CostModel import RMSEModel, LogisticModel, build_several_cost_model
from src.utils.Constants import generate_param
from src.utils.data.DataPreparation import build_data_linear, build_data_logistic, add_bias_term
from src.utils.runner.RunnerUtilities import single_run_descent

nb_epoch = 20
dim_test = 20
nb_devices = 10

linear_w_true = generate_param(dim_test)
linear_X, linear_Y = build_data_linear(linear_w_true, n_dimensions=dim_test,
                                       n_devices=20, with_seed=False, without_noise=False)
linear_X = add_bias_term(linear_X)

logistic_w = torch.FloatTensor([10, 10]).to(dtype=torch.float64)
logistic_X, logistic_Y = build_data_logistic(logistic_w, n_dimensions=2,
                                             n_devices=nb_devices, with_seed=False)


class PerformancesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(PerformancesTest, cls).setUpClass()

        ### RMSE ###

        # Creating cost models which will be used to computed cost/loss, gradients, L ...
        cls.linear_cost_models = build_several_cost_model(RMSEModel, linear_X, linear_Y, nb_devices)

        # Defining parameters for the performances test.
        cls.linear_params = Parameters(n_dimensions=dim_test + 1,
                                       nb_devices=nb_devices,
                                       compression_model=SQuantization(1, dim_test + 1),
                                       step_formula=deacreasing_step_size,
                                       nb_epoch=nb_epoch,
                                       use_averaging=False,
                                       cost_models=cls.linear_cost_models,
                                       stochastic=True)

        obj_min_by_N_descent = SGD_Descent(Parameters(n_dimensions=dim_test + 1,
                                                      nb_devices=nb_devices,
                                                      nb_epoch=200,
                                                      momentum=0.,
                                                      verbose=True,
                                                      cost_models=cls.linear_cost_models,
                                                      stochastic=False,
                                                      bidirectional=False
                                                      ))
        obj_min_by_N_descent.run(cls.linear_cost_models)
        cls.linear_obj = obj_min_by_N_descent.losses[-1]

        # For LOGISTIC:
        
        cls.logistic_cost_models = build_several_cost_model(LogisticModel, logistic_X, logistic_Y, nb_devices)
        
        # Defining parameters for the performances test.
        cls.logistic_params = Parameters(n_dimensions=2,
                                         nb_devices=nb_devices,
                                         compression_model=SQuantization(1, 3),
                                         step_formula=deacreasing_step_size,
                                         nb_epoch=nb_epoch,
                                         use_averaging=False,
                                         cost_models=cls.logistic_cost_models,
                                         stochastic=True)

        obj_min_by_N_descent = SGD_Descent(Parameters(n_dimensions=2,
                                                      nb_devices=nb_devices,
                                                      nb_epoch=200,
                                                      momentum=0.,
                                                      verbose=True,
                                                      cost_models=cls.logistic_cost_models,
                                                      stochastic=False,
                                                      bidirectional=False
                                                      ))
        obj_min_by_N_descent.run(cls.logistic_cost_models)
        cls.logistic_obj = obj_min_by_N_descent.losses[-1]

    def test_artemis(self):
        model_descent = single_run_descent(self.linear_cost_models, ArtemisDescent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2)
        model_descent = single_run_descent(self.logistic_cost_models, ArtemisDescent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_bi_qsgd(self):
        model_descent = single_run_descent(self.linear_cost_models, ArtemisDescent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2)
        model_descent = single_run_descent(self.logistic_cost_models, ArtemisDescent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_diana(self):
        model_descent = single_run_descent(self.linear_cost_models, SGD_Descent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2.5)
        model_descent = single_run_descent(self.logistic_cost_models, SGD_Descent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)


    def test_qsgd(self):
        model_descent = single_run_descent(self.linear_cost_models, SGD_Descent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2.5)
        model_descent = single_run_descent(self.logistic_cost_models, SGD_Descent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_vanilla(self):
        model_descent = single_run_descent(self.linear_cost_models, SGD_Descent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -3)
        model_descent = single_run_descent(self.logistic_cost_models, SGD_Descent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)


if __name__ == '__main__':
    unittest.main()
