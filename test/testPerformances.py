"""
Created by Constantin Philippenko, 10th July 2020.

Test to verify performances of implemented gradients descent.
"""

import unittest

import torch
import numpy as np

from src.machinery.GradientDescent import FL_VanillaSGD, ArtemisDescent
from src.machinery.Parameters import Parameters
from src.models.CostModel import RMSEModel, LogisticModel
from src.utils.Constants import generate_param
from src.utils.DataPreparation import build_data_linear, build_data_logistic, add_bias_term
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

        # Defining parameters for the performances test.
        cls.linear_params = Parameters(n_dimensions=dim_test + 1,
                                       nb_devices=nb_devices,
                                       quantization_param=1,
                                       step_formula=None,
                                       nb_epoch=nb_epoch,
                                       use_averaging=False,
                                       cost_model=RMSEModel(),
                                       stochastic=True)

        obj_min_by_N_descent = FL_VanillaSGD(Parameters(n_dimensions=dim_test + 1,
                                                        nb_devices=nb_devices,
                                                        nb_epoch=200,
                                                        momentum=0.,
                                                        quantization_param=0,
                                                        verbose=True,
                                                        cost_model=RMSEModel(),
                                                        stochastic=False,
                                                        bidirectional=False,
                                                        ))
        obj_min_by_N_descent.set_data(linear_X[:nb_devices], linear_Y[:nb_devices])
        obj_min_by_N_descent.run()
        cls.linear_obj = obj_min_by_N_descent.losses[-1]

        # For LOGISTIC:

        # Defining parameters for the performances test.
        cls.logistic_params = Parameters(n_dimensions=2,
                                         nb_devices=nb_devices,
                                         quantization_param=1,
                                         step_formula=None,
                                         nb_epoch=nb_epoch,
                                         use_averaging=False,
                                         cost_model=LogisticModel(),
                                         stochastic=True)

        obj_min_by_N_descent = FL_VanillaSGD(Parameters(n_dimensions=2,
                                                        nb_devices=nb_devices,
                                                        nb_epoch=200,
                                                        momentum=0.,
                                                        quantization_param=0,
                                                        verbose=True,
                                                        cost_model=LogisticModel(),
                                                        stochastic=False,
                                                        bidirectional=False,
                                                        ))
        obj_min_by_N_descent.set_data(logistic_X[:nb_devices], logistic_Y[:nb_devices])
        obj_min_by_N_descent.run()
        cls.logistic_obj = obj_min_by_N_descent.losses[-1]

    def test_artemis(self):
        model_descent = single_run_descent(linear_X, linear_Y, ArtemisDescent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2)
        model_descent = single_run_descent(logistic_X, logistic_Y, ArtemisDescent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_bi_qsgd(self):
        model_descent = single_run_descent(linear_X, linear_Y, ArtemisDescent, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2)
        model_descent = single_run_descent(logistic_X, logistic_Y, ArtemisDescent, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_diana(self):
        model_descent = single_run_descent(linear_X, linear_Y, FL_VanillaSGD, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2.5)
        model_descent = single_run_descent(logistic_X, logistic_Y, FL_VanillaSGD, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)


    def test_qsgd(self):
        model_descent = single_run_descent(linear_X, linear_Y, FL_VanillaSGD, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -2.5)
        model_descent = single_run_descent(logistic_X, logistic_Y, FL_VanillaSGD, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)

    def test_vanilla(self):
        model_descent = single_run_descent(linear_X, linear_Y, FL_VanillaSGD, self.linear_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.linear_obj), -3)
        model_descent = single_run_descent(logistic_X, logistic_Y, FL_VanillaSGD, self.logistic_params)
        self.assertLess(np.log10(model_descent.losses[-1] - self.logistic_obj), -3.5)


if __name__ == '__main__':
    unittest.main()
