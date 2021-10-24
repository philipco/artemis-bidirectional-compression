"""
Created by Philippenko, 8th June 2020.
"""
from src.deeplearning import DeepLearningRun
from src.machinery import GradientDescent
from src.machinery.GradientDescent import AGradientDescent

from src.utils.Utilities import compute_number_of_bits

import numpy as np


class AverageOfSeveralIdenticalRun:
    """
    This class gathers the result of multiple gradient descents performed in the same condition with the aim to later
    average the loss, and compute the variance.
    """

    def __init__(self):
        self.train_losses = []
        self.averaged_train_losses = []
        self.norm_error_feedback = []
        self.dist_to_model = []
        self.h_i_to_optimal_grad = []
        self.var_models = []
        self.theoretical_nb_bits = None
        self.parameters = None
        self.omega_c = None
        self.artificial = False

        # Required for Deep Learning
        self.test_losses = []
        self.test_accuracies = []

    def append(self, new_descent: AGradientDescent):
        self.parameters = new_descent.parameters
        if self.theoretical_nb_bits is None:
            compress_model = True if new_descent.get_name() == "DwnComprModel" else False
            self.theoretical_nb_bits = compute_number_of_bits(new_descent.parameters, len(new_descent.train_losses), compress_model)
            self.omega_c = new_descent.parameters.up_compression_model.omega_c

        self.train_losses.append(new_descent.train_losses)
        self.averaged_train_losses.append(new_descent.averaged_train_losses)
        self.norm_error_feedback.append(new_descent.norm_error_feedback)
        self.dist_to_model.append(new_descent.dist_to_model)
        self.h_i_to_optimal_grad.append(new_descent.h_i_to_optimal_grad)
        self.var_models.append(new_descent.var_models)

    def append_from_DL(self, new_run: DeepLearningRun):
        self.parameters = new_run.parameters
        self.train_losses.append(new_run.train_losses)
        self.test_losses.append(new_run.test_losses)
        self.test_accuracies.append(new_run.test_accuracies)

    def append_list(self, my_list, my_list_averaged, my_list_norm_ef, my_list_dist_model, my_list_h_i_to_optimal_grad,
                    my_list_var_models):
        """Used when running experiments with different step size/compression.

        :param my_list:
        :param my_list_averaged:
        :param my_list_norm_ef:
        :return:
        """
        number_points = len(my_list[0])
        for i in range(number_points):
            loss, loss_avg, norm_ef, dist_model, h_i_to_optimal_grad, var_models = [], [], [], [], [], []
            for list, list_avg, list_ef, list_dist, list_h_i, list_var_models in \
                    zip(my_list, my_list_averaged, my_list_norm_ef, my_list_dist_model, my_list_h_i_to_optimal_grad,
                        my_list_var_models):
                loss.append(list[i])
                loss_avg.append(list_avg[i])
                norm_ef.append(list_ef[i])
                dist_model.append(list_dist[i])
                h_i_to_optimal_grad.append(list_h_i[i])
                var_models.append(list_var_models[i])
            self.train_losses.append(np.array(loss))
            self.averaged_train_losses.append(np.array(loss_avg))
            self.norm_error_feedback.append(np.array(norm_ef))
            self.dist_to_model.append(np.array(dist_model))
            self.h_i_to_optimal_grad.append(np.array(h_i_to_optimal_grad))
            self.var_models.append(np.array(var_models))
        self.artificial = True