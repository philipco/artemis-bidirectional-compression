"""
Created by Philippenko, 4th May 2020.

This file give two functions (for single or multiple runs) to carry out a full gradient descent and retrieve results.
"""

import torch

from src.machinery.Parameters import Parameters
from src.machinery.GradientDescent import AGradientDescent
from src.machinery.PredefinedParameters import PredefinedParameters

from src.models.CostModel import RMSEModel

from src.utils.Constants import NB_EPOCH, NB_RUN
from src.utils.runner.MultipleDescentRun import MultipleDescentRun


def multiple_run_descent(predefined_parameters: PredefinedParameters, X, Y,
                         nb_epoch=NB_EPOCH,
                         quantization_param: int = 1,
                         step_formula=None,
                         use_averaging=False,
                         model=RMSEModel(),
                         stochastic=True,
                         batch_size=1) -> MultipleDescentRun:
    """

    Args:
        predefined_parameters: predefined parameters
        X: data
        Y: labels
        nb_epoch: number of epoch for the each run
        quantization_param:
        step_formula: lambda function to compute the step size at each iteration.
        use_averaging: true if using Polyak-Rupper averaging.
        model: cost model of the problem (e.g least-square, logistic ...).
        stochastic: true if running stochastic descent.

    Returns:
    """
    print(predefined_parameters.name())

    multiple_descent = MultipleDescentRun()
    for i in range(NB_RUN):
        params = predefined_parameters.define(n_dimensions=X[0].shape[1],
                                              nb_devices=len(X),
                                              quantization_param=quantization_param,
                                              step_formula=step_formula,
                                              nb_epoch=nb_epoch,
                                              use_averaging=use_averaging,
                                              model=model,
                                              stochastic=stochastic,
                                              batch_size=batch_size)
        model_descent = predefined_parameters.type_FL()(params)
        model_descent.set_data(X, Y)
        model_descent.run()
        multiple_descent.append(model_descent)
    return multiple_descent


def single_run_descent(X: torch.FloatTensor, Y: torch.FloatTensor,
                  model: AGradientDescent, parameters: Parameters) -> AGradientDescent:
    model_descent = model(parameters)
    model_descent.set_data(X, Y)
    model_descent.run()
    return model_descent
