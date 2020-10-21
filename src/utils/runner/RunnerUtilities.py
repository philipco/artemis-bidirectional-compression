"""
Created by Philippenko, 4th May 2020.

This file give two functions (for single or multiple runs) to carry out a full gradient descent and retrieve results.
"""
import gc
import time
from pathlib import Path

from src.machinery.Parameters import Parameters
from src.machinery.GradientDescent import AGradientDescent
from src.machinery.PredefinedParameters import PredefinedParameters

from src.utils.Constants import NB_EPOCH
from src.utils.runner.MultipleDescentRun import MultipleDescentRun

nb_run = 5  # Number of gradient descent before averaging.


def multiple_run_descent(predefined_parameters: PredefinedParameters, cost_models,
                         nb_epoch=NB_EPOCH,
                         quantization_param: int = 1,
                         step_formula=None,
                         use_averaging=False,
                         stochastic: bool = True,
                         streaming: bool = False,
                         batch_size: int = 1,
                         fraction_sampled_workers: float = 1.,
                         logs_file: str = None) -> MultipleDescentRun:
    """
    Run several time the same algorithm in the same conditions and gather all result in the MultipleDescentRun class.

    Args:
        predefined_parameters: predefined parameters
        X: data
        Y: labels
        nb_epoch: number of epoch for the each run
        quantization_param:
        step_formula: lambda function to compute the step size at each iteration.
        use_averaging: true if using Polyak-Rupper averaging.
        stochastic: true if running stochastic descent.

    Returns:
    """
    multiple_descent = MultipleDescentRun()
    for i in range(nb_run):
        start_time = time.time()
        params = predefined_parameters.define(n_dimensions=cost_models[0].X.shape[1],
                                              nb_devices=len(cost_models),
                                              quantization_param=quantization_param,
                                              step_formula=step_formula,
                                              nb_epoch=nb_epoch,
                                              use_averaging=use_averaging,
                                              cost_models=cost_models,
                                              stochastic=stochastic,
                                              streaming=streaming,
                                              batch_size=batch_size,
                                              fraction_sampled_workers=fraction_sampled_workers)
        model_descent = predefined_parameters.type_FL()(params)
        model_descent.run(cost_models)
        multiple_descent.append(model_descent)
        elapsed_time = time.time() - start_time

        if logs_file:
            Path("logs/").mkdir(parents=True, exist_ok=True)
            logs = open("logs/" + logs_file, "a+")
            logs.write("{0} - run {1}, final loss : {2}, memory : {3} Mbytes\n"
                       .format(predefined_parameters.name(), i, model_descent.losses[-1], model_descent.memory_info))
            logs.close()
        del model_descent
        gc.collect()
    return multiple_descent


def single_run_descent(cost_models, model: AGradientDescent, parameters: Parameters) -> AGradientDescent:
    model_descent = model(parameters)
    model_descent.run(cost_models)
    return model_descent
