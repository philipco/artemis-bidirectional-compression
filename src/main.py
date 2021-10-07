"""
Created by Philippenko, 2 January 2020

How to use Artemis implementation on synthetic data with a linear regression.
"""
from math import sqrt

from src.machinery.GradientDescent import SGD_Descent
from src.models.CostModel import RMSEModel, build_several_cost_model
from src.machinery.Parameters import Parameters
from src.machinery.PredefinedParameters import Artemis, Diana, SQuantization, VanillaSGD

from src.utils.data.DataPreparation import build_data_linear, add_bias_term
from src.utils.Constants import generate_param
from src.utils.ErrorPlotter import plot_error_dist

from tqdm import tqdm


from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents
from src.utils.runner.RunnerUtilities import multiple_run_descent

dim_notebook = 20
nb_devices = 20
nb_devices_for_the_run = nb_devices

def deacreasing_step_size(it, L, omega, N): return 1 / (L * sqrt(it))

if __name__ == '__main__':
    ### Following takes around 11 minutes. ###

    # 1) Generating data.
    w_true = generate_param(dim_notebook - 1)
    X, Y = build_data_linear(w_true, n_dimensions=dim_notebook - 1,
                             n_devices=nb_devices, with_seed=False, without_noise=False)
    X = add_bias_term(X) # Add a column of ones.

    # 2) Creating cost models which will be used to computed cost/loss, gradients, L ...
    cost_models = build_several_cost_model(RMSEModel, X, Y, nb_devices)

    # 3) Computing objective function.
    obj_min_descent = SGD_Descent(Parameters(n_dimensions=dim_notebook,
                                                  nb_devices=nb_devices,
                                                  nb_epoch=4000,
                                                  momentum=0.,
                                                  verbose=True,
                                                  cost_models=cost_models,
                                                  stochastic=False
                                                  ), None)
    obj_min_descent.run(cost_models)
    obj_min = obj_min_descent.train_losses[-1]

    # 4) Defining settings of the run.
    compression = SQuantization(level=1, dim=dim_notebook, norm=2)
    step_size = deacreasing_step_size

    # 4) Running descent for two algorithms: Vanilla SGD, Diana and Artemis.
    all_descent = {}
    for type_params in tqdm([VanillaSGD(), Diana(), Artemis()]):
        multiple_sg_descent = multiple_run_descent(type_params, cost_models=cost_models,
                                                   compression_model=compression,
                                                   use_averaging=True,
                                                   stochastic=True,
                                                   nb_epoch=40,
                                                   step_formula=step_size,
                                                   batch_size=1,
                                                   # logs_file=filename,
                                                   fraction_sampled_workers=1)
        all_descent[type_params.name()] = multiple_sg_descent
    res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)

    # 5) Plotting results.
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                    all_error=res.get_std(obj_min), x_legend="Number of passes on data")
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                    x_points=res.X_number_of_bits, x_legend="Communicated bits",
                    all_error=res.get_std(obj_min))




