"""
Created by Philippenko, 2 January 2020
"""
from src.machinery.GradientDescent import SGD_Descent
from src.models.CostModel import RMSEModel, build_several_cost_model
from src.machinery.Parameters import Parameters
from src.machinery.PredefinedParameters import Artemis, Diana

from src.utils.data.DataPreparation import build_data_linear
from src.utils.Constants import generate_param
from src.utils.ErrorPlotter import plot_error_dist

from tqdm import tqdm


from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents
from src.utils.runner.RunnerUtilities import multiple_run_descent

# How to use Artemis implementation.

dim_notebook = 20
nb_devices = 2
nb_devices_for_the_run = nb_devices

if __name__ == '__main__':
    ### Following takes around 5 minutes. ###

    # 1) Generating data.
    w_true = generate_param(dim_notebook)
    X, Y = build_data_linear(w_true, n_dimensions=dim_notebook,
                             n_devices=nb_devices, with_seed=False, without_noise=False)

    # 2) Creating cost models which will be used to computed cost/loss, gradients, L ...
    cost_models = build_several_cost_model(RMSEModel, X, Y, nb_devices)

    # 3) Computing objective function.
    obj_min_descent = SGD_Descent(Parameters(n_dimensions=dim_notebook,
                                             nb_devices=nb_devices_for_the_run,
                                             nb_epoch=600,
                                             momentum=0.,
                                             verbose=True,
                                             cost_models=cost_models,
                                             stochastic=False,
                                             bidirectional=False
                                             ))
    obj_min_descent.run(cost_models)
    obj_min = obj_min_descent.losses[-1]

    # 4) Running descent for two algorithms: Diana and Artemis
    all_descent = {}
    myX = X[:nb_devices_for_the_run]
    myY = Y[:nb_devices_for_the_run]
    X_number_of_bits = []
    for type_params in tqdm([Diana(), Artemis()]):
        multiple_sg_descent = multiple_run_descent(type_params, cost_models=cost_models, nb_epoch=7)
        all_descent[type_params.name()] = multiple_sg_descent
    res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)

    # 5) Plotting results.
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.getter_std(obj_min))
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    x_points=res.X_number_of_bits, x_legend="Communicated bits", all_error=res.getter_std(obj_min))
    plot_error_dist(res.get_error_feedback(), res.names, res.nb_devices_for_the_run, dim_notebook,
                    ylegends=r"$\log_{10}(||EF_k||^2$)")




