"""
Created by Philippenko, 2 January 2020
"""
from src.machinery.GradientDescent import ArtemisDescent
from src.models.CostModel import RMSEModel
from src.machinery.Parameters import Parameters, Artemis, Diana

from src.utils.DataPreparation import build_data_linear
from src.utils.Constants import generate_param
from src.utils.ErrorPlotter import plot_error_dist

from tqdm import tqdm


from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents
from src.utils.runner.RunnerUtilities import multiple_run_descent

# How to use Artemis implementation.

nb_devices_for_the_run = 10
dim_notebook = 20
MAX_NB_DEVICES = 40


if __name__ == '__main__':
    ### Following takes around 5 minutes. ###

    # 1) Generating data.
    w_true = generate_param(dim_notebook)
    X, Y = build_data_linear(w_true, n_dimensions=dim_notebook,
                             n_devices=MAX_NB_DEVICES, with_seed=False, without_noise=False)

    # 2) Computing objective function.
    obj_min_descent = ArtemisDescent(Parameters(n_dimensions=dim_notebook,
                                                nb_devices=nb_devices_for_the_run,
                                                nb_epoch=600,
                                                quantization_param=0,
                                                momentum=0.,
                                                verbose=False,
                                                cost_model=RMSEModel(),
                                                stochastic=False,
                                                bidirectional=False
                                                ))
    obj_min_descent.set_data(X, Y)
    obj_min_descent.run()
    obj_min = obj_min_descent.losses[-1]

    # 3) Running descent for two algorithms: Diana and Artemis
    all_descent = {}
    myX = X[:nb_devices_for_the_run]
    myY = Y[:nb_devices_for_the_run]
    X_number_of_bits = []
    for type_params in tqdm([Diana(), Artemis()]):
        multiple_sg_descent = multiple_run_descent(type_params, myX, myY, nb_epoch=40)
        all_descent[type_params.name()] = multiple_sg_descent
    res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)

    # 4) Plotting results.
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_std(obj_min))
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    x_points=res.X_number_of_bits, x_legend="Communicated bits", all_error=res.get_std(obj_min))




