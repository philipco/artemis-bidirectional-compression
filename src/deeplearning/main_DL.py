"""
Created by Philippenko, 2th April 2021.
"""

from tqdm import tqdm

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.Train import tune_step_size, run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver

from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents


if __name__ == '__main__':

    compression_by_default = SQuantization(2)

    # if not file_exist("obj_min.pkl"):
    #     params = Artemis().define(cost_models=None,
    #                        n_dimensions=None,
    #                        nb_epoch=5,
    #                        nb_devices=20,
    #                        batch_size=32,
    #                        fraction_sampled_workers=1,
    #                        up_compression_model=compression_by_default)
    #     params = cast_to_DL(params)
    #     pickle_saver(obj_min, "obj_min")
    #     params = tune_step_size(params)
    #     multiple_sg_descent = run_tuned_exp(params)

    all_descent = {}
    nb_devices_for_the_run = 4
    for type_params in tqdm([Qsgd(), BiQSGD()]):
        params = type_params.define(cost_models=None,
                                  n_dimensions=None,
                                  nb_epoch=5,
                                  nb_devices=nb_devices_for_the_run,
                                  batch_size=128,
                                  fraction_sampled_workers=1,
                                  up_compression_model=compression_by_default)

        params = cast_to_DL(params)

        #params = tune_step_size(params)
        params.optimal_step_size = 0.1
        multiple_sg_descent = run_tuned_exp(params)

        all_descent[type_params.name()] = multiple_sg_descent
    res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)
    pickle_saver(res, "res")

    res = pickle_loader("res")

    # TEMP #
    iid = False
    dim_notebook = 42

    # Plotting without averaging
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data\n({0})".format(iid),
                    picture_name="picture")
    plot_error_dist(res.get_test_losses(), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_test_losses_std(), x_legend="Number of passes on data\n({0})".format(iid),
                    picture_name="picture2")

