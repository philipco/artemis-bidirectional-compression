"""
Created by Philippenko, 2th April 2021.
"""
import numpy
from tqdm import tqdm

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnModels import *
from src.deeplearning.Train import tune_step_size, run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver

from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)
    compression_by_default = SQuantization(1)

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

    with open("log.txt", 'a') as f:
        print("==== NEW RUN ====", file=f)

    all_descent = {}
    nb_devices_for_the_run = 8
    for type_params in [VanillaSGD(), BiQSGD(), Artemis()]:
        print(type_params)
        params = type_params.define(cost_models=None,
                                  n_dimensions=None,
                                  nb_epoch=2,
                                  nb_devices=nb_devices_for_the_run,
                                  batch_size=64,
                                  fraction_sampled_workers=1,
                                  up_compression_model=compression_by_default)

        params = cast_to_DL(params)
        params.dataset = "mnist"
        params.model = MNIST_CNN() #SimplestNetwork() #resnet18()
        params.log_file = "log.txt"
        params.up_learning_rate = 0.01
        params.momentum = 0.9

        # with open(params.log_file, 'a') as f:
        #     print("Starting tuning step size.", file=f)
        #
        # params = tune_step_size(params)
        # if params.optimal_step_size == 0.:

        params.optimal_step_size = 0.1

        with open(params.log_file, 'a') as f:
            print(type_params, file=f)
            print("Optimal step size: ", params.optimal_step_size, file=f)

        multiple_sg_descent = run_tuned_exp(params)

        all_descent[type_params.name()] = multiple_sg_descent
    res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)
    pickle_saver(res, params.dataset)

    res = pickle_loader(params.dataset)

    # TEMP #
    iid = False
    dim_notebook = 42

    # Plotting without averaging
    plot_error_dist(res.get_loss(numpy.array(0), in_log=False), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_std(numpy.array(0), in_log=False), x_legend="Number of passes on data",
                    picture_name="train_losses")
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data",
                    picture_name="test_accuracies")
    plot_error_dist(res.get_test_losses(in_log=False), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_test_losses_std(in_log=False), x_legend="Number of passes on data",
                    picture_name="test_losses")

