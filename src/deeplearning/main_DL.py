"""
Created by Philippenko, 2th April 2021.
"""
import numpy

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnModels import *
from src.deeplearning.Train import tune_step_size, run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver

from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    with open("log.txt", 'a') as f:
        print("==== NEW RUN ====", file=f)

    dataset = "mnist"
    model = MNIST_CNN  # MNIST_CNN #MNIST_FullyConnected
    momentum = 0.9
    optimal_step_size = 0.1
    level_quantiz = 2
    batch_size = 256

    compression_by_default = SQuantization(level_quantiz)

    all_descent = {}
    nb_devices_for_the_run = 20
    for type_params in [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis(), Dore(), DoubleSqueeze()]:
        print(type_params)
        torch.cuda.empty_cache()
        params = type_params.define(cost_models=None,
                                    n_dimensions=None,
                                    nb_epoch=16,
                                    nb_devices=nb_devices_for_the_run,
                                    batch_size=batch_size,
                                    fraction_sampled_workers=1,
                                    up_compression_model=compression_by_default)

        params = cast_to_DL(params)
        params.dataset = dataset
        params.model = model
        params.log_file = "log.txt"
        params.momentum = momentum
        params.optimal_step_size = optimal_step_size
        params.print()

        with open(params.log_file, 'a') as f:
            print(type_params, file=f)
            print("Optimal step size: ", params.optimal_step_size, file=f)

        multiple_sg_descent = run_tuned_exp(params)

        all_descent[type_params.name()] = multiple_sg_descent
        res = ResultsOfSeveralDescents(all_descent, nb_devices_for_the_run)
        pickle_saver(res, "{0}_{1}_m{2}_lr{3}_s{4}_mem0.01".format(dataset, model.__name__, momentum, optimal_step_size, level_quantiz))

    res = pickle_loader("{0}_{1}_m{2}_lr{3}_s{4}_mem0.01".format(dataset, model.__name__, momentum, optimal_step_size, level_quantiz))

    # TEMP #
    iid = False
    dim_notebook = 42

    # Plotting without averaging
    plot_error_dist(res.get_loss(numpy.array(0), in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_std(numpy.array(0), in_log=True), x_legend="Number of passes on data",
                    picture_name="train_losses_{0}_m{1}_lr{2}_s{3}".format(dataset, momentum, optimal_step_size,
                                                                           level_quantiz))
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices_for_the_run,
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data", batch_size=batch_size,
                    picture_name="test_accuracies_{0}_m{1}_lr{2}_s{3}".format(dataset, momentum, optimal_step_size,
                                                                              level_quantiz))
    plot_error_dist(res.get_test_losses(in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_test_losses_std(in_log=True), x_legend="Number of passes on data",
                    picture_name="test_losses_{0}_m{1}_lr{2}_s{3}".format(dataset, momentum, optimal_step_size,
                                                                                level_quantiz))
