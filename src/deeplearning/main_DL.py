"""
Created by Philippenko, 2th April 2021.
"""
import numpy
import logging
import torchvision.models as tm

import numpy as np

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnModels import *
from src.deeplearning.Train import tune_step_size, run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver, get_project_root, file_exist

from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents
from src.utils.runner.RunnerUtilities import choose_algo, create_path_and_folders

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)

    with open("log.txt", 'a') as f:
        print("==== NEW RUN ====", file=f)

    fraction_sampled_workers = 1
    dataset = "quantum" #"cifar10"
    model = Quantum_Linear#tm.resnet18  # MNIST_CNN #MNIST_FullyConnected
    momentum = 0.
    weight_decay = 0#5e-4
    optimal_step_size = 0.12 #0.12
    up_quantization_level = 1
    down_quantization_level = 1
    batch_size = 400
    nb_devices = 20
    algos = "mcm-vs-existing"
    list_algos = choose_algo(algos)
    iid = "non-iid"

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers)

    default_up_compression = SQuantization(up_quantization_level)
    default_down_compression = SQuantization(down_quantization_level)

    exp_name = "{0}_m{1}_lr{2}_sup{3}_sdwn{4}_b{5}_wd{6}".format(model.__name__, momentum, optimal_step_size,
                                                                 default_up_compression.level,
                                                                 default_down_compression.level, batch_size,
                                                                 weight_decay)

    if False:#not file_exist("{0}/obj_min.pkl".format(pickle_path)):
        print(pickle_path)
        params = VanillaSGD().define(cost_models=None,
                                    n_dimensions=None,
                                    nb_epoch=800,
                                    nb_devices=nb_devices,
                                    batch_size=10000,
                                    fraction_sampled_workers=1,
                                    up_compression_model=SQuantization(0))

        params = cast_to_DL(params)
        params.dataset = dataset
        params.model = model
        params.log_file = "log.txt"
        params.momentum = momentum
        params.optimal_step_size = optimal_step_size
        params.weight_decay = weight_decay
        params.momentum = 0.9

        obj_min_by_N_descent = run_tuned_exp(params)

        obj_min = np.mean(np.array(obj_min_by_N_descent.train_losses[-1]))
        pickle_saver(obj_min, "{0}/obj_min_dl".format(pickle_path))

    all_descent = {}
    for type_params in [VanillaSGD(), Diana(), Artemis(), MCM()]:#, Artemis(), BiQSGD()]:  # Qsgd(), Diana(), BiQSGD(), Artemis(), Dore(), DoubleSqueeze()]:
        print(type_params)
        torch.cuda.empty_cache()
        params = type_params.define(cost_models=None,
                                    n_dimensions=None,
                                    nb_epoch=8,
                                    nb_devices=nb_devices,
                                    batch_size=batch_size,
                                    fraction_sampled_workers=1,
                                    up_compression_model=default_up_compression,
                                    down_compression_model=default_down_compression)

        params = cast_to_DL(params)
        params.dataset = dataset
        params.model = model
        params.log_file = "log.txt"
        params.momentum = momentum
        params.weight_decay = weight_decay
        params.optimal_step_size = optimal_step_size
        params.print()

        with open(params.log_file, 'a') as f:
            print(type_params, file=f)
            print("Optimal step size: ", params.optimal_step_size, file=f)

        multiple_sg_descent = run_tuned_exp(params)

        all_descent[type_params.name()] = multiple_sg_descent
        res = ResultsOfSeveralDescents(all_descent, nb_devices)
        pickle_saver(res, "{0}/{1}".format(algos_pickle_path, exp_name))

    res = pickle_loader("{0}/{1}".format(algos_pickle_path, exp_name))
    obj_min = pickle_loader("{0}/obj_min_dl".format(pickle_path))

    # TEMP #
    iid = False

    # Plotting without averaging
    plot_error_dist(res.get_loss(np.array(0), in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_std(np.array(0), in_log=True), x_legend="Number of passes on data", ylegends="train_loss",
                    picture_name="{0}/{1}_train_losses".format(picture_path, exp_name))
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices_for_the_run, ylegends="accuracy",
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data", batch_size=batch_size,
                    picture_name="{0}/{1}_test_accuracies".format(picture_path, exp_name))
    plot_error_dist(res.get_test_losses(in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_test_losses_std(in_log=True), x_legend="Number of passes on data", ylegends="test_loss",
                    picture_name="{0}/{1}_test_losses".format(picture_path, exp_name))
