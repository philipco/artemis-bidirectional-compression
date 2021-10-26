"""
Created by Philippenko, 2th April 2021.

Main class to run experiments in deep learning.
"""
import copy
import sys
import logging
import time

from pympler import asizeof

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnDataPreparation import create_loaders
from src.deeplearning.NonConvexSettings import *
from src.deeplearning.Train import Train, compute_L
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver, file_exist, seed_everything, \
    create_folder_if_not_existing, remove_file
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

from src.utils.runner.RunnerUtilities import create_path_and_folders, NB_RUN, choose_algo

logging.basicConfig(level=logging.INFO)


def run_experiments_in_deeplearning(dataset: str, plot_only: bool = False) -> None:
    """Runs and plots experiments for a given dataset using an appropriate neural network.

    :param dataset: Name of the dataset
    :param plot_only: True if the goal is not to rerun all experiments but only to regenerate figures.
    """
    fraction_sampled_workers = 1
    batch_size = batch_sizes[dataset]
    nb_devices = 20
    algos = sys.argv[2]
    iid = sys.argv[3]
    stochastic = True

    create_folder_if_not_existing(algos)
    log_file = algos + "/log_" + dataset + "_" + iid + ".txt"
    with open(log_file, 'a') as f:
        print("==== NEW RUN ====", file=f)

    with open(log_file, 'a') as f:
        print("stochastic -> {0}, iid -> {1}, batch_size -> {2}, norm -> {3}, s -> {4}, momentum -> {5}, model -> {6}"
            .format(stochastic, iid, batch_size, norm_quantization[dataset], quantization_levels[dataset],
                    momentums[dataset], models[dataset].__name__), file=f)

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers)

    default_up_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])
    default_down_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])

    loaders = create_loaders(dataset, iid, nb_devices, batch_size, stochastic)
    _, train_loader_workers_full, _ = loaders
    dim = next(iter(train_loader_workers_full[0]))[0].shape[1]
    if optimal_steps_size[dataset] is None:
        L = compute_L(train_loader_workers_full)
        optimal_steps_size[dataset] = 1/L
        print("Step size:", optimal_steps_size[dataset])

    exp_name = name_of_the_experiments(dataset, stochastic)
    pickle_file = "{0}/{1}".format(algos_pickle_path, exp_name)

    list_algos = choose_algo(algos, stochastic, fraction_sampled_workers)

    if not plot_only:
        if file_exist(pickle_file + ".pkl"):
            remove_file(pickle_file  + ".pkl")

        for type_params in list_algos:
            print(type_params)
            torch.cuda.empty_cache()
            params = type_params.define(cost_models=None,
                                        n_dimensions=dim,
                                        nb_epoch=300,
                                        nb_devices=nb_devices,
                                        stochastic=stochastic,
                                        batch_size=batch_size,
                                        fraction_sampled_workers=fraction_sampled_workers,
                                        up_compression_model=default_up_compression,
                                        down_compression_model=default_down_compression)

            params = cast_to_DL(params, dataset, models[dataset], optimal_steps_size[dataset], weight_decay[dataset], iid)
            params.log_file = log_file
            params.momentum = momentums[dataset]
            params.criterion = criterion[dataset]
            params.print()

            with open(params.log_file, 'a') as f:
                print(type_params, file=f)
                print("Optimal step size: ", params.optimal_step_size, file=f)

            multiple_descent = AverageOfSeveralIdenticalRun()
            seed_everything(seed=42)
            start = time.time()
            for i in range(NB_RUN):
                print('Run {:3d}/{:3d}:'.format(i + 1, NB_RUN))
                fixed_params = copy.deepcopy(params)
                try:
                    training = Train(loaders, fixed_params)
                    multiple_descent.append_from_DL(training.run_training())
                except ValueError as err:
                    print(err)
                    continue
            with open(log_file, 'a') as f:
                print("Time of the run: {:.2f}s".format(time.time() - start), file=f)

            with open(params.log_file, 'a') as f:
                print("{0} size of the multiple SG descent: {1:.2e} bits\n".format(type_params.name(),
                                                                                        asizeof.asizeof(multiple_descent)),
                           file=f)

            if file_exist(pickle_file + ".pkl"):
                res = pickle_loader(pickle_file)
                res.add_descent(multiple_descent, type_params.name(), deep_learning_run=True)
            else:
                res = ResultsOfSeveralDescents(nb_devices)
                res.add_descent(multiple_descent, type_params.name(), deep_learning_run=True)

            pickle_saver(res, pickle_file)

    # obj_min_cvx = pickle_loader("{0}/obj_min".format(pickle_path))
    obj_min = 0#pickle_loader("{0}/obj_min".format(pickle_path))

    res = pickle_loader(pickle_file)

    # obj_min = min(res.get_loss(np.array(0), in_log=False)[0])

    # print("Obj min in convex:", obj_min_cvx)
    print("Obj min in dl:", obj_min)

    # Plotting
    plot_error_dist(res.get_loss(np.array(obj_min)), res.names, all_error=res.get_std(np.array(obj_min)),
                    x_legend="Number of passes on data", ylegends="train_loss",
                    picture_name="{0}/{1}_train_losses".format(picture_path, exp_name))
    plot_error_dist(res.get_loss(np.array(obj_min)), res.names, x_points=res.X_number_of_bits, ylegends="train_loss",
                    all_error=res.get_std(np.array(obj_min)), x_legend="Communicated bits",
                    picture_name="{0}/{1}_train_losses_bits".format(picture_path, exp_name))
    plot_error_dist(res.get_test_accuracies(), res.names, ylegends="accuracy", all_error=res.get_test_accuracies_std(),
                    x_legend="Number of passes on data",
                    picture_name="{0}/{1}_test_accuracies".format(picture_path, exp_name))
    plot_error_dist(res.get_test_losses(in_log=True), res.names, all_error=res.get_test_losses_std(in_log=True),
                    x_legend="Number of passes on data", ylegends="test_loss",
                    picture_name="{0}/{1}_test_losses".format(picture_path, exp_name))

if __name__ == '__main__':

    run_experiments_in_deeplearning(sys.argv[1])


