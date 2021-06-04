"""
Created by Philippenko, 2th April 2021.
"""
import sys
import logging

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnModels import *
from src.deeplearning.Train import tune_step_size, run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver, get_project_root, file_exist

from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents
from src.utils.runner.RunnerUtilities import choose_algo, create_path_and_folders

logging.basicConfig(level=logging.INFO)

models = {"cifar10": ResNet18, "mnist": MNIST_CNN, "quantum": Quantum_Linear}
momentums = {"cifar10": 0.9, "mnist": 0, "quantum": 0}
optimal_steps_size = {"cifar10": 0.1, "mnist": 0.1, "quantum": 0.12}
quantization_levels= {"cifar10": 4, "mnist": 1, "quantum": 1}
norm_quantization = {"cifar10": np.inf, "mnist": np.inf, "quantum": np.inf}
weight_decay = {"cifar10": 5e-4, "mnist": 0, "quantum": 0}

def run_experiments_in_deeplearning(dataset: str):

    with open("log.txt", 'a') as f:
        print("==== NEW RUN ====", file=f)

    fraction_sampled_workers = 1
    batch_size = 128
    nb_devices = 20
    algos = "mcm-vs-existing"
    iid = "non-iid"

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers)

    default_up_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])
    default_down_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])

    exp_name = "{0}_m{1}_lr{2}_sup{3}_sdwn{4}_b{5}_wd{6}".format(models[dataset].__name__, momentums[dataset],
                                                                 optimal_steps_size[dataset],
                                                                 default_up_compression.level,
                                                                 default_down_compression.level, batch_size,
                                                                 weight_decay[dataset])

    all_descent = {}
    for type_params in [VanillaSGD(), Diana(), Artemis(), MCM()]:
        print(type_params)
        torch.cuda.empty_cache()
        params = type_params.define(cost_models=None,
                                    n_dimensions=None,
                                    nb_epoch=200,
                                    nb_devices=nb_devices,
                                    batch_size=batch_size,
                                    fraction_sampled_workers=1,
                                    up_compression_model=default_up_compression,
                                    down_compression_model=default_down_compression)

        params = cast_to_DL(params, dataset, models[dataset], optimal_steps_size[dataset], weight_decay[dataset])
        params.log_file = "log.txt"
        params.momentum = momentums[dataset]
        params.print()

        with open(params.log_file, 'a') as f:
            print(type_params, file=f)
            print("Optimal step size: ", params.optimal_step_size, file=f)

        multiple_sg_descent = run_tuned_exp(params)

        all_descent[type_params.name()] = multiple_sg_descent

        res = ResultsOfSeveralDescents(all_descent, nb_devices)
        # res.add_descent(multiple_sg_descent, type_params.name())
        pickle_saver(res, "{0}/{1}".format(algos_pickle_path, exp_name))

    res = pickle_loader("{0}/{1}".format(algos_pickle_path, exp_name))

    # Plotting without averaging
    plot_error_dist(res.get_loss(np.array(0), in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_std(np.array(0), in_log=True), x_legend="Number of passes on data", ylegends="train_loss",
                    picture_name="{0}/{1}_train_losses".format(picture_path, exp_name))
    plot_error_dist(res.get_loss(np.array(0), in_log=True), res.names, res.nb_devices_for_the_run,
                    batch_size=batch_size, x_points=res.X_number_of_bits, ylegends="train_loss",
                    all_error=res.get_std(np.array(0), in_log=True), x_legend="Communicated bits",
                    picture_name="{0}/{1}_train_losses_bits".format(picture_path, exp_name))
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices_for_the_run, ylegends="accuracy",
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data", batch_size=batch_size,
                    picture_name="{0}/{1}_test_accuracies".format(picture_path, exp_name))
    plot_error_dist(res.get_test_losses(in_log=True), res.names, res.nb_devices_for_the_run, batch_size=batch_size,
                    all_error=res.get_test_losses_std(in_log=True), x_legend="Number of passes on data", ylegends="test_loss",
                    picture_name="{0}/{1}_test_losses".format(picture_path, exp_name))

if __name__ == '__main__':

    run_experiments_in_deeplearning(sys.argv[1])


