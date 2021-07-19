"""
Created by Philippenko, 2th April 2021.
"""
import copy
import sys
import logging

from src.deeplearning.DLParameters import cast_to_DL
from src.deeplearning.NnDataPreparation import create_loaders
from src.deeplearning.NnModels import *
from src.deeplearning.Train import run_tuned_exp
from src.machinery.PredefinedParameters import *
from src.utils.ErrorPlotter import plot_error_dist
from src.utils.Utilities import pickle_loader, pickle_saver, file_exist, seed_everything
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

from src.utils.runner.RunnerUtilities import create_path_and_folders, NB_RUN

logging.basicConfig(level=logging.INFO)


batch_sizes = {"cifar10": 128, "mnist": 128, "fashion_mnist": 128, "femnist": 128,
          "a9a": 50, "phishing": 50, "quantum": 400}
models = {"cifar10": ResNet18, "mnist": MNIST_Linear, "fashion_mnist": MNIST_Linear, "femnist": MNIST_Linear,
          "a9a": A9A_Linear, "phishing": Phishing_Linear, "quantum": Quantum_Linear}
momentums = {"cifar10": 0.9, "mnist": 0, "fashion_mnist": 0, "femnist": 0, "a9a": 0, "phishing": 0, "quantum": 0}
optimal_steps_size = {"cifar10": 0.1, "mnist": 0.1, "fashion_mnist": 0.1, "femnist": 0.1, "a9a":0.2593,
                      "phishing": 0.2141, "quantum": 0.2863}
quantization_levels= {"cifar10": 4, "mnist": 8, "fashion_mnist": 1, "femnist": 1, "a9a":1, "phishing": 1, "quantum": 1}
norm_quantization = {"cifar10": np.inf, "mnist": 2, "fashion_mnist": np.inf, "femnist": np.inf, "a9a": 2, "phishing": 2,
                     "quantum": 2}
weight_decay = {"cifar10": 5e-4, "mnist": 0, "fashion_mnist": 0, "femnist": 0, "a9a":0, "phishing": 0, "quantum": 0}
criterion = {"cifar10": nn.CrossEntropyLoss(), "mnist": nn.CrossEntropyLoss(), "fashion_mnist": nn.CrossEntropyLoss(),
             "femnist": nn.CrossEntropyLoss(), "a9a":  torch.nn.BCELoss(reduction='mean'),
             "phishing":  torch.nn.BCELoss(reduction='mean'), "quantum": torch.nn.BCELoss(reduction='mean')}

def run_experiments_in_deeplearning(dataset: str, plot_only: bool = False):

    log_file = "log_" + dataset + ".txt"
    with open(log_file, 'a') as f:
        print("==== NEW RUN ====", file=f)

    fraction_sampled_workers = 1
    batch_size = batch_sizes[dataset]
    nb_devices = 20
    algos = "mcm-vs-existing"
    iid = "iid"
    stochastic = True

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers)

    default_up_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])
    default_down_compression = SQuantization(quantization_levels[dataset], norm=norm_quantization[dataset])

    exp_name = "{0}_m{1}_lr{2}_sup{3}_sdwn{4}_b{5}_wd{6}".format(models[dataset].__name__, momentums[dataset],
                                                                 optimal_steps_size[dataset],
                                                                 default_up_compression.level,
                                                                 default_down_compression.level, batch_size,
                                                                 weight_decay[dataset])

    if not stochastic:
        exp_name += "-full"

    if not file_exist("{0}/obj_min_dl.pkl".format(pickle_path)):
        with open(log_file, 'a') as f:
            print("==> Computing objective loss.", file=f)
        params = VanillaSGD().define(cost_models=None,
                                     n_dimensions=None,
                                     stochastic=False,
                                     nb_epoch=10000,
                                     nb_devices=nb_devices,
                                     batch_size=batch_size,
                                     fraction_sampled_workers=1,
                                     up_compression_model=SQuantization(0, norm=norm_quantization[dataset]),
                                     down_compression_model=SQuantization(0, norm=norm_quantization[dataset]))

        params = cast_to_DL(params, dataset, models[dataset], optimal_steps_size[dataset], weight_decay[dataset], iid)
        params.log_file = log_file
        params.momentum = momentums[dataset]
        params.criterion = criterion[dataset]

        obj_min = run_tuned_exp(params, create_loaders(params)).train_losses[-1]
        pickle_saver(obj_min, "{0}/obj_min_dl".format(pickle_path))

    if not plot_only:
        all_descent = {}
        # res = pickle_loader("{0}/{1}".format(algos_pickle_path, exp_name))
        for type_params in [VanillaSGD(), Diana(), Artemis(), MCM()]:
            print(type_params)
            torch.cuda.empty_cache()
            params = type_params.define(cost_models=None,
                                        n_dimensions=None,
                                        nb_epoch=200,
                                        nb_devices=nb_devices,
                                        stochastic=stochastic,
                                        batch_size=batch_size,
                                        fraction_sampled_workers=1,
                                        up_compression_model=default_up_compression,
                                        down_compression_model=default_down_compression)

            params = cast_to_DL(params, dataset, models[dataset], optimal_steps_size[dataset], weight_decay[dataset], iid)
            params.log_file = log_file
            params.momentum = momentums[dataset]
            params.criterion = criterion[dataset]
            params.print()

            loaders = create_loaders(params)

            with open(params.log_file, 'a') as f:
                print(type_params, file=f)
                print("Optimal step size: ", params.optimal_step_size, file=f)

            multiple_descent = AverageOfSeveralIdenticalRun()
            seed_everything(seed=42)
            for i in range(NB_RUN):
                print('Run {:3d}/{:3d}:'.format(i + 1, NB_RUN))
                fixed_params = copy.deepcopy(params)
                multiple_descent.append_from_DL(run_tuned_exp(fixed_params, loaders))

            all_descent[type_params.name()] = multiple_descent
            res = ResultsOfSeveralDescents(all_descent, nb_devices)
            # res.add_descent(multiple_descent, type_params.name())

            pickle_saver(res, "{0}/{1}".format(algos_pickle_path, exp_name))

    # obj_min_cvx = pickle_loader("{0}/obj_min".format(pickle_path))
    obj_min = pickle_loader("{0}/obj_min_dl".format(pickle_path))

    res = pickle_loader("{0}/{1}".format(algos_pickle_path, exp_name))

    # obj_min = min(res.get_loss(np.array(0), in_log=False)[0])

    # print("Obj min in convex:", obj_min_cvx)
    print("Obj min in dl:", obj_min)

    # Plotting without averaging
    plot_error_dist(res.get_loss(np.array(obj_min)), res.names, res.nb_devices, batch_size=batch_size,
                    all_error=res.get_std(np.array(obj_min)), x_legend="Number of passes on data", ylegends="train_loss",
                    picture_name="{0}/{1}_train_losses".format(picture_path, exp_name))
    plot_error_dist(res.get_loss(np.array(obj_min)), res.names, res.nb_devices,
                    batch_size=batch_size, x_points=res.X_number_of_bits, ylegends="train_loss",
                    all_error=res.get_std(np.array(obj_min)), x_legend="Communicated bits",
                    picture_name="{0}/{1}_train_losses_bits".format(picture_path, exp_name))
    plot_error_dist(res.get_test_accuracies(), res.names, res.nb_devices, ylegends="accuracy",
                    all_error=res.get_test_accuracies_std(), x_legend="Number of passes on data", batch_size=batch_size,
                    picture_name="{0}/{1}_test_accuracies".format(picture_path, exp_name))
    plot_error_dist(res.get_test_losses(in_log=True), res.names, res.nb_devices, batch_size=batch_size,
                    all_error=res.get_test_losses_std(in_log=True), x_legend="Number of passes on data", ylegends="test_loss",
                    picture_name="{0}/{1}_test_losses".format(picture_path, exp_name))

if __name__ == '__main__':

    run_experiments_in_deeplearning(sys.argv[1])


