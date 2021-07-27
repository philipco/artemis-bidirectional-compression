"""
Created by Philippenko, 23rd July 2021.
"""
import copy
import sys
import logging

from src.deeplearning.main_DL import *
from src.machinery.PredefinedParameters import *
from src.utils.Utilities import pickle_saver

from src.utils.runner.RunnerUtilities import create_path_and_folders

logging.basicConfig(level=logging.INFO)


def compute_obj_min_dl(dataset: str, iid: str):

    log_file = "log_" + dataset + ".txt"
    with open(log_file, 'a') as f:
        print("==== COMPUTE OBJ LOSS ====", file=f)

    fraction_sampled_workers = 1
    batch_size = batch_sizes[dataset]
    nb_devices = 20
    stochastic = True

    algos = "uni-vs-bi" # Values of algos does not matter.

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers)

    loaders = create_loaders(dataset, iid, nb_devices, batch_size, stochastic)
    _, train_loader_workers_full, _, _ = loaders
    if optimal_steps_size[dataset] is None:
        L = compute_L(train_loader_workers_full)
        optimal_steps_size[dataset] = 1/L
        print("Step size:", optimal_steps_size[dataset])

    with open(log_file, 'a') as f:
        print("==> Computing objective loss.", file=f)
    params = VanillaSGD().define(cost_models=None,
                                 n_dimensions=None,
                                 stochastic=False,
                                 nb_epoch=40000,
                                 nb_devices=nb_devices,
                                 batch_size=batch_size,
                                 fraction_sampled_workers=1,
                                 up_compression_model=SQuantization(0, norm=norm_quantization[dataset]),
                                 down_compression_model=SQuantization(0, norm=norm_quantization[dataset]))

    params = cast_to_DL(params, dataset, models[dataset], optimal_steps_size[dataset], weight_decay[dataset], iid)
    params.log_file = log_file
    params.momentum = momentums[dataset]
    params.criterion = criterion[dataset]

    obj_min = run_tuned_exp(params, loaders).train_losses[-1]
    pickle_saver(obj_min, "{0}/obj_min_dl".format(pickle_path))


if __name__ == '__main__':

    compute_obj_min_dl(sys.argv[1], "iid")
    compute_obj_min_dl(sys.argv[1], "non-iid")


