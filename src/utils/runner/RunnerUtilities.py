"""
Created by Philippenko, 4th May 2020.

This file give two functions (for single or multiple runs) to carry out a full gradient descent and retrieve results.
"""
import gc
import time
import tracemalloc

from pympler import asizeof
from tqdm import tqdm

from src.machinery.PredefinedParameters import *
from src.models.CompressionModel import CompressionModel

from src.utils.Constants import NB_EPOCH
from src.utils.PathDataset import get_path_to_pickle
from src.utils.Utilities import pickle_saver, get_project_root, create_folder_if_not_existing, pickle_loader, file_exist, remove_file
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

NB_RUN = 5  # Number of gradient descent before averaging.


def choose_algo(algos: str, stochastic: bool = True, fraction_sampled_workers: int = 1):
    assert algos in ['uni-vs-bi', "with-without-ef", "compress-model", "mcm-vs-existing", "mcm-1-mem", "mcm-one-way",
                     "mcm-other-options", "artemis-vs-existing", "artemis-and-ef"], \
        "The possible choice of algorithms are : " \
        "uni-vs-bi (to compare uni-compression with bi-compression), " \
        "with-without-ef (to compare algorithms using or not error-feedback), " \
        "compress-model (algorithms compressing the model)," \
        "mcm-other-options."
    if algos == 'uni-vs-bi':
        if fraction_sampled_workers==1:
            list_algos = [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis()]
        else:
            list_algos = [VanillaSGD(), VanillaSGDMem(), Qsgd(), Diana(), BiQSGD(), Artemis()]
    if algos == 'artemis-and-ef':
        list_algos = [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis(), Dore()]#, DoubleSqueeze()]
    elif algos == "with-without-ef":
        list_algos = [Qsgd(), Diana(), Artemis(), Dore(), DoubleSqueeze()]
    elif algos == "compress-model":
        list_algos = [VanillaSGD(), Artemis(), RArtemis(), ModelCompr(), MCM(), RandMCM()]
    elif algos == "mcm-vs-existing":
        if fraction_sampled_workers == 1:
            list_algos = [VanillaSGD(), Diana(), Artemis(), Dore(), MCM(), RandMCM()]
        else:
            list_algos = [VanillaSGD(), Diana(), Artemis(), Dore(), RandMCM()]
    elif algos == "mcm-1-mem":
        list_algos = [VanillaSGD(), Artemis(), RandMCM(), RandMCM1Mem(), RandMCM1MemReset()]
    elif algos == "mcm-other-options":
        list_algos = [ArtemisND(), MCM0(), MCM1(), MCM()]
    elif algos == "mcm-one-way":
        list_algos = [VanillaSGD(), DianaOneWay(), ArtemisOneWay(), DoreOneWay(), MCMOneWay(), RandMCMOneWay()]
    elif algos == "artemis-vs-existing":
        if stochastic:
            list_algos = [VanillaSGD(), FedAvg(), FedPAQ(), Diana(), Artemis(), Dore(), DoubleSqueeze()]
        else:
            list_algos = [VanillaSGD(), FedSGD(), FedPAQ(), Diana(), Artemis(), Dore(), DoubleSqueeze()]
    return list_algos


def create_path_and_folders(nb_devices: int, dataset: str, iid: str, algos: str, fraction_sampled_workers: int =1, model_name: str=None):
    if model_name is not None:
        foldername = "{0}-{1}-N{2}/{3}".format(dataset, iid, nb_devices, model_name)
    else:
        foldername = "{0}-{1}-N{2}".format(dataset, iid, nb_devices)
    picture_path = "{0}/pictures/{1}/{2}".format(get_project_root(), foldername, algos)
    if fraction_sampled_workers != 1:
        picture_path += "/pp-{0}".format(fraction_sampled_workers)
    # Contains the pickle of the dataset
    data_path = "{0}/pickle".format(get_path_to_pickle(), foldername)
    # Contains the pickle of the minimum objective.
    pickle_path = "{0}/{1}".format(data_path, foldername)
    # Contains the pickle of the gradient descent for each kind of algorithms.
    algos_pickle_path = "{0}/{1}".format(pickle_path, algos)
    if fraction_sampled_workers != 1:
        algos_pickle_path += "/pp-{0}".format(fraction_sampled_workers)

    # Create folders for pictures and pickle files
    create_folder_if_not_existing(algos_pickle_path)
    create_folder_if_not_existing(picture_path)
    return data_path, pickle_path, algos_pickle_path, picture_path

def multiple_run_descent(predefined_parameters: PredefinedParameters, cost_models, compression_model: CompressionModel,
                         nb_epoch=NB_EPOCH,
                         step_formula=None,
                         use_averaging=False,
                         stochastic: bool = True,
                         streaming: bool = False,
                         batch_size: int = 1,
                         fraction_sampled_workers: float = 1.,
                         logs_file: str = None) -> AverageOfSeveralIdenticalRun:
    """
    Run several time the same algorithm in the same conditions and gather all result in the MultipleDescentRun class.

    Args:
        predefined_parameters: predefined parameters
        X: data
        Y: labels
        nb_epoch: number of epoch for the each run
        quantization_param:
        step_formula: lambda function to compute the step size at each iteration.
        use_averaging: true if using Polyak-Rupper averaging.
        stochastic: true if running stochastic descent.

    Returns:
    """
    multiple_descent = AverageOfSeveralIdenticalRun()
    for i in range(NB_RUN):
        start_time = time.time()
        params = predefined_parameters.define(n_dimensions=cost_models[0].X.shape[1],
                                              nb_devices=len(cost_models),
                                              step_formula=step_formula,
                                              nb_epoch=nb_epoch,
                                              use_averaging=use_averaging,
                                              cost_models=cost_models,
                                              stochastic=stochastic,
                                              streaming=streaming,
                                              batch_size=batch_size,
                                              fraction_sampled_workers=fraction_sampled_workers,
                                              up_compression_model=compression_model,
                                              down_compression_model=compression_model)
        model_descent = predefined_parameters.type_FL()(params, logs_file)
        model_descent.run(cost_models)
        multiple_descent.append(model_descent)
        elapsed_time = time.time() - start_time

        if logs_file:
            logs = open("{0}/logs.txt".format(logs_file), "a+")
            logs.write("{0} - run {1}, final loss : {2}, memory : {3} Mbytes\n"
                       .format(predefined_parameters.name(), i, model_descent.train_losses[-1], model_descent.memory_info))
            logs.close()
        del model_descent
        gc.collect()
    return multiple_descent


def single_run_descent(cost_models, model: AGradientDescent, parameters: Parameters) -> AGradientDescent:
    model_descent = model(parameters)
    model_descent.run(cost_models)
    return model_descent


def run_one_scenario(cost_models, list_algos, logs_file: str, experiments_settings: str, batch_size: int = 1,
                     stochastic: bool = True, nb_epoch: int = 250, step_size = None,
                     compression: CompressionModel = None, use_averaging: bool = False,
                     fraction_sampled_workers: int = 1, modify_run = None) -> None:

    pickle_file = "{0}/descent-{1}".format(logs_file, experiments_settings)

    if modify_run is None:
        if file_exist(pickle_file + ".pkl"):
            remove_file(pickle_file  + ".pkl")
        algos = list_algos
    else:
        algos = [list_algos[i] for i in modify_run]
    for type_params in tqdm(algos):
        multiple_sg_descent = multiple_run_descent(type_params, cost_models=cost_models,
                                                   compression_model=compression,
                                                   use_averaging=use_averaging,
                                                   stochastic=stochastic,
                                                   nb_epoch=nb_epoch,
                                                   step_formula=step_size,
                                                   batch_size=batch_size,
                                                   logs_file=logs_file,
                                                   fraction_sampled_workers=fraction_sampled_workers)

        if logs_file:
            logs = open("{0}/logs.txt".format(logs_file), "a+")
            logs.write("{0} size of the multiple SG descent: {1:.2e} bits\n".format(type_params.name(), asizeof.asizeof(multiple_sg_descent)))
            logs.close()

        if file_exist(pickle_file + ".pkl"):
            res = pickle_loader(pickle_file)
            res.add_descent(multiple_sg_descent, type_params.name(), deep_learning_run=False)
        else:
            res = ResultsOfSeveralDescents(len(cost_models))
            res.add_descent(multiple_sg_descent, type_params.name(), deep_learning_run=False)

        pickle_saver(res, pickle_file)
        del res
        del multiple_sg_descent


def run_for_different_scenarios(cost_models, list_algos, values, labels, experiments_settings: str,
                                logs_file: str, batch_size: int = 1,
                                stochastic: bool = True, nb_epoch: int = 250, step_formula = None,
                                compression: CompressionModel = None, scenario: str = "step") -> None:

    assert scenario in ["step", "compression", "alpha"], "There is three possible scenarios : to analyze by step size," \
                                                         " by compression operators, or by value of alpha."

    nb_devices_for_the_run = len(cost_models)

    all_kind_of_compression_res = []
    all_descent_various_gamma = {}
    descent_by_algo_and_step_size = {}

    # Corresponds to descent with optimal gamma for each algorithm
    optimal_descents = {}

    for param_algo in tqdm(list_algos):
        losses_by_algo, losses_avg_by_algo, norm_ef_by_algo, dist_model_by_algo = [], [], [], []
        h_i_to_optimal_grad_by_algo, var_models_by_algo = [], []
        descent_by_step_size = {}
        for (value, label) in zip(values, labels):

            if scenario == "step":
                multiple_sg_descent = multiple_run_descent(param_algo, cost_models=cost_models,
                                                           use_averaging=True, stochastic=stochastic, batch_size=batch_size,
                                                           step_formula=value, nb_epoch=nb_epoch, compression_model=compression,
                                                           logs_file=logs_file)

            if scenario in ["compression", "alpha"]:
                multiple_sg_descent = multiple_run_descent(param_algo, cost_models=cost_models,
                                                           use_averaging=True, stochastic=stochastic, batch_size=batch_size,
                                                           step_formula=step_formula,
                                                           compression_model=value, nb_epoch=nb_epoch,
                                                           logs_file=logs_file)

            descent_by_step_size[label] = multiple_sg_descent
            losses_by_label, losses_avg_by_label, norm_ef_by_label, dist_model_by_label = [], [], [], []
            h_i_to_optimal_grad_by_label, var_models_by_label = [], []

            # Picking the minimum values for each of the run.
            for seq_losses, seq_losses_avg, seq_norm_ef, seq_dist_model, seq_h_i_optimal, seq_var_models in \
                    zip(multiple_sg_descent.train_losses, multiple_sg_descent.averaged_train_losses,
                        multiple_sg_descent.norm_error_feedback, multiple_sg_descent.dist_to_model,
                        multiple_sg_descent.h_i_to_optimal_grad, multiple_sg_descent.var_models):

                losses_by_label.append(min(seq_losses))
                losses_avg_by_label.append(min(seq_losses_avg))
                norm_ef_by_label.append(seq_norm_ef[-1])
                dist_model_by_label.append(seq_dist_model[-1])
                var_models_by_label.append(seq_var_models[-1])
                h_i_to_optimal_grad_by_label.append(seq_h_i_optimal[-1])

            losses_by_algo.append(losses_by_label)
            losses_avg_by_algo.append(losses_avg_by_label)
            norm_ef_by_algo.append(norm_ef_by_label)
            dist_model_by_algo.append(dist_model_by_label)
            var_models_by_algo.append(var_models_by_label)
            h_i_to_optimal_grad_by_algo.append(h_i_to_optimal_grad_by_label)

        res_by_algo_and_step_size = ResultsOfSeveralDescents(nb_devices_for_the_run)
        res_by_algo_and_step_size.add_dict_of_descent(descent_by_step_size)
        descent_by_algo_and_step_size[param_algo.name()] = res_by_algo_and_step_size

        # Find optimal descent for the algo:
        min_loss_desc = 10e12
        opt_desc = None
        for desc in descent_by_step_size.values():
            if min_loss_desc > min([desc.train_losses[j][-1] for j in range(len(desc.train_losses))]):
                min_loss_desc = min([desc.train_losses[j][-1] for j in range(len(desc.train_losses))])
                opt_desc = desc
        # Adding the optimal descent to the dict of optimal descent
        optimal_descents[param_algo.name()] = opt_desc


        artificial_multiple_descent = AverageOfSeveralIdenticalRun()
        artificial_multiple_descent.append_list(losses_by_algo, losses_avg_by_algo, norm_ef_by_algo, dist_model_by_algo,
                                                h_i_to_optimal_grad_by_algo, var_models_by_algo)
        all_descent_various_gamma[param_algo.name()] = artificial_multiple_descent
        all_kind_of_compression_res.append(all_descent_various_gamma)

    res_various_gamma = ResultsOfSeveralDescents(nb_devices_for_the_run)
    res_various_gamma.add_dict_of_descent(all_descent_various_gamma, deep_learning_run=False)

    pickle_saver(res_various_gamma, "{0}/{1}-{2}".format(logs_file, scenario, experiments_settings))

    res_opt_gamma = ResultsOfSeveralDescents(nb_devices_for_the_run)
    res_opt_gamma.add_dict_of_descent(optimal_descents, deep_learning_run=False)

    pickle_saver(res_opt_gamma, "{0}/{1}-optimal-{2}".format(logs_file, scenario, experiments_settings))

    pickle_saver(descent_by_algo_and_step_size, "{0}/{1}-descent_by_algo-{2}"
                 .format(logs_file, scenario, experiments_settings))
