"""
Created by Philippenko, 4th May 2020.

This file give two functions (for single or multiple runs) to carry out a full gradient descent and retrieve results.
"""
import gc
import time
from tqdm import tqdm

from src.machinery.PredefinedParameters import *
from src.models.CompressionModel import CompressionModel

from src.utils.Constants import NB_EPOCH
from src.utils.Utilities import pickle_saver, get_project_root, create_folder_if_not_existing
from src.utils.runner.AverageOfSeveralIdenticalRun import AverageOfSeveralIdenticalRun
from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

nb_run = 5  # Number of gradient descent before averaging.


def choose_algo(algos: str, stochastic: bool = True, fraction_sampled_workers: int = 1):
    assert algos in ['uni-vs-bi', "with-without-ef", "compress-model", "mcm-vs-existing", "mcm-1-mem", "mcm-one-way",
                     "mcm-other-options", "artemis-vs-existing", "artemis-and-ef"], \
        "The possible choice of algorithms are : " \
        "uni-vs-bi (to compare uni-compression with bi-compression), " \
        "with-without-ef (to compare algorithms using or not error-feedback), " \
        "compress-model (algorithms compressing the model)."
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
        list_algos = [VanillaSGD(), Artemis(), RandMCM(), RandMCM1MemReset()]
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
    data_path = "{0}/pickle".format(get_project_root(), foldername)
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
    for i in range(nb_run):
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
        model_descent = predefined_parameters.type_FL()(params)
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


def run_one_scenario(cost_models, list_algos, filename: str, batch_size: int = 1, stochastic: bool = True,
                     nb_epoch: int = 250, step_size = None, compression: CompressionModel = None,
                     use_averaging: bool = False, fraction_sampled_workers: int = 1) -> None:
    all_descent = {}
    stochasticity = 'sto' if stochastic else "full"
    if stochastic:
        experiments_settings = "{0}-b{1}".format(stochasticity, batch_size)
    else:
        experiments_settings = stochasticity
    for type_params in tqdm(list_algos):
        multiple_sg_descent = multiple_run_descent(type_params, cost_models=cost_models,
                                                   compression_model=compression,
                                                   use_averaging=use_averaging,
                                                   stochastic=stochastic,
                                                   nb_epoch=nb_epoch,
                                                   step_formula=step_size,
                                                   batch_size=batch_size,
                                                   logs_file=filename,
                                                   fraction_sampled_workers=fraction_sampled_workers)
        all_descent[type_params.name()] = multiple_sg_descent
        res = ResultsOfSeveralDescents(all_descent, len(cost_models))
        pickle_saver(res, "{0}/descent-{1}".format(filename, experiments_settings))

def run_for_different_scenarios(cost_models, list_algos, values, labels, filename: str, batch_size: int = 1,
                                stochastic: bool = True, nb_epoch: int = 250, step_formula = None,
                                compression: CompressionModel = None, scenario: str = "step") -> None:

    assert scenario in ["step", "compression"], "There is two possible scenarios : to analyze by step size, or by compression operators."

    nb_devices_for_the_run = len(cost_models)

    all_kind_of_compression_res = []
    all_descent_various_gamma = {}
    descent_by_algo_and_step_size = {}

    # Corresponds to descent with optimal gamma for each algorithm
    optimal_descents = {}

    for param_algo in tqdm(list_algos):
        losses_by_algo, losses_avg_by_algo, norm_ef_by_algo, dist_model_by_algo = [], [], [], []
        var_models_by_algo = []
        descent_by_step_size = {}
        for (value, label) in zip(values, labels):

            if scenario == "step":
                multiple_sg_descent = multiple_run_descent(param_algo, cost_models=cost_models,
                                                           use_averaging=True, stochastic=stochastic, batch_size=batch_size,
                                                           step_formula=value, nb_epoch=nb_epoch, compression_model=compression,
                                                           logs_file=filename)

            if scenario == "compression":
                multiple_sg_descent = multiple_run_descent(param_algo, cost_models=cost_models,
                                                           use_averaging=True, stochastic=stochastic, batch_size=batch_size,
                                                           step_formula=step_formula,
                                                           compression_model=value, nb_epoch=nb_epoch,
                                                           logs_file=filename)

            descent_by_step_size[label] = multiple_sg_descent
            losses_by_label, losses_avg_by_label, norm_ef_by_label, dist_model_by_label = [], [], [], []
            var_models_by_label = []

            # Picking the minimum values for each of the run.
            for seq_losses, seq_losses_avg, seq_norm_ef, seq_dist_model, seq_var_models in \
                    zip(multiple_sg_descent.train_losses, multiple_sg_descent.averaged_train_losses,
                        multiple_sg_descent.norm_error_feedback, multiple_sg_descent.dist_to_model,
                        multiple_sg_descent.var_models):

                losses_by_label.append(min(seq_losses))
                losses_avg_by_label.append(min(seq_losses_avg))
                norm_ef_by_label.append(seq_norm_ef[-1])
                dist_model_by_label.append(seq_dist_model[-1])
                var_models_by_label.append(seq_var_models[-1])

            losses_by_algo.append(losses_by_label)
            losses_avg_by_algo.append(losses_avg_by_label)
            norm_ef_by_algo.append(norm_ef_by_label)
            dist_model_by_algo.append(dist_model_by_label)
            var_models_by_algo.append(var_models_by_label)

        descent_by_algo_and_step_size[param_algo.name()] = ResultsOfSeveralDescents(descent_by_step_size,
                                                                                          nb_devices_for_the_run)

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
                                                var_models_by_algo)
        all_descent_various_gamma[param_algo.name()] = artificial_multiple_descent
        all_kind_of_compression_res.append(all_descent_various_gamma)

    res_various_gamma = ResultsOfSeveralDescents(all_descent_various_gamma, nb_devices_for_the_run)

    stochasticity = 'sto' if stochastic else "full"
    if stochastic:
        experiments_settings = "{0}-b{1}".format(stochasticity, batch_size)
    else:
        experiments_settings = stochasticity

    pickle_saver(res_various_gamma, "{0}/{1}-{2}".format(filename, scenario, experiments_settings))

    res_opt_gamma = ResultsOfSeveralDescents(optimal_descents, nb_devices_for_the_run)
    pickle_saver(res_opt_gamma, "{0}/{1}-optimal-{2}".format(filename, scenario, experiments_settings))

    pickle_saver(descent_by_algo_and_step_size, "{0}/{1}-descent_by_algo-{2}"
                 .format(filename, scenario, experiments_settings))
