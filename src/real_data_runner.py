"""
Created by Philippenko, 15 January 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, interact
from math import sqrt

from sklearn.preprocessing import scale

from src.models.CostModel import LogisticModel, RMSEModel, build_several_cost_model

from src.machinery.GradientDescent import ArtemisDescent, SGD_Descent
from src.machinery.GradientUpdateMethod import ArtemisUpdate
from src.machinery.Parameters import *
from src.machinery.PredefinedParameters import *

from src.utils.ErrorPlotter import *
from src.utils.data.RealDatasetPreparation import prepare_quantum, prepare_superconduct
from src.utils.Constants import *
from src.utils.data.DataClustering import *
from src.utils.data.DataPreparation import build_data_logistic, add_bias_term
from src.utils.Utilities import pickle_loader, pickle_saver, file_exist, create_folder_if_not_existing, get_project_root
from src.utils.runner.RunnerUtilities import *
from src.utils.runner.ResultsOfSeveralDescents import ResultsOfSeveralDescents

def iid_step_size(it, L, omega, N): return 1 / (8 * L)
def noniid_step_size(it, L, omega, N): return 1 / L

def run_real_dataset(nb_devices: int, stochastic: bool, dataset: str, iid: str, algos: str,
                     use_averaging: bool = False, scenario: str = None):

    foldername = "{0}-{1}-N{2}".format(dataset, iid, nb_devices)
    picture_path = "{0}/pictures".format(get_project_root())
    pickle_path = "{0}/notebook/pickle/{1}".format(get_project_root(), foldername)


    # Create folders for pictures and pickle files
    create_folder_if_not_existing(pickle_path)
    create_folder_if_not_existing(picture_path)

    assert algos in ["uni-vs-bi", "with-without-ef", "compress-model"], "The possible choice of algorithms are : " \
        "uni-vs-bi (to compare uni-compression with bi-compression), "\
        "with-without-ef (to compare algorithms using or not error-feedback), " \
        "compress-model (algorithms compressing the model)."
    assert dataset in ["quantum", "superconduct"], "The available dataset are ['quantum', 'superconduct']."
    assert iid in ["iid", "non-iid"], "The iid option are ['iid', 'non-iid']."
    # assert stochasticity in ["sto", "full"], "The option for stochastic are ['sto', 'full']."
    assert scenario in [None, "compression", "step"], "The possible scenario are [None, 'compression', 'step']."

    nb_cluster = nb_devices

    # Select the correct dataset
    if dataset == "quantum":
        X, Y, dim_notebook = prepare_quantum(nb_cluster, iid=False)
        batch_size = 400
        model = LogisticModel
    elif dataset == "superconduct":
        X, Y, dim_notebook = prepare_superconduct(nb_cluster, iid=False)
        batch_size = 50
        model = RMSEModel

    # Select the correct list of algorithms:
    if algos == "uni-vs-bi":
        list_algos = [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis()]
    elif algos == "with-without-ef":
        list_algos = [Qsgd(), Diana(), Artemis(), ArtemisEF(), DoubleSqueeze()]
    elif algos == "compress-model":
        list_algos = [VanillaSGD(), Artemis(), RArtemis(), ModelComprMem(), RModelComprMem()]

    values_compression = [SQuantization(0, dim_notebook),
                          SQuantization(16, dim_notebook),
                          SQuantization(8, dim_notebook),
                          SQuantization(4, dim_notebook),
                          SQuantization(3, dim_notebook),
                          SQuantization(2, dim_notebook),
                          SQuantization(1, dim_notebook)
                          ]

    label_compression = ["SGD"] + [str(value.omega_c)[:4] for value in values_compression[1:]]


    # Rebalancing cluster: the biggest one must not be more than 10times bigger than the smallest one.
    X_r, Y_r = rebalancing_clusters(X, Y)

    # Creating cost models which will be used to computed cost/loss, gradients, L ...
    cost_models = build_several_cost_model(model, X_r, Y_r, nb_devices)

    if not file_exist("obj_min.pkl", pickle_path):
        obj_min_by_N_descent = SGD_Descent(Parameters(n_dimensions=dim_notebook,
                                                  nb_devices=nb_cluster,
                                                  nb_epoch=10000,
                                                  momentum=0.,
                                                  verbose=True,
                                                  cost_models=cost_models,
                                                  stochastic=False,
                                                  bidirectional=False
                                                  ))
        obj_min_by_N_descent.run(cost_models)
        obj_min = obj_min_by_N_descent.losses[-1]
        pickle_saver(obj_min, "{0}/obj_min".format(pickle_path))

    # Choice of step size
    if iid == "iid":
        step_size = iid_step_size
    else:
        step_size = noniid_step_size

    stochasticity = ['sto'] if stochastic else "full"

    if scenario == "compression":
        run_for_different_scenarios(cost_models, list_algos, values_compression, label_compression,
                                    filename=pickle_path,
                                    batch_size=batch_size, stochastic=stochastic, step_formula=step_size, scenario=scenario)
    elif scenario == "step":
        run_for_different_scenarios(cost_models, list_algos, step_formula, label_step_formula,
                                    filename=pickle_path,
                                    batch_size=batch_size, stochastic=stochastic, scenario=scenario,
                                    compression=SQuantization(dim_notebook, 1))
    else:
        run_one_scenario(cost_models=cost_models, list_algos=list_algos,
                         filename=pickle_path, batch_size=batch_size,
                         stochastic=stochastic, nb_epoch=500, step_size=step_size,
                         compression=SQuantization(1, dim_notebook))

    obj_min = pickle_loader("{0}/obj_min".format(pickle_path))
    res = pickle_loader("{0}/descent-{1}-b{2}".format(pickle_path, stochasticity, batch_size))

    # Plotting without averaging
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    all_error=res.get_std(obj_min), x_legend="Number of passes on data\n({0})".format(iid),
                    picture_name="{0}-it-noavg".format(picture_path))
    plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                    x_points=res.X_number_of_bits, x_legend="Communicated bits ({0})".format(iid),
                    all_error=res.get_std(obj_min), picture_name="{0}-bits-noavg".format(picture_path))

    # Plotting with averaging
    if use_averaging:
        plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run,
                        dim_notebook, all_error=res.get_std(obj_min, averaged=True),
                        x_legend="Number of passes on data\n(Avg, {0})".format(iid),
                        picture_name="{0}-it-avg".format(picture_path))
        plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, all_error=res.get_std(obj_min, averaged=True),
                        x_legend="Communicated bits (Avg, {0})".format(iid),
                        picture_name="{0}-bits-avg".format(picture_path))

