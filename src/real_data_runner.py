"""
Created by Philippenko, 15 January 2021
"""

from src.models.CostModel import LogisticModel, RMSEModel, build_several_cost_model
from src.machinery.PredefinedParameters import *

from src.utils.ErrorPlotter import *
from src.utils.data.DataPreparation import build_data_logistic, build_data_linear
from src.utils.data.RealDatasetPreparation import prepare_quantum, prepare_superconduct
from src.utils.Constants import *
from src.utils.data.DataClustering import *
from src.utils.Utilities import pickle_loader, file_exist, create_folder_if_not_existing, get_project_root
from src.utils.runner.RunnerUtilities import *


def iid_step_size(it, L, omega, N): return 1 / (8 * L)
def deacreasing_step_size(it, L, omega, N): return 1 / (L * sqrt(it))
def batch_step_size(it, L, omega, N): return 1 / L


def run_real_dataset(nb_devices: int, stochastic: bool, dataset: str, iid: str, algos: str,
                     use_averaging: bool = False, scenario: str = None):

    foldername = "{0}-{1}-N{2}".format(dataset, iid, nb_devices)
    picture_path = "{0}/pictures/{1}".format(get_project_root(), foldername)
    pickle_path = "{0}/notebook/pickle/{1}".format(get_project_root(), foldername)

    # Create folders for pictures and pickle files
    create_folder_if_not_existing(pickle_path)
    create_folder_if_not_existing(picture_path)

    assert algos in ["uni-vs-bi", "with-without-ef", "compress-model"], "The possible choice of algorithms are : " \
        "uni-vs-bi (to compare uni-compression with bi-compression), "\
        "with-without-ef (to compare algorithms using or not error-feedback), " \
        "compress-model (algorithms compressing the model)."
    assert dataset in ["quantum", "superconduct", "synth_logistic", "synth_linear", "synth_linear_nonoised"], \
        "The available dataset are ['quantum', 'superconduct', 'synth_linear', 'synth_linear_nonoised']."
    assert iid in ["iid", "non-iid"], "The iid option are ['iid', 'non-iid']."
    assert scenario in [None, "compression", "step"], "The possible scenario are [None, 'compression', 'step']."

    nb_devices = nb_devices

    # Select the correct dataset
    if dataset == "quantum":
        X, Y, dim_notebook = prepare_quantum(nb_devices, iid=False)
        batch_size = 400
        model = LogisticModel
    elif dataset == "superconduct":
        X, Y, dim_notebook = prepare_superconduct(nb_devices, iid=False)
        batch_size = 50
        model = RMSEModel
    elif dataset == "synth_logistic":
        dim_notebook = 2
        batch_size = 1
        model = LogisticModel
        if not file_exist("data.pkl", pickle_path):
            w = torch.FloatTensor([10, 10]).to(dtype=torch.float64)
            X, Y = build_data_logistic(w, n_dimensions=2, n_devices=nb_devices, with_seed=False)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
    elif dataset == "synth_linear":
        dim_notebook = 20
        if not file_exist("data.pkl", pickle_path):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook- 1,
                                     n_devices=nb_devices, with_seed=False, without_noise=True)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
        model = RMSEModel
        batch_size = 1
    elif dataset == "synth_linear_nonoised":
        dim_notebook = 20
        if not file_exist("data.pkl", pickle_path):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook-1,
                                     n_devices=nb_devices, with_seed=False, without_noise=False)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
        model = RMSEModel
        batch_size = 1

    nb_epoch = 20 if stochastic else 200

    # Select the list of algorithms:
    if algos == "uni-vs-bi":
        list_algos = [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis()]
    elif algos == "with-without-ef":
        list_algos = [Qsgd(), Diana(), Artemis(), ArtemisEF(), DoubleSqueeze()]
    elif algos == "compress-model":
        list_algos = [VanillaSGD(), Artemis(), RArtemis(), ModelCompr(), ModelComprMem(), RModelComprMem()]

    compression_by_default = SQuantization(1, dim_notebook)

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
                                                  nb_devices=nb_devices,
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
    if stochastic and batch_size == 1:
        step_size = iid_step_size
    else:
        step_size = batch_step_size
    if "synth" in dataset and stochastic:
        step_size = deacreasing_step_size

    stochasticity = 'sto' if stochastic else "full"

    if scenario == "compression":
        run_for_different_scenarios(cost_models, list_algos, values_compression, label_compression,
                                    filename=pickle_path,
                                    batch_size=batch_size, stochastic=stochastic, step_formula=step_size,
                                    scenario=scenario)
    elif scenario == "step":
        run_for_different_scenarios(cost_models, list_algos, step_formula, label_step_formula,
                                    filename=pickle_path,
                                    batch_size=batch_size, stochastic=stochastic, scenario=scenario,
                                    compression=compression_by_default)
    else:
        print(step_size)
        print(stochastic)
        run_one_scenario(cost_models=cost_models, list_algos=list_algos,
                         filename=pickle_path, batch_size=batch_size,
                         stochastic=stochastic, nb_epoch=nb_epoch, step_size=step_size,
                         use_averaging=use_averaging,
                         compression=compression_by_default)

    obj_min = pickle_loader("{0}/obj_min".format(pickle_path))

    if scenario is None:
        res = pickle_loader("{0}/descent-{1}-b{2}".format(pickle_path, stochasticity, batch_size))

        # Plotting without averaging
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), x_legend="Number of passes on data\n({0})".format(iid),
                        picture_name="{0}/it-noavg-{1}-b{2}".format(picture_path, stochasticity, batch_size))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, x_legend="Communicated bits ({0})".format(iid),
                        all_error=res.get_std(obj_min), picture_name="{0}/bits-noavg-{1}-b{2}"
                        .format(picture_path, stochasticity, batch_size))

        # Plotting with averaging
        if use_averaging:
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run,
                            dim_notebook, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Number of passes on data\n(Avg, {0})".format(iid),
                            picture_name="{0}/it-avg-{1}-b{2}"
                            .format(picture_path, stochasticity, batch_size))
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run, dim_notebook,
                            x_points=res.X_number_of_bits, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Communicated bits (Avg, {0})".format(iid),
                            picture_name="{0}/bits-avg-{1}-b{2}"
                            .format(picture_path, stochasticity, batch_size))

    if scenario == "step":
        res = pickle_loader("{0}/{1}-{2}-b{3}".format(pickle_path, scenario, stochasticity, batch_size))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        batch_size=batch_size,
                        all_error=res.get_std(obj_min),
                        x_legend="Step size ({0}, {1})".format(iid, str(compression_by_default.omega_c)[:4]),
                        one_on_two_points=False, xlabels=label_step_formula,
                        picture_name="{0}/{1}-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))

        res = pickle_loader("{0}/{1}-optimal-{2}-b{3}".format(pickle_path, scenario, stochasticity, batch_size))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, batch_size=batch_size,
                        x_legend="Communicated bits\n(non-iid)", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))

    if scenario == "compression":
        res = pickle_loader("{0}/{1}-{2}-b{3}".format(pickle_path, scenario, stochasticity, batch_size))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook, batch_size=batch_size,
                        all_error=res.get_std(obj_min), x_legend="$\omega_c$ ({0})".format(iid),
                        one_on_two_points=False, xlabels=label_compression,
                        picture_name="{0}/{1}-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))

        res = pickle_loader("{0}/{1}-optimal-{2}-b{3}".format(pickle_path, scenario, stochasticity, batch_size))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, batch_size=batch_size,
                        x_legend="Communicated bits\n(non-iid)", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}-b{3}".format(picture_path, scenario, stochasticity, batch_size))


if __name__ == '__main__':
    run_real_dataset(nb_devices=20, stochastic=False, dataset="quantum", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True, scenario="step")
    run_real_dataset(nb_devices=20, stochastic=False, dataset="superconduct", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True, scenario="step")
    run_real_dataset(nb_devices=20, stochastic=False, dataset="quantum", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True, scenario="compression")
    run_real_dataset(nb_devices=20, stochastic=False, dataset="superconduct", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True, scenario="compression")
    run_real_dataset(nb_devices=20, stochastic=False, dataset="quantum", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True)
    run_real_dataset(nb_devices=20, stochastic=False, dataset="superconduct", iid="non-iid", algos="uni-vs-bi",
                     use_averaging=True)
