"""
Created by Philippenko, 15 January 2021
"""
import sys

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


def run_experiments(nb_devices: int, stochastic: bool, dataset: str, iid: str, algos: str,
                    use_averaging: bool = False, scenario: str = None, plot_only: bool = False):

    print("Running with following parameters: {0}".format(["{0} -> {1}".format(k, v) for (k, v)
                                                           in zip(locals().keys(), locals().values())]))

    assert algos in ['uni-vs-bi', "with-without-ef", "compress-model", "mcm-vs-existing", "mcm-one-way", "mcm-other-options",
                     "artemis-vs-existing"],\
        "The possible choice of algorithms are : " \
        "uni-vs-bi (to compare uni-compression with bi-compression), " \
        "with-without-ef (to compare algorithms using or not error-feedback), " \
        "compress-model (algorithms compressing the model)."
    assert dataset in ["quantum", "superconduct", 'synth_logistic', 'synth_linear_noised', 'synth_linear_nonoised'], \
        "The available dataset are ['quantum', 'superconduct', 'synth_linear_noised', 'synth_linear_nonoised']."
    assert iid in ['iid', 'non-iid'], "The iid option are ['iid', 'non-iid']."
    assert scenario in [None, "compression", "step"], "The possible scenario are [None, 'compression', 'step']."

    foldername = "{0}-{1}-N{2}".format(dataset, iid, nb_devices)
    picture_path = "{0}/pictures/{1}/{2}".format(get_project_root(), foldername, algos)
    # Contains the pickle of the dataset
    data_path = "{0}/pickle".format(get_project_root(), foldername)
    # Contains the pickle of the minimum objective.
    pickle_path = "{0}/{1}".format(data_path, foldername)
    # Contains the pickle of the gradient descent for each kind of algorithms.
    algos_pickle_path = "{0}/{1}".format(pickle_path, algos)

    # Create folders for pictures and pickle files
    create_folder_if_not_existing(algos_pickle_path)
    create_folder_if_not_existing(picture_path)

    nb_devices = nb_devices
    nb_epoch = 100 if stochastic else 400

    iid_data = True if iid == 'iid' else False

    # Select the correct dataset
    if dataset == "quantum":
        X, Y, dim_notebook = prepare_quantum(nb_devices, data_path=data_path, pickle_path=pickle_path, iid=iid_data)
        batch_size = 400
        model = LogisticModel
        nb_epoch = 500 if stochastic else 400
    elif dataset == "superconduct":
        X, Y, dim_notebook = prepare_superconduct(nb_devices, data_path= data_path, pickle_path=pickle_path, iid=iid_data)
        batch_size = 50
        model = RMSEModel
        nb_epoch = 500 if stochastic else 400
    elif dataset == 'synth_logistic':
        dim_notebook = 2
        batch_size = 1
        model = LogisticModel
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w = torch.FloatTensor([10, 10]).to(dtype=torch.float64)
            X, Y = build_data_logistic(w, n_dimensions=2, n_devices=nb_devices, with_seed=False)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
    elif dataset == 'synth_linear_noised':
        dim_notebook = 20
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook-1,
                                     n_devices=nb_devices, with_seed=False, without_noise=False)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
        model = RMSEModel
        batch_size = 1
    elif dataset == 'synth_linear_nonoised':
        dim_notebook = 20
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook-1,
                                     n_devices=nb_devices, with_seed=False, without_noise=True)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
        model = RMSEModel
        batch_size = 1

    # Select the list of algorithms:
    if algos == 'uni-vs-bi':
        list_algos = [VanillaSGD(), Qsgd(), Diana(), BiQSGD(), Artemis()]
    elif algos == "with-without-ef":
        list_algos = [Qsgd(), Diana(), Artemis(), Dore(), DoubleSqueeze()]
    elif algos == "compress-model":
        list_algos = [VanillaSGD(), Artemis(), RArtemis(), ModelCompr(), MCM(), RandMCM()]
    elif algos == "mcm-vs-existing":
        list_algos = [VanillaSGD(), Diana(), Artemis(), Dore(), MCM(), RandMCM()]
    elif algos == "mcm-other-options":
        list_algos = [ArtemisND(), MCM0(), MCM1(), MCM()]
    elif algos == "mcm-one-way":
        list_algos = [VanillaSGD(), DianaOneWay(), ArtemisOneWay(), DoreOneWay(), MCMOneWay(), RandMCMOneWay()]
    elif algos == "artemis-vs-existing":
        if stochastic:
            list_algos = [VanillaSGD(), FedAvg(), FedPAQ(), Diana(), Artemis(), Dore(), DoubleSqueeze()]
        else:
            list_algos = [VanillaSGD(), FedSGD(), FedPAQ(), Diana(), Artemis(), Dore(), DoubleSqueeze()]

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

    if not file_exist("{0}/obj_min.pkl".format(pickle_path)):
        obj_min_by_N_descent = SGD_Descent(Parameters(n_dimensions=dim_notebook,
                                                  nb_devices=nb_devices,
                                                  nb_epoch=40000,
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
    if 'synth' in dataset and stochastic:
        step_size = deacreasing_step_size

    stochasticity = 'sto' if stochastic else "full"

    if not plot_only:
        if scenario == "compression":
            run_for_different_scenarios(cost_models, list_algos[1:], values_compression, label_compression,
                                        filename=algos_pickle_path,
                                        batch_size=batch_size, stochastic=stochastic, step_formula=step_size,
                                        scenario=scenario)
        elif scenario == "step":
            run_for_different_scenarios(cost_models, list_algos, step_formula, label_step_formula,
                                        filename=algos_pickle_path,
                                        batch_size=batch_size, stochastic=stochastic, scenario=scenario,
                                        compression=compression_by_default)
        else:
            run_one_scenario(cost_models=cost_models, list_algos=list_algos,
                             filename=algos_pickle_path, batch_size=batch_size,
                             stochastic=stochastic, nb_epoch=nb_epoch, step_size=step_size,
                             use_averaging=use_averaging,
                             compression=compression_by_default)

    obj_min = pickle_loader("{0}/obj_min".format(pickle_path))

    if stochastic:
        experiments_settings = "{0}-b{1}".format(stochasticity, batch_size)
    else:
        experiments_settings = stochasticity

    if scenario is None:
        res = pickle_loader("{0}/descent-{1}".format(algos_pickle_path, experiments_settings))

        # Plotting without averaging
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), x_legend="Number of passes on data\n({0})".format(iid),
                        picture_name="{0}/it-noavg-{1}".format(picture_path, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, x_legend="Communicated bits ({0})".format(iid),
                        all_error=res.get_std(obj_min), picture_name="{0}/bits-noavg-{1}"
                        .format(picture_path, experiments_settings))

        # Plotting with averaging
        if use_averaging:
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run,
                            dim_notebook, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Number of passes on data\n(Avg, {0})".format(iid),
                            picture_name="{0}/it-avg-{1}"
                            .format(picture_path, experiments_settings))
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices_for_the_run, dim_notebook,
                            x_points=res.X_number_of_bits, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Communicated bits (Avg, {0})".format(iid),
                            picture_name="{0}/bits-avg-{1}"
                            .format(picture_path, experiments_settings))

    if scenario == "step":
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        batch_size=batch_size,
                        all_error=res.get_std(obj_min),
                        x_legend="Step size ({0}, {1})".format(iid, str(compression_by_default.omega_c)[:4]),
                        one_on_two_points=True, xlabels=label_step_formula,
                        picture_name="{0}/{1}-{2}".format(picture_path, scenario, experiments_settings))

        res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, batch_size=batch_size,
                        x_legend="Communicated bits\n(non-iid)", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}".format(picture_path, scenario, experiments_settings))

    if scenario == "compression":
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook, batch_size=batch_size,
                        all_error=res.get_std(obj_min), x_legend="$\omega_c$ ({0})".format(iid),
                        one_on_two_points=True, xlabels=label_compression,
                        picture_name="{0}/{1}-{2}".format(picture_path, scenario, experiments_settings))

        res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices_for_the_run, dim_notebook,
                        x_points=res.X_number_of_bits, batch_size=batch_size,
                        x_legend="Communicated bits\n(non-iid)", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}".format(picture_path, scenario, experiments_settings))


if __name__ == '__main__':

    if sys.argv[1] == "synth":
        run_experiments(nb_devices=20, stochastic=False, dataset='synth_logistic', iid='non-iid', algos=sys.argv[2],
                        use_averaging=True)
        run_experiments(nb_devices=20, stochastic=True, dataset='synth_logistic', iid='non-iid', algos=sys.argv[2],
                        use_averaging=True)
        run_experiments(nb_devices=20, stochastic=False, dataset='synth_linear_noised', iid='non-iid', algos=sys.argv[2],
                        use_averaging=True)
        run_experiments(nb_devices=20, stochastic=True, dataset='synth_linear_noised', iid='non-iid', algos=sys.argv[2],
                        use_averaging=True)
        run_experiments(nb_devices=20, stochastic=True, dataset='synth_linear_nonoised', iid='non-iid', algos=sys.argv[2],
                        use_averaging=True)

    elif sys.argv[1] == "real":
        for sto in [True, False]:
            for iid in ["non-iid", "iid"]:
                for dataset in ["quantum", "superconduct"]:
                    run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid=iid, algos=sys.argv[2],
                                    use_averaging=True)
                    run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid='non-iid', algos=sys.argv[2],
                                    use_averaging=True, scenario="step")
                    run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid='non-iid', algos=sys.argv[2],
                                      use_averaging=True, scenario="compression")

