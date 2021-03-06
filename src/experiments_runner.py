"""
Created by Philippenko, 15 January 2021
"""
import sys

from src.models.CostModel import LogisticModel, RMSEModel, build_several_cost_model

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


def run_experiments(nb_devices: int, stochastic: bool, dataset: str, iid: str, algos: str, use_averaging: bool = False,
                    scenario: str = None, fraction_sampled_workers: int = 0.5, plot_only: bool = True):

    print("Running with following parameters: {0}".format(["{0} -> {1}".format(k, v) for (k, v)
                                                           in zip(locals().keys(), locals().values())]))
    assert dataset in ["quantum", "superconduct", 'synth_logistic', 'synth_linear_noised', 'synth_linear_nonoised'], \
        "The available dataset are ['quantum', 'superconduct', 'synth_linear_noised', 'synth_linear_nonoised']."
    assert iid in ['iid', 'non-iid'], "The iid option are ['iid', 'non-iid']."
    assert scenario in [None, "compression", "step"], "The possible scenario are [None, 'compression', 'step']."

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos, fraction_sampled_workers)

    list_algos = choose_algo(algos, stochastic, fraction_sampled_workers)
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
                                                  stochastic=False
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
                                        filename=algos_pickle_path, batch_size=batch_size, stochastic=stochastic,
                                        step_formula=step_size, scenario=scenario)
        elif scenario == "step":
            run_for_different_scenarios(cost_models, list_algos, step_formula, label_step_formula,
                                        filename=algos_pickle_path, batch_size=batch_size, stochastic=stochastic,
                                        scenario=scenario, compression=compression_by_default)
        else:
            run_one_scenario(cost_models=cost_models, list_algos=list_algos, filename=algos_pickle_path,
                             batch_size=batch_size, stochastic=stochastic, nb_epoch=nb_epoch, step_size=step_size,
                             use_averaging=use_averaging, compression=compression_by_default,
                             fraction_sampled_workers=fraction_sampled_workers)

    obj_min = pickle_loader("{0}/obj_min".format(pickle_path))

    if stochastic:
        experiments_settings = "{0}-b{1}".format(stochasticity, batch_size)
    else:
        experiments_settings = stochasticity

    if scenario is None:
        res = pickle_loader("{0}/descent-{1}".format(algos_pickle_path, experiments_settings))

        # Plotting without averaging
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        all_error=res.get_std(obj_min), x_legend="Number of passes on data\n({0})".format(iid),
                        picture_name="{0}/it-noavg-{1}".format(picture_path, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        x_points=res.X_number_of_bits, x_legend="Communicated bits ({0})".format(iid),
                        all_error=res.get_std(obj_min), picture_name="{0}/bits-noavg-{1}"
                        .format(picture_path, experiments_settings))

        # Plotting with averaging
        if use_averaging:
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices,
                            dim_notebook, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Number of passes on data\n(Avg, {0})".format(iid),
                            picture_name="{0}/it-avg-{1}"
                            .format(picture_path, experiments_settings))
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, res.nb_devices, dim_notebook,
                            x_points=res.X_number_of_bits, all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Communicated bits (Avg, {0})".format(iid),
                            picture_name="{0}/bits-avg-{1}"
                            .format(picture_path, experiments_settings))

    if scenario == "step":
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        batch_size=batch_size,
                        all_error=res.get_std(obj_min),
                        x_legend="Step size ({0}, {1})".format(iid, str(compression_by_default.omega_c)[:4]),
                        one_on_two_points=True, xlabels=label_step_formula,
                        picture_name="{0}/{1}-{2}".format(picture_path, scenario, experiments_settings))

        res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        x_points=res.X_number_of_bits, batch_size=batch_size,
                        x_legend="Communicated bits\n(non-iid)", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}".format(picture_path, scenario, experiments_settings))

    if scenario == "compression":
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook, batch_size=batch_size,
                        all_error=res.get_std(obj_min), x_legend="$\omega_c$ ({0})".format(iid),
                        one_on_two_points=True, xlabels=label_compression,
                        picture_name="{0}/{1}-{2}".format(picture_path, scenario, experiments_settings))

        res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
                        all_error=res.get_std(obj_min), batch_size=batch_size,
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, res.nb_devices, dim_notebook,
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

