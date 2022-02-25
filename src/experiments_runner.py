"""
Created by Philippenko, 15 January 2021
"""
import sys

from src.models.CostModel import build_several_cost_model
from src.models.RegularizationModel import L2Regularization
from src.utils.ConvexSettings import batch_sizes, models

from src.utils.ErrorPlotter import *
from src.utils.data.DataPreparation import build_data_logistic, build_data_linear
from src.utils.data.RealDatasetPreparation import get_preparation_operator_of_dataset
from src.utils.Constants import *
from src.utils.data.DataClustering import *
from src.utils.runner.RunnerUtilities import *


def iid_step_size(it, L, omega, N): return 1 / (8 * L)
def deacreasing_step_size(it, L, omega, N): return 1 / (L * sqrt(it))
def batch_step_size(it, L, omega, N): return 1 / L


def run_experiments(nb_devices: int, stochastic: bool, dataset: str, iid: str, algos: str, use_averaging: bool = False,
                    scenario: str = None, fraction_sampled_workers: int = 1, plot_only: bool = False, modify_run=None,
                    dirichlet = None, pp_strategy: str = "pp2"):

    print("Running with following parameters: {0}".format(["{0} -> {1}".format(k, v) for (k, v)
                                                           in zip(locals().keys(), locals().values())]))
    assert dataset in ["quantum", "superconduct", "mushroom", "phishing", "a9a", "abalone", "covtype", 'synth_logistic',
                       'madelon', 'gisette', 'w8a', 'synth_linear_noised', 'synth_linear_nonoised'], \
        "The available dataset are ['quantum', 'superconduct', 'synth_linear_noised', 'synth_linear_nonoised']."
    assert iid in ['iid', 'non-iid'], "The iid option are ['iid', 'non-iid']."
    assert scenario in [None, "compression", "step", "alpha"], "The possible scenario are [None, 'compression', 'step', 'alpha']."

    # tracemalloc.start()
    # hp = hpy()

    data_path, pickle_path, algos_pickle_path, picture_path = create_path_and_folders(nb_devices, dataset, iid, algos,
                                                                                      fraction_sampled_workers,
                                                                                      pp_strategy=pp_strategy)

    list_algos = choose_algo(algos, stochastic, fraction_sampled_workers, pp_strategy)
    nb_devices = nb_devices
    nb_epoch = 200 if stochastic else 800

    iid_data = True if iid == 'iid' else False

    batch_size = batch_sizes[dataset]
    model = models[dataset]

    if dataset not in ['synth_logistic', 'synth_linear_noised', 'synth_linear_nonoised']:
        prepare_dataset = get_preparation_operator_of_dataset(dataset)
        X, Y, dim_notebook = prepare_dataset(nb_devices, data_path=data_path, pickle_path=pickle_path, iid=iid_data, dirichlet = dirichlet)

    elif dataset == 'synth_logistic':
        nb_epoch = 100 if stochastic else 400
        dim_notebook = 2
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w = torch.FloatTensor([10, 10]).to(dtype=torch.float64)
            X, Y = build_data_logistic(w, n_dimensions=2, n_devices=nb_devices, with_seed=False)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
    elif dataset == 'synth_linear_noised':
        nb_epoch = 100 if stochastic else 400
        dim_notebook = 20
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook-1,
                                     n_devices=nb_devices, with_seed=False, without_noise=False)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")
    elif dataset == 'synth_linear_nonoised':
        nb_epoch = 100 if stochastic else 400
        dim_notebook = 20
        if not file_exist("{0}/data.pkl".format(pickle_path)):
            w_true = generate_param(dim_notebook-1)
            X, Y = build_data_linear(w_true, n_dimensions=dim_notebook-1,
                                     n_devices=nb_devices, with_seed=False, without_noise=True)
            X = add_bias_term(X)
            pickle_saver((X, Y), pickle_path + "/data")
        else:
            X, Y = pickle_loader(pickle_path + "/data")

    default_level_of_quantization = 1 if fraction_sampled_workers == 1 else 2
    compression_by_default = SQuantization(default_level_of_quantization, dim_notebook, norm=2)

    values_compression = [SQuantization(0, dim_notebook, norm=2),
                          SQuantization(16, dim_notebook, norm=2),
                          SQuantization(8, dim_notebook, norm=2),
                          SQuantization(4, dim_notebook, norm=2),
                          SQuantization(3, dim_notebook, norm=2),
                          SQuantization(2, dim_notebook, norm=2),
                          SQuantization(1, dim_notebook, norm=2)
                          ]

    label_compression = ["SGD"] + [str(value.omega_c)[:4] for value in values_compression[1:]]

    values_alpha = [SQuantization(1, dim_notebook, norm=2, constant=0.25),
                    SQuantization(1, dim_notebook, norm=2, constant=0.5),
                    SQuantization(1, dim_notebook, norm=2, constant=1),
                    SQuantization(1, dim_notebook, norm=2, constant=2),
                    SQuantization(1, dim_notebook, norm=2, constant=4),
                    SQuantization(1, dim_notebook, norm=2, constant=8)
                    ]

    label_alpha = [str(value.constant) for value in values_alpha]

    # Creating cost models which will be used to computed cost/loss, gradients, L ...
    default_regularizer = L2Regularization(regularization_rate=0.0)
    cost_models = build_several_cost_model(model, X, Y, nb_devices, regularization=default_regularizer)

    # hp.setrelheap()

    if dirichlet is not None:
        obj_type = "dirichlet-{0}".format(dirichlet)
    else:
        obj_type = "TSNE"
    obj_name = "{0}/obj_min-{1}".format(pickle_path, obj_type)

    if not file_exist("{0}.pkl".format(obj_name)) \
            or not file_exist("{0}/grads_min-{1}.pkl".format(pickle_path, obj_type)):
        obj_min_by_N_descent = SGD_Descent(Parameters(n_dimensions=dim_notebook,
                                                  nb_devices=nb_devices,
                                                  nb_epoch=40000,
                                                  momentum=0.,
                                                  verbose=True,
                                                  cost_models=cost_models,
                                                  stochastic=False
                                                  ), None)

        obj_min_by_N_descent.run(cost_models)
        obj_min = obj_min_by_N_descent.train_losses[-1]

        pickle_saver(obj_min, obj_name)

        grads_min = [worker.local_update.g_i for worker in obj_min_by_N_descent.workers]
        pickle_saver(grads_min, "{0}/grads_min-{1}".format(pickle_path, obj_type))

    # Choice of step size
    if stochastic and batch_size == 1:
        step_size = iid_step_size
    if algos == "artemis-vs-existing":
        step_size = lambda it, L, omega, N: 1 / (2*L)
        label_step_size = "1/2L"
    else:
        step_size = lambda it, L, omega, N: 1 / L
        label_step_size = "1/L"

    stochasticity = 'sto' if stochastic else "full"
    reg = "-reg{0}".format(
        (default_regularizer.regularization_rate)) if default_regularizer.regularization_rate != 0 else ""
    if stochastic:
        experiments_settings = "{0}-{1}-b{2}{3}".format(compression_by_default.get_name(), stochasticity,
                                                        batch_size, reg)
    else:
        experiments_settings = "{0}-{1}{2}".format(compression_by_default.get_name(), stochasticity, reg)

    if not plot_only:
        if scenario == "compression":
            run_for_different_scenarios(cost_models, list_algos[1:], values_compression, label_compression,
                                        experiments_settings=experiments_settings,
                                        logs_file=algos_pickle_path, batch_size=batch_size, stochastic=stochastic,
                                        step_formula=step_size, scenario=scenario)
        elif scenario == "step":
            run_for_different_scenarios(cost_models, list_algos, step_formula, label_step_formula,
                                        experiments_settings=experiments_settings,
                                        logs_file=algos_pickle_path, batch_size=batch_size, stochastic=stochastic,
                                        scenario=scenario, compression=compression_by_default)
        elif scenario == "alpha":
            run_for_different_scenarios(cost_models, [Artemis(), MCM(), RandMCM()], values_alpha, label_alpha,
                                        experiments_settings=experiments_settings,
                                        logs_file=algos_pickle_path, batch_size=batch_size, stochastic=stochastic,
                                        scenario=scenario, compression=compression_by_default)
        else:
            run_one_scenario(cost_models=cost_models, list_algos=list_algos, logs_file=algos_pickle_path,
                             batch_size=batch_size, experiments_settings=experiments_settings,
                             stochastic=stochastic, nb_epoch=nb_epoch, step_size=step_size,
                             use_averaging=use_averaging, compression=compression_by_default,
                             fraction_sampled_workers=fraction_sampled_workers, modify_run=modify_run, pp_strategy=pp_strategy)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    #
    # print("[ Top 10 ]")
    # for stat in top_stats[:10]:
    #     print(stat)
    #
    # h = hp.heap()
    # print(h)

    obj_min = pickle_loader(obj_name)
    print("Obj min:", obj_min)

    if scenario is None:
        res = pickle_loader("{0}/descent-{1}".format(algos_pickle_path, experiments_settings))

        # Plotting without averaging
        plot_error_dist(res.get_loss(obj_min), res.names,
                        all_error=res.get_std(obj_min), x_legend="Number of passes on data",
                        picture_name="{0}/it-noavg-{1}".format(picture_path, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names,
                        x_points=res.X_number_of_bits, x_legend="Communicated bits",
                        all_error=res.get_std(obj_min), picture_name="{0}/bits-noavg-{1}"
                        .format(picture_path, experiments_settings))

        # Plotting with averaging
        if use_averaging:
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names,
                            all_error=res.get_std(obj_min, averaged=True), x_legend="Number of passes on data (Avg)",
                            picture_name="{0}/it-avg-{1}".format(picture_path, experiments_settings))
            plot_error_dist(res.get_loss(obj_min, averaged=True), res.names, x_points=res.X_number_of_bits,
                            all_error=res.get_std(obj_min, averaged=True),
                            x_legend="Communicated bits (Avg)", picture_name="{0}/bits-avg-{1}"
                            .format(picture_path, experiments_settings))

    if scenario == "step":
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, all_error=res.get_std(obj_min),
                        x_legend="Step size ({0}, {1})".format(iid, str(compression_by_default.omega_c)[:4]),
                        one_on_two_points=True, xlabels=label_step_formula,
                        picture_name="{0}/{1}-{2}".format(picture_path, scenario, experiments_settings))

        res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))

        plot_error_dist(res.get_loss(obj_min), res.names, all_error=res.get_std(obj_min),
                        x_legend="(non-iid)", ylim=True,
                        picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, x_points=res.X_number_of_bits,
                        x_legend="Communicated bits", all_error=res.get_std(obj_min), ylim=True,
                        picture_name="{0}/{1}-optimal-bits-{2}".format(picture_path, scenario, experiments_settings))

    if scenario in ["compression", "alpha"]:
        create_folder_if_not_existing("{0}/{1}".format(picture_path, scenario))
        res = pickle_loader("{0}/{1}-{2}".format(algos_pickle_path, scenario, experiments_settings))
        plot_error_dist(res.get_loss(obj_min), res.names, all_error=res.get_std(obj_min),
                        x_legend=["$\omega_c$", "$\\alpha_{dwn}^{-1} \\times (\omega_c + 1)^{-1}$"][scenario=="alpha"],
                        one_on_two_points=True, xlabels=[label_compression, label_alpha][scenario=="alpha"],
                        picture_name="{0}/{1}/{2}".format(picture_path, scenario, experiments_settings))

        res_by_algo = pickle_loader("{0}/{1}-descent_by_algo-{2}".format(algos_pickle_path, scenario, experiments_settings))
        for key in res_by_algo.keys():
            res = res_by_algo[key]
            plot_error_dist(res.get_loss(obj_min), res.names, all_error=res.get_std(obj_min),
                            x_legend="Number of passes on data", ylim=1, picture_name="{0}/{1}/{2}-it-noavg-{3}"
                            .format(picture_path, scenario, key, experiments_settings))

        # res = pickle_loader("{0}/{1}-optimal-{2}".format(algos_pickle_path, scenario, experiments_settings))
        #
        # plot_error_dist(res.get_loss(obj_min), res.names,
        #                 all_error=res.get_std(obj_min),
        #                 x_legend="(non-iid)", ylim=True,
        #                 picture_name="{0}/{1}-optimal-it-{2}".format(picture_path, scenario, experiments_settings))
        # plot_error_dist(res.get_loss(obj_min), res.names,
        #                 x_points=res.X_number_of_bits,
        #                 x_legend="Communicated bits", all_error=res.get_std(obj_min), ylim=True,
        #                 picture_name="{0}/{1}-optimal-bits-{2}".format(picture_path, scenario, experiments_settings))


if __name__ == '__main__':

    if sys.argv[1] == "synth":
        if sys.argv[2] == "logistic":
            run_experiments(nb_devices=20, stochastic=False, dataset='synth_logistic', iid='non-iid', algos=sys.argv[3],
                            use_averaging=True)
            run_experiments(nb_devices=20, stochastic=True, dataset='synth_logistic', iid='non-iid', algos=sys.argv[3],
                            use_averaging=True)
        elif sys.argv[2] == "linear":
            run_experiments(nb_devices=20, stochastic=False, dataset='synth_linear_noised', iid='non-iid', algos=sys.argv[3],
                            use_averaging=True)
            run_experiments(nb_devices=20, stochastic=True, dataset='synth_linear_noised', iid='non-iid', algos=sys.argv[3],
                            use_averaging=True)
            run_experiments(nb_devices=20, stochastic=True, dataset='synth_linear_nonoised', iid='non-iid', algos=sys.argv[3],
                            use_averaging=True)
        else:
            raise ValueError("Arg 2 should be either 'logistic', either 'linear'.")

    elif sys.argv[1] == "real":
        for sto in [False, True]:
            for dataset in [sys.argv[2]]:
                run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid=sys.argv[4], algos=sys.argv[3],
                                use_averaging=True, scenario=None, fraction_sampled_workers=1, pp_strategy="pp1")
                run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid=sys.argv[4], algos=sys.argv[3],
                                use_averaging=True, fraction_sampled_workers=0.5, pp_strategy="pp1")
                run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid=sys.argv[4], algos=sys.argv[3],
                                use_averaging=True, fraction_sampled_workers=0.5, pp_strategy="pp2")
                run_experiments(nb_devices=20, stochastic=sto, dataset=dataset, iid=sys.argv[4], algos="artemis-vs-existing",
                                use_averaging=True, scenario=None, fraction_sampled_workers=1, pp_strategy="pp1")
        run_experiments(nb_devices=20, stochastic=True, dataset=sys.argv[2], iid=sys.argv[4], algos=sys.argv[3],
                        use_averaging=True, fraction_sampled_workers=1, pp_strategy="pp1", scenario="step")

