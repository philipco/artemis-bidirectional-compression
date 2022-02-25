import os
import pickle



# def rebuild_pickle_by_hyperparameters(algos_pickle_path, scenario, experiments_settings):
#
#     descent_by_algo_and_hyperparameters = pickle_loader("{0}/{1}/descent_by_algo-{2}".format(algos_pickle_path,
#                                                                                        scenario,
#                                                                                        experiments_settings))
#
#     all_descent_various_gamma = {i: {} for i in TIMESTAMP}
#     for algo, res_by_algo_and_hyperparameters in descent_by_algo_and_hyperparameters.keys():
#         losses_by_algo = {i: [] for i in TIMESTAMP}
#         losses_avg_by_algo = {i: [] for i in TIMESTAMP}
#         norm_ef_by_algo = {i: [] for i in TIMESTAMP}
#         dist_model_by_algo = {i: [] for i in TIMESTAMP}
#         h_i_to_optimal_grad_by_algo = {i: [] for i in TIMESTAMP}
#         avg_h_i_to_optimal_grad_by_algo = {i: [] for i in TIMESTAMP}
#         var_models_by_algo = {i: [] for i in TIMESTAMP}
#         descent_by_hyperparameters = {}
#         # for (value, label) in zip(xvalues, xlabels):



def pickle_saver(data, filename: str) -> None:
    """Save a python object into a pickle file.
    If a file with the same name already exists, remove it.
    Store the file into a folder pickle/ which need to already exist.
    Args:
        data: the python object to save.
        filename: the filename where the object is saved.
    """
    file_to_save = "{0}.pkl".format(filename)
    if os.path.exists(file_to_save):
        os.remove(file_to_save)
    pickle_out = open(file_to_save, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_loader(filename: str):
    """Load a python object saved with pickle.
    Args:
        filename: the file where the object is stored.
    Returns:
        The python object to load.
    """
    pickle_in = open("{0}.pkl".format(filename), "rb")
    return pickle.load(pickle_in)