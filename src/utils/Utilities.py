"""
Created by Philippenko, 8th June 2020.

In this python file, we put all utilities function not related with the proper run.
"""

import pickle
import os

from math import sqrt, log
from src.machinery.Parameters import Parameters
import pandas as pd

from pympler import tracker


def number_of_bits_needed_to_communicates_compressed(nb_devices: int, s: int, d: int) -> int:
    """Computing the theoretical number of bits used for a single way when using compression (with Elias encoding)."""
    frac = 2*(s**2+d) / (s * (s+sqrt(d)))
    return nb_devices * (3 + 3/2) * log(frac) * s * (s + sqrt(d)) + 32


def number_of_bits_needed_to_communicates_no_compressed(nb_devices:int, d: int) -> int:
    """Computing the theoretical number of bits used for a single way when using compression (with Elias encoding)."""
    return nb_devices * d * 32


def compute_number_of_bits(type_params: Parameters, nb_epoch: int, compress_model: bool):
    """Computing the theoretical number of bits used by an algorithm (with Elias encoding)."""
    # Initialization, the first element needs to be removed at the end.
    number_of_bits = [0]
    nb_devices = type_params.nb_devices
    d = type_params.n_dimensions
    fraction = type_params.fraction_sampled_workers
    for i in range(nb_epoch):
        nb_bits = 0
        if type_params.up_compression_model.omega_c != 0:
            s = type_params.up_compression_model.level
            nb_bits += number_of_bits_needed_to_communicates_compressed(nb_devices, s, d) * fraction
        else:
            nb_bits += number_of_bits_needed_to_communicates_no_compressed(nb_devices, d) * fraction
        if type_params.down_compression_model.omega_c != 0:
            s = type_params.down_compression_model.level
            nb_bits += number_of_bits_needed_to_communicates_compressed(nb_devices, s, d) * [1, fraction][compress_model]
        else:
            nb_bits += number_of_bits_needed_to_communicates_no_compressed(nb_devices, d)

        number_of_bits.append(nb_bits + number_of_bits[-1])
    return number_of_bits[1:]


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

def get_project_root() -> str:
    import pathlib
    path = str(pathlib.Path().absolute())
    if not path.find("artemis"):
        raise ValueError("Current directory looks to be higher than root of the project: {}".format(path))
    split = path.split("artemis")
    return split[0] + "artemis"


def create_folder_if_not_existing(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def file_exist(filename: str):
    return os.path.isfile(filename)


def check_memory_usage():

    mem = tracker.SummaryTracker()
    memory = pd.DataFrame(mem.create_summary(), columns=['object', 'number_of_objects', 'memory'])
    memory['mem_per_object'] = memory['memory'] / memory['number_of_objects']
    print(memory.sort_values('memory', ascending=False).head(10))
    print("============================================================")
    print(memory.sort_values('mem_per_object', ascending=False).head(10))


def drop_nan_values(values):
    return [x for x in values if str(x) != 'nan']


def keep_until_found_nan(values):
    result = []
    k = 0
    while str(values[k]) != 'nan' and k < len(values) - 1:
        print(values[k])
        result.append(values[k])
        k+=1
    return result
