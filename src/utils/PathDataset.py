"""
Created by Philippenko, 2sd August 2021.

Customization required to run the code on a remote cluster.
"""
from src.utils.Utilities import get_project_root


def get_path_to_datasets() -> str:
    """Return the path to the datasets. For sake of anonymization, the path to datasets on clusters is not keep on
    GitHub and must be personalized locally"""
    print("Path to dataset: {0}".format(get_project_root() + "/../DATASETS"))
    return get_project_root() + "/../DATASETS"

def get_path_to_pickle() -> str:
    """"Return the path to the pickle folder. """
    return get_project_root()