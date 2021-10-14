"""
Created by Philippenko, 2sd August 2021.
"""
from src.utils.Utilities import get_project_root


def get_path_to_datasets() -> str:
    """Return the path to the datasets. For sake of anonymization, the path to datasets on clusters is not keep on GitHub
    and must be personalized locally"""
    return get_project_root()

def get_path_to_pickle() -> str:
    return get_project_root()