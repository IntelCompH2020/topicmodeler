"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides a series of auxiliary functions for creation and management of topic models.
"""

import os
from pathlib import Path

import pickle


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return 0


def file_lines(fname):
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
        the file whose number of lines is calculated

    Returns
    -------
    number of lines
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def look_for_path(tm_path, path_name):
    """
    Given the path ("tm_path") to the TMmodels folder, if a model with the name "path_name" is located at the root of the TMmodels folder, such a path is returned; otherwise, the topic model represented by "path_name" is a submodel and its path is recursively searched within the TMmodels folder; once it is found, such a path is returned

    Parameters
    ----------
    tm_path: Path
        Path to the TMmodels folder
    path_name: str
        Name of the topic model being looked for
    Returns
    -------
    tm_path: Path
        Path to the searched topic model
    """

    if tm_path.joinpath(path_name).is_dir():
        return tm_path
    else:
        for root, dirs, files in os.walk(tm_path):
            for dir in dirs:
                if dir.endswith(path_name):
                    tm_path = Path(os.path.join(root, dir)).parent
        return tm_path
