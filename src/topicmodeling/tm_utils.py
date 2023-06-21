"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides a series of auxiliary functions for creation and management of topic models.
"""

import os
from pathlib import Path
import pathlib

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
    
def pickler_avitm_for_ewb_inferencer(path_model_infer:pathlib.Path,
                                     avitm_model):
    """
    Pickle the AVITM model and the corpus for the EWB Inferencer

    Parameters
    ----------
    path_model_infer: pathlib.Path
        Path to the AVITM model to be pickled
    avitm_model: AVITM
        AVITM model
    
    Returns
    -------
    0
    """

    # Avitm instance
    train_data = avitm_model.__dict__['train_data'].__dict__
    validation_data = avitm_model.__dict__['validation_data'].__dict__
    early_stopping = avitm_model.__dict__['early_stopping'].__dict__
    decoder = avitm_model.__dict__['model']
    rest_avitm = avitm_model.__dict__
    entries_to_remove = ('early_stopping', 'train_data', 'validation_data', 'model')
    for k in entries_to_remove:
        rest_avitm.pop(k, None)

    # Decoder and encoder instances
    _modules = decoder.__dict__['_modules']
    inf_net = _modules['inf_net'].__dict__
    rest_decoder = decoder.__dict__
    rest_decoder['_modules'].pop("inf_net", None)
    
    all_data = [train_data, validation_data, early_stopping, rest_avitm, rest_decoder, inf_net]
    
    pickler(path_model_infer, all_data)
    
    return 0
