"""
* *IntelComp H2020 project*

Contains the class implementing the functionality required
by the Interactive Model Trainer for creating training datasets
that can be used for topic modeling, domain classification, etc

"""

import sys
import shutil
import json
import argparse
from pathlib import Path
import datetime as DT


class CorpusManager(object):
    """
    Main class to manage functionality for the creation and export
    of training datasets
    """

    def listDownloaded(self, path_parquet):
        """
        Returns a dictionary with all datasets downloaded from the Data Catalogue 

        Parameters
        ----------
        path_parquet : pathlib.Path
            Path to the folder hosting the parquet datasets

        Returns
        -------
        allDtsets : Dictionary (path -> dictionary)
            One dictionary entry per dataset
            key is the absolute path to the dataset
            value is a dictionary with metadata
        """
        metafile = path_parquet.joinpath('datasetMeta.json')
        with open(metafile, 'r', encoding='utf8') as fin:
            allDtsets = json.load(fin)
        allDtsets = {path_parquet.joinpath(Dts).resolve().as_posix(): allDtsets[Dts]
                        for Dts in allDtsets.keys()}

        return allDtsets

    def listTrDtsets(self, path_dataset):
        """
        Returns a dictionary with all datasets downloaded from the Data Catalogue 

        Parameters
        ----------
        path_dataset : pathlib.Path
            Path to the folder hosting the training datasets

        Returns
        -------
        allTrDtsets : Dictionary (path -> dictionary)
            One dictionary entry per dataset
            key is the absolute path to the dataset
            value is a dictionary with metadata
        """
        allTrDtsets = {}
        jsonfiles = [el for el in path_dataset.iterdir() if el.suffix == '.json']

        for TrDts in jsonfiles:
            with open(TrDts, 'r', encoding='utf8') as fin:
                allTrDtsets[TrDts.resolve().as_posix()] = json.load(fin)

        return allTrDtsets

    def saveTrDtset(self, path_datasets, Dtset):
        """
        Saves a (logical) training dataset in the indicated dataset folder 

        Parameters
        ----------
        path_dataset : pathlib.Path
            Path to the folder hosting the training datasets

        Dtset :
            Dictionary with Training Dataset information

        Returns
        -------
        status: int
            - 0 if the dataset could not be created
            - 1 if the dataset was created successfully
            - 2 if the dataset replaced an existing dataset
        """

        if not path_datasets.is_dir():
            return 0
        else:
            # Add current date to Dtset creation
            Dtset['creation_date'] = DT.datetime.now()
            path_Dtset = path_datasets.joinpath(Dtset['name'] + '.json')
            if path_Dtset.is_file():
                # Copy current json file to the backup folder
                path_old = path_datasets.joinpath(Dtset['name'] + '.json.old')
                shutil.move(path_Dtset, path_old)
                with path_Dtset.open('w', encoding='utf-8') as fout:
                    json.dump(Dtset, fout, ensure_ascii=False, indent=2, default=str)
                return 2
            else:
                with path_Dtset.open('w', encoding='utf-8') as fout:
                    json.dump(Dtset, fout, ensure_ascii=False, indent=2, default=str)
                return 1

    def deleteTrDtset(self, path_TrDtset):
        """
        Deletes a (logical) training dataset 

        Parameters
        ----------
        path_TrDtset : pathlib.Path
            Path to the json file with the training dataset information

        Returns
        -------
        status : int
            - 0 if the dataset could not be deleted
            - 1 if the dataset was deleted successfully
        """

        if not path_TrDtset.is_file():
            return 0
        else:
            try:
                path_TrDtset.unlink()
                return 1
            except:
                return 0

    def renameTrDtset(self, name: Path, new_name: Path):
        """
        Renames a dataset

        Parameters
        ----------
        name : pathlib.Path
            Path to the json file to be renamed

        new_name : pathlib.Path
            Path to the new name for the json file

        Returns
        -------
        status : int
            - 0 if the dataset could not be renamed
            - 1 if the dataset was renamed successfully

        """

        if not name.is_file():
            print(f"File '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(
                f"File '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            with name.open("r", encoding="utf8") as fin:
                Dtset = json.load(fin)
            Dtset["name"] = new_name.stem
            with new_name.open("w", encoding="utf-8") as fout:
                json.dump(Dtset, fout, ensure_ascii=False, indent=2, default=str)
            name.unlink()
            return 1
        except:
            return 0

    def copyTrDtset(self, name: Path):
        """
        Creates a copy of a wordlist

        Parameters
        ----------
        name : pathlib.Path
            Path to the json file to be copied

        Returns
        -------
        status : int
            - 0 if the wordlist could not be renamed
            - 1 if the wordlist was renamed successfully

        """

        if not name.is_file():
            print(f"File '{name.as_posix()}' does not exist.")
            return 0
        try:
            path_copy = name.with_name(f"{name.stem}-copy.json")
            shutil.copy(name, path_copy)
            with path_copy.open("r", encoding="utf8") as fin:
                Dtset = json.load(fin)
            Dtset["name"] = path_copy.stem
            with path_copy.open("w", encoding="utf-8") as fout:
                json.dump(Dtset, fout, ensure_ascii=False, indent=2, default=str)
            return 1
        except:
            return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scripts for Corpus Management Service')
    parser.add_argument("--path_downloaded", type=str, default=None, required=True,
                        metavar=("path_to_datasets"),
                        help="path to downloaded datasets")
    parser.add_argument("--path_datasets", type=str, default=None, required=True,
                        metavar=("path_to_datasets"),
                        help="path to training datasets")
    parser.add_argument('--listDownloaded', action='store_true', default=False,
                        help='List datasets downloaded from HDFS with metadata.')
    parser.add_argument('--saveTrDtset', action='store_true', default=False,
                        help='Save Training Dataset')
    parser.add_argument('--listTrDtsets', action='store_true', default=False,
                        help='List Training Datasets')
    parser.add_argument('--deleteTrDtset', type=str, default=None,
                        metavar=("filename"),
                        help='Delete a Training Dataset')
    parser.add_argument("--renameTrDtset", type=str, default=None, nargs=2,
                        metavar=("filename", "new_filename"),
                        help="Rename wordlist with selected name to new name")
    parser.add_argument("--copyTrDtset", type=str, default=None,
                        metavar=("filename"),
                        help="Make a copy of wordlist with selected name")
    args = parser.parse_args()

    cm = CorpusManager()

    dwds_path = Path(args.path_downloaded)
    trds_path = Path(args.path_datasets)

    if args.listDownloaded:
        allDtsets = cm.listDownloaded(dwds_path)
        sys.stdout.write(json.dumps(allDtsets))

    if args.saveTrDtset:
        Dtset = [line for line in sys.stdin][0]
        Dtset = json.loads(Dtset.replace('\\"', '"'))

        status = cm.saveTrDtset(trds_path, Dtset)
        sys.stdout.write(str(status))

    if args.listTrDtsets:
        allTrDtsets = cm.listTrDtsets(trds_path)
        sys.stdout.write(json.dumps(allTrDtsets))

    if args.deleteTrDtset:
        status = cm.deleteTrDtset(trds_path.joinpath(f"{args.deleteTrDtset}.json"))
        sys.stdout.write(str(status))
    
    if args.renameTrDtset:
        status = cm.renameTrDtset(
            trds_path.joinpath(f"{args.renameTrDtset[0]}.json"),
            trds_path.joinpath(f"{args.renameTrDtset[1]}.json"),
        )
        sys.stdout.write(str(status))

    if args.copyTrDtset:
        status = cm.copyTrDtset(trds_path.joinpath(f"{args.copyTrDtset}.json"))
        sys.stdout.write(str(status))
