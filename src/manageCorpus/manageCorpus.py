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
        allDtsets = {}
        for Dts in path_parquet.iterdir():
            metafile = Dts.joinpath('datasetMeta.json')
            if metafile.exists():
                with open(metafile, 'r', encoding='utf8') as fin:
                    allDtsets[Dts.resolve().as_posix()] = json.load(fin)

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scripts for Corpus Management Service')
    parser.add_argument('--listDownloaded', action='store_true', default=False,
                        help='List datasets downloaded from HDFS with metadata.')
    parser.add_argument('--saveTrDtset', action='store_true', default=False,
                        help='Save Training Dataset')
    parser.add_argument('--listTrDtsets', action='store_true', default=False,
                        help='List Training Datasets')
    parser.add_argument('--deleteTrDtset', action='store_true', default=False,
                        help='Delete a Training Dataset')
    parser.add_argument('--parquet', type=str, default=None,
                        help="path to downloaded parquet datasets")
    parser.add_argument('--path_datasets', type=str, default=None,
                        help="path to project datasets")
    parser.add_argument('--path_TrDtset', type=str, default=None,
                        help="path to Training dataset that will be deleted")
    args = parser.parse_args()

    cm = CorpusManager()

    if args.listDownloaded:
        if not args.parquet:
            sys.exit('You need to indicate the location of downloaded datasets')

        allDtsets = cm.listDownloaded(Path(args.parquet))
        sys.stdout.write(json.dumps(allDtsets))

    if args.saveTrDtset:
        if not args.path_datasets:
            sys.exit('You need to indicate the location of training datasets')
        Dtset = [line for line in sys.stdin][0]
        Dtset = json.loads(Dtset.replace('\\"', '"'))

        status = cm.saveTrDtset(Path(args.path_datasets), Dtset)
        sys.stdout.write(str(status))

    if args.listTrDtsets:
        if not args.path_datasets:
            sys.exit('You need to indicate the location of training datasets')

        allTrDtsets = cm.listTrDtsets(Path(args.path_datasets))
        sys.stdout.write(json.dumps(allTrDtsets))

    if args.deleteTrDtset:
        if not args.path_TrDtset:
            sys.exit('You need to indicate the location of the training dataset that will be deleted')

        status = cm.deleteTrDtset(Path(args.path_TrDtset))
        sys.stdout.write(str(status))
