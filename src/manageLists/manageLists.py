"""
* *IntelComp H2020 project*

Contains the class implementing the functionality required
by the Interactive Model Trainer for creating lists of
* keywords
* equivalent terms
* stopwords
"""

import sys
import shutil
import json
import argparse
from pathlib import Path
import datetime as DT


class ListManager(object):
    """
    Main class to manage functionality for the creation, edition, etc 
    of lists of stopwords/keywords/equivalent_terms
    """

    def listWordLists(self, path_wordlists):
        """
        Returns a dictionary with all wordlists available in the folder 

        Parameters
        ----------
        path_wordlists : pathlib.Path
            Path to the folder hosting the wordlists

        Returns
        -------
        allWdLists : Dictionary (path -> dictionary)
            One dictionary entry per wordlist
            key is the absolute path to the wordlist
            value is a dictionary with metadata
        """
        allWdLists = {}
        jsonfiles = [el for el in path_wordlists.iterdir() if el.suffix == '.json']

        for WdList in jsonfiles:
            with open(WdList, 'r', encoding='utf8') as fin:
                allWdLists[WdList.resolve().as_posix()] = json.load(fin)

        return allWdLists

    def createList(self, path_wordlists, WdList):
        """
        Saves a (logical) training dataset in the indicated dataset folder 

        Parameters
        ----------
        path_wordlists : pathlib.Path
            Path to the folder hosting the wordlists

        WdList :
            Dictionary with WordList

        Returns
        -------
        status: int
            - 0 if the wordlist could not be created
            - 1 if the wordlist was created successfully
            - 2 if the wordlist replaced an existing one
        """

        if not path_wordlists.is_dir():
            return 0
        else:
            # Add current date to wordlist creation
            WdList['creation_date'] = DT.datetime.now()
            path_WdList = path_wordlists.joinpath(WdList['name'] + '.json')
            if path_WdList.is_file():
                # Create backup of existing list
                path_old = path_wordlists.joinpath(WdList['name'] + '.json.old')
                shutil.move(path_WdList, path_old)
                with path_WdList.open('w', encoding='utf-8') as fout:
                    json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
                return 2
            else:
                with path_WdList.open('w', encoding='utf-8') as fout:
                    json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
                return 1

    def deleteWordList(self, path_WdList):
        """
        Deletes a wordlist

        Parameters
        ----------
        path_WdList : pathlib.Path
            Path to the json file with the wordlist information

        Returns
        -------
        status : int
            - 0 if the wordlist could not be deleted
            - 1 if the wordlist was deleted successfully
        """

        if not path_WdList.is_file():
            return 0
        else:
            try:
                path_WdList.unlink()
                return 1
            except:
                return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scripts for List Management Service')
    parser.add_argument('--listWordLists', action='store_true', default=False,
                        help='List Available WordLists')
    parser.add_argument('--createWordList', action='store_true', default=False,
                        help='Save Training Dataset')
    parser.add_argument('--path_wordlists', type=str, default=None,
                        help="path to project wordlists")
    parser.add_argument('--deleteWordList', action='store_true', default=False,
                        help='Delete a wordlist')
    parser.add_argument('--path_WdList', type=str, default=None,
                        help="path to wordlist that will be deleted")
    
    """parser.add_argument('--saveTrDtset', action='store_true', default=False,
                        help='Save Training Dataset')
    parser.add_argument('--listTrDtsets', action='store_true', default=False,
                        help='List Training Datasets')
    
    parser.add_argument('--parquet', type=str, default=None,
                        help="path to downloaded parquet datasets")
    parser.add_argument('--path_datasets', type=str, default=None,
                        help="path to project datasets")
    """
    args = parser.parse_args()

    lm = ListManager()

    if args.listWordLists:
        if not args.path_wordlists:
            sys.exit('You need to indicate the location of wordlists')

        allWdLists = lm.listWordLists(Path(args.path_wordlists))
        sys.stdout.write(json.dumps(allWdLists))

    if args.createWordList:
        if not args.path_wordlists:
            sys.exit('You need to indicate the location of training datasets')
        WdList = [line for line in sys.stdin][0]
        WdList = json.loads(WdList.replace('\\"', '"'))

        status = lm.createList(Path(args.path_wordlists), WdList)
        sys.stdout.write(str(status))

    if args.deleteWordList:
        if not args.path_WdList:
            sys.exit('You need to indicate the location of the wordlist that will be deleted')

        status = lm.deleteWordList(Path(args.path_WdList))
        sys.stdout.write(str(status))
