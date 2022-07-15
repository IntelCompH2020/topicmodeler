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

    def listWordLists(self, path_wordlists: Path):
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
        jsonfiles = [el for el in path_wordlists.iterdir() if el.suffix == ".json"]

        for WdList in jsonfiles:
            with open(WdList, "r", encoding="utf8") as fin:
                allWdLists[WdList.resolve().as_posix()] = json.load(fin)

        return allWdLists

    def createWordList(self, path_wordlists: Path, WdList: dict):
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
            print(f"Directory '{path_wordlists.as_posix()}' does not exist.")
            return 0
        else:
            # Add current date to wordlist creation
            WdList["creation_date"] = DT.datetime.now()
            path_WdList = path_wordlists.joinpath(WdList["name"] + ".json")
            if path_WdList.is_file():
                # Create backup of existing list
                path_old = path_wordlists.joinpath(WdList["name"] + ".json.old")
                shutil.move(path_WdList, path_old)
                with path_WdList.open("w", encoding="utf-8") as fout:
                    json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
                return 2
            else:
                with path_WdList.open("w", encoding="utf-8") as fout:
                    json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
                return 1

    def deleteWordList(self, path_WdList: Path):
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
            print(f"File '{path_WdList.as_posix()}' does not exist.")
            return 0
        else:
            try:
                path_WdList.unlink()
                return 1
            except:
                return 0

    def renameWordList(self, name: Path, new_name: Path):
        """
        Renames a wordlist

        Parameters
        ----------
        name : pathlib.Path
            Path to the json file to be renamed
        
        new_name : pathlib.Path
            Path to the new name for the json file

        Returns
        -------
        status : int
            - 0 if the wordlist could not be renamed
            - 1 if the wordlist was renamed successfully
        
        """

        if not name.is_file():
            print(f"File '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(f"File '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            with name.open("r", encoding="utf8") as fin:
                WdList = json.load(fin)
            WdList["name"] = new_name.stem
            with new_name.open("w", encoding="utf-8") as fout:
                json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
            name.unlink()
            return 1
        except:
            return 0
    
    def copyWordList(self, name: Path):
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
                WdList = json.load(fin)
            WdList["name"] = path_copy.stem
            with path_copy.open("w", encoding="utf-8") as fout:
                json.dump(WdList, fout, ensure_ascii=False, indent=2, default=str)
            return 1
        except:
            return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scripts for List Management Service")
    parser.add_argument("--path_wordlists", type=str, default=None, required=True,
                        metavar=("path_to_wordlists"),
                        help="path to project wordlists")
    parser.add_argument("--listWordLists", action="store_true", default=False,
                        help="List Available WordLists")
    parser.add_argument("--createWordList", action="store_true", default=False,
                        help="Save wordlist")
    parser.add_argument("--deleteWordList", type=str, default=None,
                        metavar=("filename"),
                        help="Delete wordlist with selected name")
    parser.add_argument("--renameWordList", type=str, default=None, nargs=2,
                        metavar=("filename", "new_filename"),
                        help="Rename wordlist with selected name to new name")
    parser.add_argument("--copyWordList", type=str, default=None,
                        metavar=("filename"),
                        help="Make a copy of wordlist with selected name")

    args = parser.parse_args()

    lm = ListManager()

    wl_path = Path(args.path_wordlists)

    if args.listWordLists:
        allWdLists = lm.listWordLists(wl_path)
        sys.stdout.write(json.dumps(allWdLists))

    if args.createWordList:
        # WdList = [line for line in sys.stdin][0]
        WdList = "".join([line for line in sys.stdin])
        WdList = json.loads(WdList.replace('\\"', '"'))

        status = lm.createWordList(wl_path, WdList)
        sys.stdout.write(str(status))

    if args.deleteWordList:
        status = lm.deleteWordList(wl_path.joinpath(f"{args.deleteWordList}.json"))
        sys.stdout.write(str(status))

    if args.renameWordList:
        status = lm.renameWordList(
            wl_path.joinpath(f"{args.renameWordList[0]}.json"),
            wl_path.joinpath(f"{args.renameWordList[1]}.json"),
        )
        sys.stdout.write(str(status))

    if args.copyWordList:
        status = lm.copyWordList(wl_path.joinpath(f"{args.copyWordList}.json"))
        sys.stdout.write(str(status))
