"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides a series of functions for Topic Model representation and curation
"""

import argparse
import json
import sys
import shutil

from pathlib import Path
import numpy as np
import scipy.sparse as sparse


class TMManager(object):
    """
    Main class to manage functionality for the management of topic models
    """

    def listTMmodels(self, path_TMmodels: Path):
        """
        Returns a dictionary with all topic models

        Parameters
        ----------
        path_TMmodels : pathlib.Path
            Path to the folder hosting the topic models

        Returns
        -------
        allTMmodels : Dictionary (path -> dictionary)
            One dictionary entry per wordlist
            key is the topic model name
            value is a dictionary with metadata
        """
        allTMmodels = {}
        modelFolders = [el for el in path_TMmodels.iterdir()]

        for TMf in modelFolders:
            modelConfig = TMf.joinpath('trainconfig.json')
            if modelConfig.is_file():
                with modelConfig.open('r', encoding='utf8') as fin:
                    modelInfo = json.load(fin)
                    allTMmodels[modelInfo['name']] = {
                        "name": modelInfo['name'],
                        "description": modelInfo['description'],
                        "visibility": modelInfo['visibility'],
                        "trainer": modelInfo['trainer'],
                        "TrDtSet": modelInfo['TrDtSet'],
                        "creation_date": modelInfo['creation_date'],
                        "hierarchy-level": modelInfo['hierarchy-level'],
                        "htm-version": modelInfo['htm-version']
                    }
                submodelFolders = [el for el in TMf.iterdir() if not el.as_posix().endswith("modelFiles") and not el.as_posix().endswith("corpus.parquet") and not el.as_posix().endswith("_old")]
                for sub_TMf in submodelFolders:
                    submodelConfig = sub_TMf.joinpath('trainconfig.json')
                    if submodelConfig.is_file():
                        with submodelConfig.open('r', encoding='utf8') as fin:
                            submodelInfo = json.load(fin)
                            corpus = "Subcorpus created from " + str(modelInfo['name'])
                            allTMmodels[submodelInfo['name']] = {
                                "name": submodelInfo['name'],
                                "description": submodelInfo['description'],
                                "visibility": submodelInfo['visibility'],
                                "trainer": submodelInfo['trainer'],
                                "TrDtSet": corpus,
                                "creation_date": submodelInfo['creation_date'],
                                "hierarchy-level": submodelInfo['hierarchy-level'],
                                "htm-version": submodelInfo['htm-version']
                            }
        return allTMmodels

    def deleteTMmodel(self, path_TMmodel: Path):
        """
        Deletes a Topic Model

        Parameters
        ----------
        path_TMmodel : pathlib.Path
            Path to the folder containing the Topic Model

        Returns
        -------
        status : int
            - 0 if the Topic Model could not be deleted
            - 1 if the Topic Model was deleted successfully
        """

        if not path_TMmodel.is_dir():
            print(f"File '{path_TMmodel.as_posix()}' does not exist.")
            return 0
        else:
            try:
                shutil.rmtree(path_TMmodel)
                return 1
            except:
                return 0

    def renameTMmodel(self, name: Path, new_name: Path):
        """
        Renames a topic model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be renamed
        
        new_name : pathlib.Path
            Path to the new name for the topic model

        Returns
        -------
        status : int
            - 0 if the model could not be renamed
            - 1 if the model was renamed successfully
        
        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            with name.joinpath('trainconfig.json').open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with name.joinpath('trainconfig.json').open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False, indent=2, default=str)
            shutil.move(name, new_name)
            return 1
        except:
            return 0

    def copyTMmodel(self, name: Path, new_name: Path):
        """
        Makes a copy of an existing topic model

        Parameters
        ----------
        name : pathlib.Path
            Path to the model to be copied
        
        new_name : pathlib.Path
            Path to the new name for the topic model

        Returns
        -------
        status : int
            - 0 if the model could not be copied
            - 1 if the model was copied successfully
        
        """
        if not name.is_dir():
            print(f"Model '{name.as_posix()}' does not exist.")
            return 0
        if new_name.is_file():
            print(f"Model '{new_name.as_posix()}' already exists. Rename or delete it first.")
            return 0
        try:
            shutil.copytree(name, new_name)
            with new_name.joinpath('trainconfig.json').open("r", encoding="utf8") as fin:
                TMmodel = json.load(fin)
            TMmodel["name"] = new_name.stem
            with new_name.joinpath('trainconfig.json').open("w", encoding="utf-8") as fout:
                json.dump(TMmodel, fout, ensure_ascii=False, indent=2, default=str)
            return 1
        except:
            return 0


class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # and needs to be backed up with a folder in which all the associated
    # files will be saved
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox
    # that produces a model according to this representation

    # The following variables will store original values of matrices alphas, betas, thetas
    # They will be used to reset the model to original values

    _TMfolder = None

    _betas_orig = None
    _thetas_orig = None
    _alphas_orig = None

    _betas = None
    _thetas = None
    _alphas = None
    _edits = None  # Store all editions made to the model
    _ntopics = None
    _betas_ds = None
    _topic_entropy = None
    _descriptions = None
    _vocab_w2id = None
    _vocab_id2w = None
    _vocab = None
    _size_vocab = None

    def __init__(self, TMfolder, logger=None):
        """Class initializer

        We just need to make sure that we have a folder where the
        model will be stored. If the folder does not exist, it will
        create a folder for the model

        Parameters
        ----------
        TMfolder: Path
            Contains the name of an existing folder or a new folder
            where the model will be created
        logger:
            External logger to use. If None, a logger will be created for the object
        """
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('TMmodel')

        # Convert strings to Paths if necessary
        self._TMfolder = Path(TMfolder)

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            try:
                self._TMfolder.mkdir(parents=True)
            except:
                self._logger.error('-- -- Topic model object (TMmodel) could not be created')

        self._logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')

    def create(self, betas=None, thetas=None, alphas=None, vocab=None):
        """Create model from the relevant matrices

        Inicializacion del model de topicos a partir de las matrices que lo caracterizan

        Ademas de inicializar las correspondientes variables del objeto, se van a calcular
        todas las variables asociadas y visualizaciones cuyo cómputo requiere tiempo
        para que estén disponibles para el resto de métodos

        Parameters
        ----------
        betas:
            Matriz numpy de tamaño n_topics x n_words (vocab de cada tópico)
        thetas:
            Matriz numpy de tamaño n_docs x n_topics (composición documental)
        alphas:
            Vector de longitud n_topics, con la importancia de cada perfil
        vocab:
            Vocabulary. List of words sorted according to betas matrix
        """

        # If folder already exists no further action is needed
        # in other case, the folder is created
        if not self._TMfolder.is_dir():
            self._logger.error('-- -- Topic model object (TMmodel) folder not ready')
            return

        self._alphas_orig = alphas
        self._betas_orig = betas
        self._thetas_orig  = thetas
        self._alphas = alphas
        self._betas = betas
        self._thetas = thetas
        self._vocab = vocab
        self._size_vocab = len(vocab)
        self._ntopics = thetas.shape[1]
        self._edits = []

        # Save original variables
        np.save(self._TMfolder.joinpath('alphas_orig.npy'), alphas)
        np.save(self._TMfolder.joinpath('betas_orig.npy'), betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas_orig.npz'), thetas)
        with self._TMfolder.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(vocab))

        # Initial sort of topics according to size. Save rearranged matrices
        self._sort_topics()
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)
        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))

        # Calculate and save additional related variables
        self._calculate_beta_ds()
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        self._calculate_topic_entropy()
        np.save(self._TMfolder.joinpath('topic_entropy.npy'), self._topic_entropy)

        self._logger.info(
            '-- -- Topic model variables saved to file')
        return

    def _sort_topics(self):
        """Sort topics according to topic size"""

        # Load information if necessary
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_edits() 

        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        # Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]

        return

    def _load_alphas(self):
        if self._alphas is None:
            self._alphas = np.load(self._TMfolder.joinpath('alphas.npy'))

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(self._TMfolder.joinpath('thetas.npz'))

    def _load_edits(self):
        if self._edits is None:
            with self._TMfolder.joinpath('edits.txt').open('r', encoding='utf8') as fin:
                self._edits = fin.readlines()

    def _calculate_beta_ds(self):
        """Calculates beta with down-scoring
        Emphasizes words appearing less frequently in topics
        """
        # Load information if necessary
        self._load_betas()
        
        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) /
                          self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)

    def _calculate_topic_entropy(self):
        """Calculates the entropy of all topics in model
        """
        # Load information if necessary
        self._load_betas()

        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = - \
            np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)


##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scripts for Topic Modeling Service")
    parser.add_argument("--path_TMmodels", type=str, default=None, required=True,
                        metavar=("path_to_TMs"),
                        help="path to topic models folder")
    parser.add_argument("--listTMmodels", action="store_true", default=False,
                        help="List Available Topic Models")
    parser.add_argument("--deleteTMmodel", type=str, default=None,
                        metavar=("modelName"),
                        help="Delete Topic Model with selected name")
    parser.add_argument("--renameTM", type=str, default=None, nargs=2,
                        metavar=("modelName", "new_modelName"),
                        help="Rename Topic Model with selected name to new name")
    parser.add_argument("--copyTM", type=str, default=None, nargs=2,
                        metavar=("modelName", "new_modelName"),
                        help="Make a copy of Topic Model")

    args = parser.parse_args()

    tmm = TMManager()

    tm_path = Path(args.path_TMmodels)

    if args.listTMmodels:
        allTMmodels = tmm.listTMmodels(tm_path)
        sys.stdout.write(json.dumps(allTMmodels))

    if args.deleteTMmodel:
        status = tmm.deleteTMmodel(tm_path.joinpath(f"{args.deleteTMmodel}"))
        sys.stdout.write(str(status))

    if args.renameTM:
        status = tmm.renameTMmodel(
            tm_path.joinpath(f"{args.renameTM[0]}"),
            tm_path.joinpath(f"{args.renameTM[1]}"),
        )
        sys.stdout.write(str(status))

    if args.copyTM:
        status = tmm.copyTMmodel(
            tm_path.joinpath(f"{args.copyTM[0]}"),
            tm_path.joinpath(f"{args.copyTM[1]}"),
        )
        sys.stdout.write(str(status))
