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
from sklearn.preprocessing import normalize
import pyLDAvis
from src.gui.utils import utils


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
    _ndocs_active = None
    _tpc_descriptions = None
    _tpc_labels = None
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
        """Creates the topic model from the relevant matrices that characterize it. In addition to the initialization of the corresponding object's variables, all the associated variables and visualizations which are computationally costly are calculated so they are available for the other methods.

        Parameters
        ----------
        betas:
            Matrix of size n_topics x n_words (vocab of each topic)
        thetas:
            Matrix of size  n_docs x n_topics (document composition)
        alphas: 
            Vector of length n_topics containing the importance of each topic
        vocab: list
            List of words sorted according to betas matrix
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

        # Initial sort of topics according to size. Calculate other variables
        self._sort_topics()
        self._calculate_beta_ds()
        self._calculate_topic_entropy()
        self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])
        self._tpc_descriptions = [el[1] for el in self.get_tpc_word_descriptions()]
        self._tpc_labels = [el[1] for el in self.get_tpc_labels()]
        
        # We are ready to save all variables in the model
        self._save_all()

        self._logger.info(
            '-- -- Topic model variables were computed and saved to file')
        return

    def _save_all(self):
        """Saves all variables in Topic Model
        * alphas, betas, thetas
        * edits
        * betas_ds, topic_entropy, ndocs_active
        * tpc_descriptions, tpc_labels
        This function should only be called after making sure all these
        variables exist and are not None
        """
        np.save(self._TMfolder.joinpath('alphas.npy'), self._alphas)
        np.save(self._TMfolder.joinpath('betas.npy'), self._betas)
        sparse.save_npz(self._TMfolder.joinpath('thetas.npz'), self._thetas)
        with self._TMfolder.joinpath('edits.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._edits))
        np.save(self._TMfolder.joinpath('betas_ds.npy'), self._betas_ds)
        np.save(self._TMfolder.joinpath('topic_entropy.npy'), self._topic_entropy)
        np.save(self._TMfolder.joinpath('ndocs_active.npy'), self._ndocs_active)
        with self._TMfolder.joinpath('tpc_descriptions.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_descriptions))
        with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
            fout.write('\n'.join(self._tpc_labels))

        # Generate also pyLDAvisualization
        # We will compute the visualization using ndocs random documents
        ndocs = 10000
        if ndocs > self._thetas.shape[0]:
            ndocs = self._thetas.shape[0]
        perm = np.sort(np.random.permutation(self._thetas.shape[0])[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocabfreq = np.round(ndocs*(self._alphas.dot(self._betas))).astype(int)
        vis_data = pyLDAvis.prepare(self._betas, self._thetas[perm, ].toarray(),
                                    doc_len, self._vocab, vocabfreq, lambda_step=0.05,
                                    sort_topics=False, n_jobs=-1)
        pyLDAvis.save_html(vis_data, self._TMfolder.joinpath('pyLDAvis.html').as_posix())
        utils.modify_pyldavis_html(self._TMfolder.as_posix())
        
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
            self._ntopics = self._alphas.shape[0]

    def _load_betas(self):
        if self._betas is None:
            self._betas = np.load(self._TMfolder.joinpath('betas.npy'))
            self._ntopics = self._betas.shape[0]
            self._size_vocab = self._betas.shape[1]

    def _load_thetas(self):
        if self._thetas is None:
            self._thetas = sparse.load_npz(self._TMfolder.joinpath('thetas.npz'))
            self._ntopics = self._thetas.shape[1]
            #self._ndocs_active = np.array((self._thetas != 0).sum(0).tolist()[0])

    def _load_ndocs_active(self):
        if self._ndocs_active is None:
            self._ndocs_active = np.load(self._TMfolder.joinpath('ndocs_active.npy'))
            self._ntopics = self._ndocs_active.shape[0]

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

    def _load_betas_ds(self):
        if self._betas_ds is None:
            self._betas_ds = np.load(self._TMfolder.joinpath('betas_ds.npy'))
            self._ntopics = self._betas_ds.shape[0]
            self._size_vocab = self._betas_ds.shape[1]

    def _load_vocab(self):
        if self._vocab is None:
            with self._TMfolder.joinpath('vocab.txt').open('r', encoding='utf8') as fin:
                self._vocab = [el.strip() for el in fin.readlines()]

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
    
    def _load_topic_entropy(self):
        if self._topic_entropy is None:
            self._topic_entropy = np.load(self._TMfolder.joinpath('topic_entropy.npy'))

    def get_tpc_word_descriptions(self, n_words=15, tfidf=True, tpc=None):
        """returns the chemical description of topics

        Parameters
        ----------
        n_words:
            Number of terms for each topic that will be included
        tfidf:
            If true, downscale the importance of words that appear
            in several topics, according to beta_ds (Blei and Lafferty, 2009)
        tpc:
            Topics for which the descriptions will be computed, e.g.: tpc = [0,3,4]
            If None, it will compute the descriptions for all topics  

        Returns
        -------
        tpc_descs: list of tuples
            Each element is a a term (topic_id, "word0, word1, ...")                      
        """

        # Load betas (including n_topics) and vocabulary 
        if tfidf:
            self._load_betas_ds()
        else:
            self._load_betas()
        self._load_vocab()
        
        if not tpc:
            tpc = range(self._ntopics)
        
        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_words]]
            else:
                words = [self._vocab[idx2]
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_words]]
            tpc_descs.append((i, ', '.join(words)))
        return tpc_descs

    def load_tpc_descriptions(self):
        if self._tpc_descriptions is None:
            with self._TMfolder.joinpath('tpc_descriptions.txt').open('r', encoding='utf8') as fin:
                self._tpc_descriptions = [el.strip() for el in fin.readlines()]

    def get_tpc_labels(self):
        """returns the labels of the topics in the model

        Returns
        -------
        tpc_labels: list of tuples
            Each element is a a term (topic_id, "label for topic topic_id")

        -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        This functions needs to be implemented by JAEM
        -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*                      
        """
        return self.get_tpc_word_descriptions()

    def load_tpc_labels(self):
        if self._tpc_labels is None:
            with self._TMfolder.joinpath('tpc_labels.txt').open('r', encoding='utf8') as fin:
                self._tpc_labels = [el.strip() for el in fin.readlines()]

    def showTopics(self):
        self._load_alphas()
        self._load_ndocs_active()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        TpcsInfo = [(str(round(el[0],4)), el[1].strip(), el[2].strip(), str(el[3]))
                            for el in zip(self._alphas, self._tpc_labels,
                                self._tpc_descriptions, self._ndocs_active)]
        return TpcsInfo

    def setTpcLabels(self, TpcLabels):
        self._tpc_labels = [el.strip() for el in TpcLabels]
        self._load_alphas()
        #Check that the number of labels is consistent with model
        if len(TpcLabels)==self._ntopics:
            with self._TMfolder.joinpath('tpc_labels.txt').open('w', encoding='utf8') as fout:
                fout.write('\n'.join(self._tpc_labels))
            return
        else:
            return 0

    def deleteTopics(self, tpcs):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()

        try:
            # Get a list of the topics that should be kept
            tpc_keep = [k for k in range(self._ntopics) if k not in tpcs]
            tpc_keep = [k for k in tpc_keep if k < self._ntopics]

            # Calculate new variables
            self._thetas = self._thetas[:, tpc_keep]
            self._thetas = normalize(self._thetas, axis=1, norm='l1')
            self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
            self._ntopics = self._thetas.shape[1]
            self._betas = self._betas[tpc_keep, :]
            self._betas_ds = self._betas_ds[tpc_keep, :]
            self._ndocs_active = self._ndocs_active[tpc_keep]
            self._topic_entropy = self._topic_entropy[tpc_keep]
            self._tpc_labels = [self._tpc_labels[i] for i in tpc_keep]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in tpc_keep]
            self._edits.append('d ' + ' '.join([str(k) for k in tpcs]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics deletion successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics deletion generated an error. Operation failed')
            return 0

    def sortTopics(self):
        """This is a costly operation, almost everything
        needs to get modified"""
        self._load_alphas()
        self._load_betas()
        self._load_thetas()
        self._load_betas_ds()
        self._load_topic_entropy()
        self.load_tpc_descriptions()
        self.load_tpc_labels()
        self._load_ndocs_active()
        self._load_edits()

        try:
            # Calculate order for the topics
            idx = np.argsort(self._alphas)[::-1]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # Calculate new variables
            self._thetas = self._thetas[:, idx]
            self._alphas = self._alphas[idx]
            self._betas = self._betas[idx, :]
            self._betas_ds = self._betas_ds[idx, :]
            self._ndocs_active = self._ndocs_active[idx]
            self._topic_entropy = self._topic_entropy[idx]
            self._tpc_labels = [self._tpc_labels[i] for i in idx]
            self._tpc_descriptions = [self._tpc_descriptions[i] for i in idx]
            self._edits.append('s ' + ' '.join([str(el) for el in idx]))

            # We are ready to save all variables in the model
            self._save_all()

            self._logger.info(
                '-- -- Topics reordering successful. All variables saved to file')
            return 1
        except:
            self._logger.info(
                '-- -- Topics reordering generated an error. Operation failed')
            return 0

    def resetTM(self):
        self._alphas_orig = np.load(self._TMfolder.joinpath('alphas_orig.npy'))
        self._betas_orig = np.load(self._TMfolder.joinpath('betas_orig.npy'))
        self._thetas_orig = sparse.load_npz(self._TMfolder.joinpath('thetas_orig.npz'))
        self._load_vocab()

        try:
            self.create(betas=self._betas_orig, thetas=self._thetas_orig,
                        alphas=self._alphas_orig, vocab=self._vocab)
            return 1
        except:
            return 0

    def save_npz(self, npzfile):
        """Saves the matrices that characterizes the topic model inot numpy npz file.

        Parameters
        ----------
        npzfile: str
            Name of the file in which the model will be saved
        """

        if isinstance(self._thetas, sparse.csr_matrix):
            np.savez(npzfile, alphas=self._alphas, betas=self._betas,
                     thetas_data=self._thetas.data, thetas_indices=self._thetas.indices,
                     thetas_indptr=self._thetas.indptr, thetas_shape=self._thetas.shape,
                     alphas_orig=self._alphas_orig, betas_orig=self._betas_orig,
                     thetas_orig_data=self._thetas_orig.data, thetas_orig_indices=self._thetas_orig.indices,
                     thetas_orig_indptr=self._thetas_orig.indptr, thetas_orig_shape=self._thetas_orig.shape,
                     ntopics=self._ntopics, betas_ds=self._betas_ds, topic_entropy=self._topic_entropy,
                     descriptions=self._tpc_descriptions, edits=self._edits)
        else:
            np.savez(npzfile, alphas=self._alphas, betas=self._betas, thetas=self._thetas, alphas_orig=self._alphas_orig, betas_orig=self._betas_orig, thetas_orig=self._thetas_orig, ntopics=self._ntopics, betas_ds=self._betas_ds, topic_entropy=self._topic_entropy,descriptions=self._tpc_descriptions, edits=self._edits)

        if len(self._edits):
            edits_file = Path(npzfile).parent.joinpath('model_edits.txt')
            with edits_file.open('w', encoding='utf8') as fout:
                [fout.write(el + '\n') for el in self._edits]


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
    parser.add_argument("--showTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Retrieve topic labels and word composition for selected model")
    parser.add_argument("--setTpcLabels", type=str, default=None,
                        metavar=("modelName"),
                        help="Set Topics Labels for selected model")
    parser.add_argument("--deleteTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Remove topics from selected model")
    parser.add_argument("--sortTopics", type=str, default=None,
                        metavar=("modelName"),
                        help="Sort topics according to size")
    parser.add_argument("--resetTM", type=str, default=None,
                        metavar=("modelName"),
                        help="Reset Topic Model to its initial values after training")

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

    if args.showTopics:
        tm = TMmodel(tm_path.joinpath(f"{args.showTopics}").joinpath('TMmodel'))
        sys.stdout.write(json.dumps(tm.showTopics()))

    if args.setTpcLabels:
        # Labels should come from standard input
        TpcLabels = "".join([line for line in sys.stdin])
        TpcLabels = json.loads(TpcLabels.replace('\\"', '"'))
        tm = TMmodel(tm_path.joinpath(f"{args.setTpcLabels}").joinpath('TMmodel'))
        status = tm.setTpcLabels(TpcLabels)
        sys.stdout.write(str(status))

    if args.deleteTopics:
        # List of topics to remove should come from standard input
        tpcs = "".join([line for line in sys.stdin])
        tpcs = json.loads(tpcs.replace('\\"', '"'))
        tm = TMmodel(tm_path.joinpath(f"{args.deleteTopics}").joinpath('TMmodel'))
        status = tm.deleteTopics(tpcs)
        sys.stdout.write(str(status))

    if args.sortTopics:
        tm = TMmodel(tm_path.joinpath(f"{args.sortTopics}").joinpath('TMmodel'))
        status = tm.sortTopics()
        sys.stdout.write(str(status))

    if args.resetTM:
        tm = TMmodel(tm_path.joinpath(f"{args.resetTM}").joinpath('TMmodel'))
        status = tm.resetTM()
        sys.stdout.write(str(status))

