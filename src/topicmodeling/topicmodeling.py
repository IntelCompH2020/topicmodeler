"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides several classes for Topic Modeling
    - textPreproc: Preparation of datasets for training topic models, including
                   - string cleaning (stopword removal + equivalent terms)
                   - BoW calculation
    - TMmodel: To represent a trained topic model + edition functions
    - MalletTrainer: To train a topic model from a given corpus
"""

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
import sys
from abc import abstractmethod
from pathlib import Path
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.topicmodeling.neural_models.contextualized_topic_models.ctm_network.ctm import CombinedTM
from src.topicmodeling.neural_models.contextualized_topic_models.utils.data_preparation import prepare_ctm_dataset
from src.topicmodeling.neural_models.pytorchavitm.avitm_network.avitm import AVITM
from src.topicmodeling.neural_models.pytorchavitm.utils.data_preparation import prepare_dataset

"""
# from scipy.spatial.distance import jensenshannon
# import pyLDAvis
import logging
import regex as javare
import re
from tqdm import tqdm
import ipdb
from gensim import corpora
from gensim.utils import check_output, tokenize
import pickle

logging.getLogger("gensim").setLevel(logging.WARNING)
"""


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


class textPreproc(object):
    """
    A simple class to carry out some simple text preprocessing tasks
    that are needed by topic modeling
    - Stopword removal
    - Replace equivalent terms
    - Calculate BoW
    - Generate the files that are needed for training of different
      topic modeling technologies

    It allows to use Gensim or Spark functions
    """

    def __init__(self, stw_files=[], eq_files=[],
                 min_lemas=15, no_below=10, no_above=0.6,
                 keep_n=100000, cntVecModel=None, logger=None):
        """
        Initilization Method
        Stopwords and the dictionary of equivalences will be loaded
        during initialization

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        eq_files: list of str
            List of paths to equivalent terms files
        min_lemas: int
            Minimum number of lemas for document filtering
        no_below: int
            Minimum number of documents to keep a term in the vocabulary
        no_above: float
            Maximum proportion of documents to keep a term in the vocab
        keep_n: int
            Maximum vocabulary size
        cntVecModel : pyspark.ml.feature.CountVectorizerModel
            CountVectorizer Model to be used for the BOW calculation
        logger: Logger object
            To log object activity
        """
        self._stopwords = self._loadSTW(stw_files)
        self._equivalents = self._loadEQ(eq_files)
        self._min_lemas = min_lemas
        self._no_below = no_below
        self._no_above = no_above
        self._keep_n = keep_n
        self._cntVecModel = cntVecModel

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('textPreproc')

    def _loadSTW(self, stw_files):
        """
        Loads all stopwords from all files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files

        Returns
        -------
        stopWords: list of str
            List of stopwords
        """

        stopWords = []
        for stwFile in stw_files:
            with Path(stwFile).open('r', encoding='utf8') as fin:
                stopWords += json.load(fin)['wordlist']

        return list(set(stopWords))

    def _loadEQ(self, eq_files):
        """
        Loads all equivalent terms from all files provided in the argument

        Parameters
        ----------
        eq_files: list of str
            List of paths to equivalent terms files

        Returns
        -------
        equivalents: dictionary
            Dictionary of term_to_replace -> new_term
        """

        equivalent = {}

        for eqFile in eq_files:
            with Path(eqFile).open('r', encoding='utf8') as fin:
                newEq = json.load(fin)['wordlist']
            newEq = [x.split(':') for x in newEq]
            newEq = [x for x in newEq if len(x) == 2]
            newEq = dict(newEq)
            equivalent = {**equivalent, **newEq}

        return equivalent

    def preprocBOW(self, trDF):
        """
        Preprocesses the documents in the dataframe to carry
        out the following tasks
            - Filter out short documents (below min_lemas)
            - Cleaning of stopwords
            - Equivalent terms application
            - BoW calculation

        Parameters
        ----------
        trDF: Pandas or Spark dataframe
            This routine works on the following column "all_lemmas"
            Other columns are left untouched

        Returns
        -------
        trDFnew: A new dataframe with a new colum bow containing the
        bow representation of the documents
        """
        if isinstance(trDF, pd.DataFrame):
            print('pandas')

            """
                    Preprocessing of files
            For the training we have access (in self._corpusFiles) to a number of lemmatized
            documents. This function:
                1) Carries out a first set of cleaning and homogeneization tasks
                2) Allow to reduce the size of the vocabulary (removing very rare or common terms)
                3) Import the training corpus into Mallet format

            # Identification of words that are too rare or common that need to be
            # removed from the dictionary.
            self.logger.info('-- -- Mallet Corpus Generation: Vocabulary generation')
            dictionary = corpora.Dictionary()

            # We iterate over all CSV files
            #### Very important
            #### Corpus that can be used for Topic Modeling must contain
            #### two fields: ["id", "lemmas"]
            print('Processing files for vocabulary creation')
            pbar = tqdm(self._corpusFiles)
            for csvFile in pbar:
                df = pd.read_csv(csvFile, escapechar="\\", on_bad_lines="skip")
                if "id" not in df.columns or "lemmas" not in df.columns:
                    self.logger.error('-- -- Traning corpus error: must contain "id" and "lemmas" - Exit')
                    sys.exit()
                # Keep only relevant fields
                id_lemas = df[["id", "lemmas"]].values.tolist()
                # Apply regular expression to identify tokens
                id_lemas = [[el[0], ' '.join(self._token_regexp.findall(el[1].replace('\n', ' ').strip()))]
                            for el in id_lemas]
                # Apply stopwords and equivalence files
                id_lemas = [[el[0], self._stwEQ.cleanstr(el[1])] for el in id_lemas]
                # Apply gensim tokenizer
                id_lemas = [[el[0], list(tokenize(el[1], lowercase=True, deacc=True))] for el in id_lemas]
                # Retain only documents with minimum extension
                id_lemas = [el for el in id_lemas if len(el[1]) >= self._min_lemas]
                # Add to dictionary
                all_lemas = [el[1] for el in id_lemas]
                dictionary.add_documents(all_lemas)

            # Remove words that appear in less than no_below documents, or in more than
            # no_above, and keep at most keep_n most frequent terms, keep track of removed
            # words for debugging purposes
            all_words = [dictionary[idx] for idx in range(len(dictionary))]
            dictionary.filter_extremes(no_below=self._no_below, no_above=self._no_above, keep_n=self._keep_n)
            kept_words = set([dictionary[idx] for idx in range(len(dictionary))])
            rmv_words = [el for el in all_words if el not in kept_words]
            # Save extreme words that will be removed
            self.logger.info(f'-- -- Saving {len(rmv_words)} extreme words to file')
            rmv_file = self._modelFolder.joinpath('commonrare_words.txt')
            with rmv_file.open('w', encoding='utf-8') as fout:
                [fout.write(el + '\n') for el in sorted(rmv_words)]
            # Save dictionary to file
            self.logger.info(f'-- -- Saving dictionary to file. Number of words: {len(kept_words)}')
            vocab_txt = self._modelFolder.joinpath('vocabulary.txt')
            with vocab_txt.open('w', encoding='utf-8') as fout:
                [fout.write(el + '\n') for el in sorted(kept_words)]
            # Save also in gensim text format
            vocab_gensim = self._modelFolder.joinpath('vocabulary.gensim')
            dictionary.save_as_text(vocab_gensim)

            ##################################################
            # Create corpus txt files
            self.logger.info('-- -- Mallet Corpus generation: TXT file')

            corpus_file = self._modelFolder.joinpath('training_data.txt')

            fcorpus = corpus_file.open('w', encoding='utf-8')

            print('Processing files for training dataset creation')
            pbar = tqdm(self._corpusFiles)
            for csvFile in pbar:
                # Document preprocessing is same as before, but now we apply an additional filter
                # and keep only words in the vocabulary
                df = pd.read_csv(csvFile, escapechar="\\", on_bad_lines="skip")
                id_lemas = df[["id", "lemmas"]].values.tolist()
                id_lemas = [[el[0], ' '.join(self._token_regexp.findall(el[1].replace('\n', ' ').strip()))]
                            for el in id_lemas]
                id_lemas = [[el[0], self._stwEQ.cleanstr(el[1])] for el in id_lemas]
                id_lemas = [[el[0], list(tokenize(el[1], lowercase=True, deacc=True))] for el in id_lemas]
                id_lemas = [[el[0], [tk for tk in el[1] if tk in kept_words]] for el in id_lemas]
                id_lemas = [el for el in id_lemas if len(el[1]) >= self._min_lemas]
                # Write to corpus file
                [fcorpus.write(el[0] + ' 0 ' + ' '.join(el[1]) + '\n') for el in id_lemas]

            fcorpus.close()

            """
        else:
            # Preprocess data using Spark
            # tokenization
            tk = Tokenizer(inputCol="all_lemmas", outputCol="tokens")
            trDF = tk.transform(trDF)

            # Removal of Stopwords - Skip if not stopwords are provided
            # to save computation time
            if len(self._stopwords):
                swr = StopWordsRemover(inputCol="tokens", outputCol="clean_tokens",
                                       stopWords=self._stopwords)
                trDF = swr.transform(trDF)
            else:
                # We need to create a copy of the tokens with the new name
                trDF = trDF.withColumn("clean_tokens", trDF["tokens"])

            # Filter according to number of lemmas in each document
            trDF = trDF.where(F.size(F.col("clean_tokens")) >= self._min_lemas)

            # Equivalences replacement
            if len(self._equivalents):
                df = trDF.select(trDF.id, F.explode(trDF.clean_tokens))
                df = df.na.replace(self._equivalents, 1)
                df = df.groupBy("id").agg(F.collect_list("col"))
                trDF = (trDF.join(df, trDF.id == df.id, "left")
                        .drop(df.id)
                        .withColumnRenamed("collect_list(col)", "final_tokens")
                        )

            if not self._cntVecModel:
                cntVec = CountVectorizer(inputCol="final_tokens",
                                         outputCol="bow", minDF=self._no_below,
                                         maxDF=self._no_above, vocabSize=self._keep_n)
                self._cntVecModel = cntVec.fit(trDF)

            trDFnew = (self._cntVecModel.transform(trDF)
                           .drop("tokens", "clean_tokens", "final_tokens")
                       )

        return trDFnew

    def saveCntVecModel(self, dirpath):
        """
        Saves a Count Vectorizer Model to the specified path
        Saves also a text document with the corresponding
        vocabulary

        Parameters
        ----------
        dirpath: pathlib.Path
            The folder where the CountVectorizerModel and the
            text file with the vocabulary will be saved

        Returns
        -------
        status: int
            - 1: If the files were generated sucessfully
            - 0: Error (Count Vectorizer Model does not exist)
        """
        if self._cntVecModel:
            cntVecModel = dirpath.joinpath('CntVecModel')
            if cntVecModel.is_dir():
                shutil.rmtree(cntVecModel)
            self._cntVecModel.save(f"file://{cntVecModel.as_posix()}")
            with dirpath.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
                fout.write(
                    '\n'.join([el for el in self._cntVecModel.vocabulary]))
            return 1
        else:
            return 0

    def exportTrData(self, trDF, dirpath, tmTrainer):
        """
        Exports the training data in the provided dataset to the
        format required by the topic modeling trainer

        Parameters
        ----------
        trDF: Pandas or Spark dataframe
            The dataframe should contain a column "bow" that will
            be used to calculate the training data
        dirpath: pathlib.Path
            The folder where the data will be saved
        tmTrainer: string
            The output format [mallet|sparkLDA|prodLDA|ctm]

        Returns
        -------
        outFile: Path
            A path containing the location of the training data in the indicated format
        """

        if isinstance(trDF, pd.DataFrame):
            print('pandas')
        else:
            if tmTrainer == "mallet":
                # We need to convert the bow back to text, and save text file
                # in mallet format
                outFile = dirpath.joinpath('corpus.txt')
                vocabulary = self._cntVecModel.vocabulary
                spark.sparkContext.broadcast(vocabulary)

                # User defined function to recover the text corresponding to BOW
                def back2text(bow):
                    text = ""
                    for idx, tf in zip(bow.indices, bow.values):
                        text += int(tf) * (vocabulary[idx] + ' ')
                    return text.strip()
                back2textUDF = F.udf(lambda z: back2text(z))

                malletDF = (trDF.withColumn("bow_text", back2textUDF(F.col("bow")))
                            .withColumn("2mallet", F.concat_ws(" 0 ", "id", "bow_text"))
                            .select("2mallet")
                            )
                # Save as text file
                # Ideally everything should get written to one text file directly from Spark
                # but this is failing repeatedly, so I avoid coalescing in Spark and
                # instead concatenate all files after creation
                tempFolder = dirpath.joinpath('tempFolder')
                #malletDF.coalesce(1).write.format("text").option("header", "false").save(f"file://{tempFolder.as_posix()}")
                malletDF.write.format("text").option("header", "false").save(
                    f"file://{tempFolder.as_posix()}")
                # Concatenate all text files
                with outFile.open("w", encoding="utf8") as fout:
                    for inFile in [f for f in tempFolder.iterdir() if f.name.endswith('.txt')]:
                        fout.write(inFile.open("r").read())
                shutil.rmtree(tempFolder)

            elif tmTrainer == "sparkLDA":
                # Save necessary columns for Spark LDA in parquet file
                outFile = dirpath.joinpath('corpus.parquet')
                trDF.select("id", "source", "bow").write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")
            elif tmTrainer == "prodLDA":
                outFile = dirpath.joinpath('lemas.parquet')
                lemas_df = (trDF.withColumn("bow_text", back2textUDF(F.col("bow")))
                            .select("id", "bow_text")
                            )
                lemas_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")
            elif tmTrainer == "ctm":
                outFile = dirpath.joinpath('lemas.parquet')
                lemas_raw_df = (trDF.withColumn("bow_text", back2textUDF(F.col("bow")))
                                .select("id", "bow_text", "all_raw_text")
                               )
                lemas_raw_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")

        return outFile


class stwEQcleaner(object):
    """Simpler version of the english lemmatizer
    It only provides stopword removal and application of equivalences

    Public methods:
        - cleanstr: Apply stopwords and equivalences on provided string
    """

    def __init__(self, stw_files=[], dict_eq_file='', logger=None):
        """
        Initilization Method
        Stopwwords and the dictionary of equivalences will be loaded
        during initialization

        Parameters
        ----------
        stw_file: list
            List of files of stopwords
        dict_eq_file: pathlib.Path
            Dictionary of equivalent words A : B means A will be replaced by B

        """
        self.__stopwords = []

        # Unigrams for word replacement
        self.__useunigrams = False
        self.__pattern_unigrams = None
        self.__unigramdictio = None
        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('stwEQcleaner')
        # Load stopwords
        for stw_file in stw_files:
            if stw_file.is_file():
                self.__stopwords += self.__loadStopFile(stw_file)
            else:
                self.logger.info('-- -- Stopwords file not found')
        self.__stopwords = set(self.__stopwords)

        # Predefined equivalences as provided in file
        if dict_eq_file.is_file():
            self.__unigramdictio, self.__pattern_unigrams = self.__loadEQFile(
                dict_eq_file)
            if len(self.__unigramdictio):
                self.__useunigrams = True
        else:
            self.logger.info('-- -- Equivalence file not found')

        return

    def cleanstr(self, rawtext):
        """Function to remove stopwords and apply equivalences

        Parameters
        ----------
        rawtext: str
            string with the text to lemmatize
        """
        if rawtext == None or rawtext == '':
            return ''
        else:
            texto = ' '.join(self.__removeSTW(rawtext.split()))
            # Make equivalences according to dictionary
            if self.__useunigrams:
                texto = self.__pattern_unigrams.sub(
                    lambda x: self.__unigramdictio[x.group()], texto)
        return texto

    def __loadStopFile(self, file):
        """Function to load the stopwords from a file. The stopwords will be
        read from the file, one stopword per line

        Parameters
        ----------
        file:
            The file to read the stopwords from
        """
        with open(file, encoding='utf-8') as f:
            stopw = f.read().splitlines()

        return [word.strip() for word in stopw if word]

    def __loadEQFile(self, file):
        """Function to load equivalences from a file. The equivalence file
        will contain an equivalence per line in the format original : target
        where original will be changed to target after lemmatization

        Parameters
        ----------
        file:
            The file to read the equivalences from
        """
        unigrams = []
        with open(file, 'r', encoding='utf-8') as f:
            unigramlines = f.readlines()
        unigramlines = [el.strip() for el in unigramlines]
        unigramlines = [x.split(' : ') for x in unigramlines]
        unigramlines = [x for x in unigramlines if len(x) == 2]

        if len(unigramlines):
            # This dictionary contains the necessary replacements to carry out
            unigramdictio = dict(unigramlines)
            unigrams = [x[0] for x in unigramlines]
            # Regular expression to find the tokens that need to be replaced
            pattern_unigrams = re.compile(r'\b(' + '|'.join(unigrams) + r')\b')
            return unigramdictio, pattern_unigrams
        else:
            return None, None

    def __removeSTW(self, tokens):
        """Removes stopwords from the provided list

        Parameters
        ----------
        tokens:
            Input list of string to be cleaned from stw
        """
        return [el for el in tokens if el not in self.__stopwords]


class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox that
    # produces a model according to this representation

    # Estas variables guardarán los valores originales de las alphas, betas, thetas
    # servirán para resetear el modelo después de su optimización
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
    _vocabfreq = None
    _size_vocab = None
    _vocabfreq_file = None

    def __init__(self, betas=None, thetas=None, alphas=None,
                 vocabfreq_file=None, from_file=None, logger=None):
        """Inicializacion del model de topicos a partir de las matrices que lo caracterizan
        Ademas de inicializar las correspondientes variables del objeto, se recalcula el Vector
        beta con downscoring (palabras comunes son penalizadas), y se calculan las
        entropias de cada topico.

        Parameters
        ----------
        betas:
            Matriz numpy de tamaño n_topics x n_words (vocab de cada tópico)
        thetas:
            Matriz numpy de tamaño n_docs x n_topics (composición documental)
        alphas:
            Vector de longitud n_topics, con la importancia de cada perfil
        vocabfreq_file:
            Ruta a un fichero con el vocabulario correspondiente al modelo. Contiene también la frecuencia de cada términos del vocabulario
        from_file:
            If not None, contains the name of a file from which the object can be initialized
        """
        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('TMmodel')

        # Convert strings to Paths if necessary
        if vocabfreq_file:
            vocabfreq_file = Path(vocabfreq_file)
        if from_file:
            from_file = Path(from_file)

        # Locate vocabfile for the model
        if not vocabfreq_file and from_file:
            # If not vocabfile was indicated, try to recover from same directory where model is
            vocabfreq_file = from_file.parent.joinpath('vocab_freq.txt')

        if not vocabfreq_file.is_file():
            self.logger.error(
                '-- -- -- It was not possible to locate a valid vocabulary file.')
            self.logger.error('-- -- -- The TMmodel could not be created')
            return

        # Load vocabulary variables from file
        self._vocab_w2id, self._vocab_id2w, self._vocabfreq = self.lee_vocabfreq(
            vocabfreq_file)
        self._size_vocab = len(self._vocabfreq)
        self._vocabfreq_file = vocabfreq_file

        # Create model from given data, or recover model from file
        if from_file:
            data = np.load(from_file)
            self._alphas = data['alphas']
            self._betas = data['betas']
            if 'thetas' in data:
                # non-sparse thetas
                self._thetas = data['thetas']
            else:
                self._thetas = sparse.csr_matrix((data['thetas_data'], data['thetas_indices'], data['thetas_indptr']),
                                                 shape=data['thetas_shape'])

            self._alphas_orig = data['alphas_orig']
            self._betas_orig = data['betas_orig']
            if 'thetas_orig' in data:
                self._thetas_orig = data['thetas_orig']
            else:
                self._thetas_orig = sparse.csr_matrix((data['thetas_orig_data'],
                                                       data['thetas_orig_indices'], data['thetas_orig_indptr']),
                                                      shape=data['thetas_orig_shape'])
            self._ntopics = data['ntopics']
            self._betas_ds = data['betas_ds']
            self._topic_entropy = data['topic_entropy']
            self._descriptions = [str(x) for x in data['descriptions']]
            self._edits = [str(x) for x in data['edits']]
        else:
            # Cuando el modelo se genera desde el principio, tenemos que
            # guardar los alphas, betas y thetas tanto en las permanentes
            # como en las actuales que se utilizan para visualizar el modelo
            self._betas_orig = betas
            self._thetas_orig = thetas
            self._alphas_orig = alphas
            self._betas = betas
            self._thetas = thetas
            self._alphas = alphas
            self._edits = []
            self._ntopics = self._thetas.shape[1]
            self._calculate_other()
            # Descripciones
            self._descriptions = [x[1] for x in
                                  self.get_topic_word_descriptions()]
            # Reordenamiento inicial de tópicos
            self.sort_topics()

        self.logger.info(
            '-- -- -- Topic model object (TMmodel) successfully created')
        return

    def _calculate_other(self):
        """This function is intended to calculate all other variables
        of the TMmodel object
            * self._betas_ds
            * self._topic_entropy
        """
        # ======
        # 1. self._betas_ds
        # Calculamos betas con down-scoring
        self._betas_ds = np.copy(self._betas)
        if np.min(self._betas_ds) < 1e-12:
            self._betas_ds += 1e-12
        deno = np.reshape((sum(np.log(self._betas_ds)) /
                          self._ntopics), (self._size_vocab, 1))
        deno = np.ones((self._ntopics, 1)).dot(deno.T)
        self._betas_ds = self._betas_ds * (np.log(self._betas_ds) - deno)
        # ======
        # 2. self._topic_entropy
        # Nos aseguramos de que no hay betas menores que 1e-12. En este caso betas nunca es sparse
        if np.min(self._betas) < 1e-12:
            self._betas += 1e-12
        self._topic_entropy = - \
            np.sum(self._betas * np.log(self._betas), axis=1)
        self._topic_entropy = self._topic_entropy / np.log(self._size_vocab)
        return

    def get_alphas(self):
        return self._alphas

    def get_betas(self):
        return self._betas

    def get_thetas(self):
        return self._thetas

    def get_ntopics(self):
        return self._ntopics

    def get_tpc_corrcoef(self):
        # Computes topic correlation. Highly correlated topics
        # co-occure together
        # Topic mean
        med = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        # Topic square mean
        thetas2 = self._thetas.multiply(self._thetas)
        med2 = np.asarray(np.mean(thetas2, axis=0)).ravel()
        # Topic stds
        stds = np.sqrt(med2 - med ** 2)
        # Topic correlation
        num = self._thetas.T.dot(
            self._thetas).toarray() / self._thetas.shape[0]
        num = num - med[..., np.newaxis].dot(med[np.newaxis, ...])
        deno = stds[..., np.newaxis].dot(stds[np.newaxis, ...])
        return num / deno

    def get_tpc_JSdist(self, thr=1e-3):
        # Computes inter-topic distance based on word distributions
        # using Jensen Shannon distance
        # For a more efficient computation with very large vocabularies
        # we implement a threshold for restricting the distance calculation
        # to columns where any elment is greater than threshold thr
        betas_aux = self._betas[:, np.where(self._betas.max(axis=0) > thr)[0]]
        js_mat = np.zeros((self._ntopics, self._ntopics))
        for k in range(self._ntopics):
            for kk in range(self._ntopics):
                js_mat[k, kk] = jensenshannon(
                    betas_aux[k, :], betas_aux[kk, :])
        return js_mat

    def get_similar_corrcoef(self, npairs):
        # Returns most similar pairs of topics by co-occurence of topics in docs
        corrcoef = self.get_tpc_corrcoef()
        selected = largest_indices(corrcoef, self._ntopics + 2 * npairs)
        return selected

    def get_similar_JSdist(self, npairs, thr=1e-3):
        # Returns most similar pairs of topics by co-occurence of topics in docs
        JSsim = 1 - self.get_tpc_JSdist(thr)
        selected = largest_indices(JSsim, self._ntopics + 2 * npairs)
        return selected

    def get_descriptions(self, tpc=None):
        if not tpc:
            return self._descriptions
        else:
            return self._descriptions[tpc]

    def set_description(self, desc_tpc, tpc):
        """Set description of topic tpc to desc_tpc

        Parameters
        ----------
        desc_tpc:
            String with the description for the topic
        tpc:
            Number of topic
        """
        if tpc > self._ntopics - 1:
            print(
                'Error setting topic description: Topic ID is larger than number of topics')
        else:
            self._descriptions[tpc] = desc_tpc
        return

    def lee_vocabfreq(self, vocabfreq_path):
        """Lee el vocabulario del modelo que se encuentra en el fichero indicado
        Devuelve dos diccionarios, uno usando las palabras como llave, y el otro
        utilizando el id de la palabra como clave
        Devuelve también la lista de frequencias de cada término del vocabulario

        Parameters
        ----------
        vocabfreq_path: pathlib.Path
            Path con la ruta al vocabulario

        Returns
        -------
        : tuple(vocab_w2id,vocab_id2w)
            * vocab_w2id         : Diccionario {pal_i : id_pal_i}
            * vocab_id2w         : Diccionario {i     : pal_i}
        """
        vocab_w2id = {}
        vocab_id2w = {}
        vocabfreq = []
        with vocabfreq_path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                wd, freq = line.strip().split('\t')
                vocab_w2id[wd] = i
                vocab_id2w[str(i)] = wd
                vocabfreq.append(int(freq))

        return (vocab_w2id, vocab_id2w, vocabfreq)

    def save_npz(self, npzfile):
        """Salva las matrices que caracterizan el modelo de tópicos en un fichero npz de numpy

        Parameters
        ----------
        npzfile:
            Nombre del fichero en el que se guardará el modelo
        """
        if isinstance(self._thetas, sparse.csr_matrix):
            np.savez(npzfile, alphas=self._alphas, betas=self._betas,
                     thetas_data=self._thetas.data, thetas_indices=self._thetas.indices,
                     thetas_indptr=self._thetas.indptr, thetas_shape=self._thetas.shape,
                     alphas_orig=self._alphas_orig, betas_orig=self._betas_orig,
                     thetas_orig_data=self._thetas_orig.data, thetas_orig_indices=self._thetas_orig.indices,
                     thetas_orig_indptr=self._thetas_orig.indptr, thetas_orig_shape=self._thetas_orig.shape,
                     ntopics=self._ntopics, betas_ds=self._betas_ds, topic_entropy=self._topic_entropy,
                     descriptions=self._descriptions, edits=self._edits)
        else:
            np.savez(npzfile, alphas=self._alphas, betas=self._betas, thetas=self._thetas,
                     alphas_orig=self._alphas_orig, betas_orig=self._betas_orig, thetas_orig=self._thetas_orig,
                     ntopics=self._ntopics, betas_ds=self._betas_ds, topic_entropy=self._topic_entropy,
                     descriptions=self._descriptions, edits=self._edits)

        if len(self._edits):
            edits_file = Path(npzfile).parent.joinpath('model_edits.txt')
            with edits_file.open('w', encoding='utf8') as fout:
                [fout.write(el + '\n') for el in self._edits]

    def thetas2sparse(self, thr):
        """Convert thetas matrix to CSR format

        Parameters
        ----------
        thr:
            Threshold to umbralize the matrix
        """
        self._thetas[self._thetas < thr] = 0
        self._thetas = sparse.csr_matrix(self._thetas, copy=True)
        self._thetas = normalize(self._thetas, axis=1, norm='l1')
        self._thetas_orig[self._thetas_orig < thr] = 0
        self._thetas_orig = sparse.csr_matrix(self._thetas_orig, copy=True)
        self._thetas_orig = normalize(self._thetas_orig, axis=1, norm='l1')

    def muestra_perfiles(self, n_palabras=10, tfidf=True, tpc=None):
        """Muestra por pantalla los perfiles del modelo lda por pantalla

        Parameters
        ----------
        n_palabas:
            Número de palabras a mostrar para cada perfil
        tfidf:
            Si True, se hace downscaling de palabras poco específicas (Blei and Lafferty, 2009)
        tpc:
            If not None, se imprimen los tópicos con ID en la lista tpc e.g.: tpc = [0,3,4]
        """
        if not tpc:
            tpc = range(self._ntopics)
        for i in tpc:
            if tfidf:
                words = [self._vocab_id2w[str(idx2)]
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [self._vocab_id2w[str(idx2)]
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            print(str(i) + '\t' +
                  str(self._alphas[i]) + '\t' + ', '.join(words))
        return

    def muestra_descriptions(self, tpc=None, simple=False):
        """Muestra por pantalla las descripciones de los perfiles del modelo lda

        Parameters
        ----------
        tpc:
            If not None, se imprimen los tópicos con ID en la lista tpc e.g.: tpc = [0,3,4]
        """
        if not tpc:
            tpc = range(self._ntopics)
        for i in tpc:
            if not simple:
                print(str(i) + '\t' +
                      str(self._alphas[i]) + '\t' + self._descriptions[i])
            else:
                print('\t'.join(self._descriptions[i].split(', ')))

    def get_topic_word_descriptions(self, n_palabras=15, tfidf=True, tpc=None):
        """Devuelve una lista con las descripciones del modelo de tópicos

        Parameters
        ----------
        n_palabas:
            Número de palabras a mostrar para cada perfil
        tfidf:
            Si True, se hace downscaling de palabras poco específicas (Blei and Lafferty, 2009)
        tpc:
            If not None, se devuelven las descripciones de los tópicos con ID en la lista tpc e.g.: tpc = [0,3,4]                        
        """
        if not tpc:
            tpc = range(self._ntopics)
        tpc_descs = []
        for i in tpc:
            if tfidf:
                words = [self._vocab_id2w[str(idx2)]
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [self._vocab_id2w[str(idx2)]
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            tpc_descs.append((i, ', '.join(words)))
        return tpc_descs

    def most_significant_words_per_topic(self, n_palabras=10, tfidf=True, tpc=None):
        """Devuelve una lista de listas de tuplas, en el formato:
        ::

            [  [(palabra1tpc1, beta), (palabra2tpc1, beta)],
            [   (palabra1tpc2, beta), (palabra2tpc2, beta)]   ]

        Parameters
        ----------
        n_palabas:
            Número de palabras que se devuelven para cada perfil
        tfidf:
            Si True, para la relevancia se emplea el downscaling de palabras poco específicas de (Blei and Lafferty, 2009)
        tpc:
            If not None, se devuelven únicamente las descripciones de los tópicos con ID en la lista tpc e.g.: tpc = [0,3,4]                        
        """
        if not tpc:
            tpc = range(self._ntopics)
        mswpt = []
        for i in tpc:
            if tfidf:
                words = [(self._vocab_id2w[str(idx2)], self._betas[i, idx2])
                         for idx2 in np.argsort(self._betas_ds[i])[::-1][0:n_palabras]]
            else:
                words = [(self._vocab_id2w[str(idx2)], self._betas[i, idx2])
                         for idx2 in np.argsort(self._betas[i])[::-1][0:n_palabras]]
            mswpt.append(words)
        return mswpt

    def ndocs_active_topic(self):
        """Returns the number of documents where each topic is active"""
        return (self._thetas != 0).sum(0).tolist()[0]

    def delete_topic(self, tpc):
        """Deletes the indicated topic

        Parameters
        ----------
        tpc:
            The topic to delete (an integer in range 0:ntopics)
        """
        # Keep record of model changes
        self._edits.append('d ' + str(tpc))
        # Update data matrices
        self._betas = np.delete(self._betas, tpc, 0)
        # It could be more efficient, but this complies with full and csr matrices
        tpc_keep = [k for k in range(self._ntopics) if k != tpc]
        self._thetas = self._thetas[:, tpc_keep]
        self._thetas = normalize(self._thetas, axis=1, norm='l1')
        self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        self._ntopics = self._thetas.shape[1]

        # Remove topic description
        del self._descriptions[tpc]
        # Recalculate all other stuff
        self._calculate_other()

        return

    def fuse_topics(self, tpcs):
        """Hard fusion of several topics

        Parameters
        ----------
        tpcs:
            List of topics for the fusion
        """
        # Keep record of model chages
        tpcs = sorted(tpcs)
        self._edits.append('f ' + ' '.join([str(el) for el in tpcs]))
        # Update data matrices. For beta we keep an average of topic vectors
        weights = self._alphas[tpcs]
        bet = weights[np.newaxis, ...].dot(
            self._betas[tpcs, :]) / (sum(weights))
        # keep new topic vector in upper position
        self._betas[tpcs[0], :] = bet
        self._betas = np.delete(self._betas, tpcs[1:], 0)
        # For theta we need to keep the sum. Since adding implies changing
        # structure, we need to convert to full matrix first
        # No need to renormalize
        thetas_full = self._thetas.toarray()
        thet = np.sum(thetas_full[:, tpcs], axis=1)
        thetas_full[:, tpcs[0]] = thet
        thetas_full = np.delete(thetas_full, tpcs[1:], 1)
        self._thetas = sparse.csr_matrix(thetas_full, copy=True)
        # Compute new alphas and number of topics
        self._alphas = np.asarray(np.mean(self._thetas, axis=0)).ravel()
        self._ntopics = self._thetas.shape[1]

        # Remove topic descriptions
        for tpc in tpcs[1:]:
            del self._descriptions[tpc]

        # Recalculate all other stuff
        self._calculate_other()

        return

    def sort_topics(self):
        """Sort topics according to topic size"""
        # Indexes for topics reordering
        idx = np.argsort(self._alphas)[::-1]
        self._edits.append('s ' + ' '.join([str(el) for el in idx]))

        # Sort data matrices
        self._alphas = self._alphas[idx]
        self._betas = self._betas[idx, :]
        self._thetas = self._thetas[:, idx]

        # Sort topic descriptions
        self._descriptions = [self._descriptions[k] for k in idx.tolist()]

        # Recalculate all other stuff
        self._calculate_other()

        return

    def reset_model(self):
        """Resetea el modelo al resultado del LDA original con todos los tópicos"""
        self.__init__(betas=self._betas_orig, thetas=self._thetas_orig,
                      alphas=self._alphas_orig, vocabfreq_file=self._vocabfreq_file)
        return

    def pyLDAvis(self, htmlfile, ndocs, njobs=-1):
        """Generación de la visualización de pyLDAvis
        La visualización se almacena en el fichero que se recibe como argumento

        Parameters
        ----------
        htmlfile:
            Path to generated html file
        ndocs:
            Number of documents used to compute the visualization
        njobs:
            Number of jobs used to accelerate pyLDAvis
        """
        if len([el for el in self._edits if el.startswith('d')]):
            self.logger.error(
                '-- -- -- pyLDAvis: El modelo ha sido editado y se han eliminado tópicos.')
            self.logger.error(
                '-- -- -- pyLDAvis: No se puede generar la visualización.')
            return

        print('Generating pyLDAvisualization. This is an intensive task, consider sampling number of documents')
        print('The corpus you are using has been trained on',
              self._thetas.shape[0], 'documents')
        # Ask user for a different number of docs, than default setting in config file
        ndocs = var_num_keyboard('int', ndocs,
                                 'How many documents should be used to compute the visualization?')
        if ndocs > self._thetas.shape[0]:
            ndocs = self._thetas.shape[0]
        perm = np.sort(np.random.permutation(self._thetas.shape[0])[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocab = [self._vocab_id2w[str(k)]
                 for k in range(len(self._vocab_id2w))]
        vis_data = pyLDAvis.prepare(self._betas, self._thetas[perm, ].toarray(),
                                    doc_len, vocab, self._vocabfreq, lambda_step=0.05,
                                    sort_topics=False, n_jobs=njobs)
        print('Se ha generado la visualización. El fichero se guardará en la carpeta del modelo:')
        print(htmlfile)
        pyLDAvis.save_html(vis_data, htmlfile)
        return

    def automatic_topic_labeling(self, pathlabeling, ranking='unsupervised', nwords=10, workers=3,
                                 num_candidates=19, num_unsup_labels=5, num_sup_labels=5):
        """Genera vector de descripciones para los tópcios de un modelo
        Las descripciones o títulos de los tópicos se guardan en `self._descriptions`

        .. sectionauthor:: Simón Roca Sotelo

        Parameters
        ----------
        pathlabeling:
            Root path to NETL files
        ranking:
            Method to rank candidates ('supervised','unsupervised','both')
        nwords:
            Number of words for representing a topic.
        workers:
            Number of workers for parallel computation
        num_candidates:
            Number of candidates for each topic
        num_unsup_labels:
            Top N unsupervised labels to propose
        num_sup_labels:
            Top N supervised labels to propose

        """

        self.logger.info('-- -- -- NETL: Running automatic_topic_labeling ...')

        # Make sure pathlabeling is a Path
        pathlabeling = Path(pathlabeling)
        # Relative paths to needed files (pre-trained models)
        doc2vecmodel = pathlabeling.joinpath(
            'pre_trained_models', 'doc2vec', 'docvecmodel.d2v')
        word2vecmodel = pathlabeling.joinpath(
            'pre_trained_models', 'word2vec', 'word2vec')
        doc2vec_indices_file = pathlabeling.joinpath(
            'support_files', 'doc2vec_indices')
        word2vec_indices_file = pathlabeling.joinpath(
            'support_files', 'word2vec_indices')
        # This is precomputed pagerank model needed to genrate pagerank features.
        pagerank_model = pathlabeling.joinpath(
            'support_files', 'pagerank-titles-sorted.txt')
        # SVM rank classify. After you download SVM Ranker classify gibve the path of svm_rank_classify here
        svm_classify = pathlabeling.joinpath(
            'support_files', 'svm_rank_classify')
        # This is trained supervised model on the whole our dataset.
        # Run train train_svm_model.py if you want a new model on different dataset.
        pretrained_svm_model = pathlabeling.joinpath(
            'support_files', 'svm_model')

        # Relative paths to temporal files created during execution.
        # The output file for supervised labels.
        out_sup = pathlabeling.joinpath('output_supervised')
        data = pathlabeling.joinpath('temp_topics.csv')
        out_unsup = pathlabeling.joinpath('output_unsupervised')
        cand_gen_output = pathlabeling.joinpath('output_candidates')

        # Deleting temporal files if they exist from a previous run.
        temp_files = [cand_gen_output, data, out_sup, out_unsup]
        [f.unlink() for f in temp_files if f.is_file()]

        # Topics to a temporal file.
        descr = [x[1] for x in self.get_topic_word_descriptions(
            n_palabras=nwords, tfidf=False)]
        with data.open('w', encoding='utf-8') as f:
            head = ['topic_id']
            for n in range(nwords):
                head.append('term' + str(n))
            f.write(','.join(head) + '\n')
            for el in descr:
                f.write(el.replace(', ', ','))
                f.write('\n')

        # Calling external script for candidate generation.
        query1 = 'python ' + str(pathlabeling.joinpath('cand_generation.py')) + ' ' + \
                 str(num_candidates) + ' ' + str(doc2vecmodel) + ' ' + str(word2vecmodel) + \
                 ' ' + str(data) + ' ' + str(cand_gen_output) + ' ' + str(doc2vec_indices_file) + \
                 ' ' + str(word2vec_indices_file) + ' ' + str(workers)

        try:
            self.logger.debug('-- -- -- NETL: Extracting candidate labels')
            self.logger.debug('-- -- -- NETL: Query is gonna be: ' + query1)
            check_output(args=query1, shell=True)
        except:
            self.logger.error(
                '-- -- -- NETL failed to extract labels. Revise your command')
            return

        final = []

        # Ranking the previously obtained candidates, in the variants mentioned above.
        try:

            if ranking == 'both' or ranking == 'supervised':
                query2 = 'python ' + str(pathlabeling.joinpath('supervised_labels.py')) + \
                         ' ' + str(num_sup_labels) + ' ' + str(pagerank_model) + ' ' + \
                         str(data) + ' ' + str(cand_gen_output) + ' ' + str(svm_classify) + \
                         ' ' + str(pretrained_svm_model) + ' ' + str(out_sup)
                try:
                    self.logger.debug(
                        '-- -- -- NETL: Executing Supervised Model')
                    self.logger.debug(
                        '-- -- -- NETL: Query is gonna be: ' + query2)
                    check_output(args=query2, shell=True)
                except:
                    self.logger.error(
                        '-- -- -- NETL failed to extract labels (sup). Revise your command')
                    return

                sup = []
                with out_sup.open('r', encoding='utf-8') as f:
                    for l in f.readlines():
                        sup.append(l.replace('\n', '').split(','))

            if ranking == 'both' or ranking == 'unsupervised':
                query3 = 'python ' + str(pathlabeling.joinpath('unsupervised_labels.py')) + \
                         ' ' + str(num_unsup_labels) + ' ' + str(data) + ' ' + \
                         str(cand_gen_output) + ' ' + str(out_unsup)
                try:
                    self.logger.info(
                        '-- -- -- NETL Executing Unsupervised model')
                    self.logger.info(
                        '-- -- -- NETL: Query is gonna be: ' + query3)
                    check_output(args=query3, shell=True)
                except:
                    self.logger.error(
                        '-- -- -- NETL failed to rank labels (unsup). Revise your command')
                    return

                unsup = []
                with out_unsup.open('r', encoding='utf-8') as f:
                    for l in f.readlines():
                        unsup.append(l.replace('\n', '').split(','))

            # Joining supervised and unsupervised labels, and getting unique labels.
            for i in range(self._ntopics):
                if ranking == 'both':
                    final.append(list(set(sup[i] + unsup[i])))
                elif ranking == 'supervised':
                    final.append(list(set(sup[i])))
                elif ranking == 'unsupervised':
                    final.append(list(set(unsup[i])))
        except Exception as e:
            self.logger.error(
                '-- -- -- NETL: Something went wrong. Revise the previous log.')

        # Deleting temporal files at the end
        self.logger.debug('-- -- -- NETL: Deleting temporal files')
        [f.unlink() for f in temp_files if f.is_file()]

        if len(final) > 0:
            for k, wds in enumerate(final):
                proposed = ', '.join(wds)
                print(10 * '=')
                print('Topic ', k)
                print('Current description is', self._descriptions[k])
                print('Proposed description is', wds)
                print('\n')
                if request_confirmation(msg='Keep newly proposed description?'):
                    self._descriptions[k] = proposed
        return


class Trainer(object):
    """
    Wrapper for a Generic Topic Model Training. Implements the
    following functionalities
    - Import of the corpus to the mallet internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(self, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('textPreproc')

        return

    def _SaveThrFig(self, thetas32, plotFile):
        """Creates a figure to illustrate the effect of thresholding
        The distribution of thetas is plotted, together with the value
        that the trainer is programmed to use for the thresholding

        Parameters
        ----------
        thetas32: 2d numpy array
            the doc-topics matrix for a topic model
        plotFile: Path
            The name of the file where the plot will be saved
        """
        allvalues = np.sort(thetas32.flatten())
        step = int(np.round(len(allvalues) / 1000))
        plt.semilogx(allvalues[::step], (100 / len(allvalues))
                     * np.arange(0, len(allvalues))[::step])
        plt.semilogx([self._thetas_thr, self._thetas_thr], [0, 100], 'r')
        plt.savefig(plotFile)
        plt.close()

        return

    @abstractmethod
    def _createTMmodel(self, modelFolder):
        """Creates an object of class TMmodel hosting the topic model that has been trained and whose output is available at the provided folder

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel

        """

        pass

    @abstractmethod
    def fit(self):
        """
        Training of Topic Model
        """

        pass


class MalletTrainer(Trainer):
    """
    Wrapper for the Mallet Topic Model Training. Implements the
    following functionalities
    - Import of the corpus to the mallet internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(self, mallet_path, ntopics=25, alpha=5.0, optimize_interval=10, num_threads=4, num_iterations=1000, doc_topic_thr=0.0, thetas_thr=0.003, token_regexp=None, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        mallet_path: str
            Full path to mallet binary
        ntopics: int
            Number of topics for the model
        alpha: float
            Parameter for the Dirichlet prior on doc distribution
        optimize_interval: int
            Number of steps betweeen parameter reestimation
        num_threads: int
            Number of threads for the optimization
        num_iterations: int
            Number of iterations for the mallet training
        doc_topic_thr: float
            Min value for topic proportions during mallet training
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        token_regexp: str
            Regular expression for mallet topic model trainer (java type)
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)

        self._mallet_path = Path(mallet_path)
        self._ntopics = ntopics
        self._alpha = alpha
        self._optimize_interval = optimize_interval
        self._num_threads = num_threads
        self._num_iterations = num_iterations
        self._doc_topic_thr = doc_topic_thr
        self._thetas_thr = thetas_thr
        self._token_regexp = token_regexp

        if not self._mallet_path.is_file():
            self._logger.error(
                f'-- -- Provided mallet path is not valid -- Stop')
            sys.exit()

        return

    def _createTMmodel(self, modelFolder):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained using mallet topic modeling and whose
        output is available at the provided folder

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel

        """

        thetas_file = modelFolder.joinpath('doc-topics.txt')

        cols = [k for k in np.arange(2, self._ntopics + 2)]

        # Sparsification of thetas matrix
        self._logger.debug('-- -- Sparsifying doc-topics matrix')
        thetas32 = np.loadtxt(thetas_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)
        # thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32)[:,2:]
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32, modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Create vocabulary files and calculate beta matrix
        # A vocabulary is available with words provided by the Count Vectorizer object,
        # but the new files need the order used by mallet
        wtcFile = modelFolder.joinpath('word-topic-counts.txt')
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self._ntopics, vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))

        with wtcFile.open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc, i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas, axis=1, norm='l1')
        # save vocabulary and frequencies
        vocabfreq_file = modelFolder.joinpath('vocab_freq_mallet.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.debug('-- -- Mallet training: Vocabulary file generated')

        tm = TMmodel(betas=betas, thetas=thetas32, alphas=alphas,
                     vocabfreq_file=vocabfreq_file)

        # Remove doc-topics file. It is no longer needed and takes a lot of space
        thetas_file.unlink()

        return tm

    def fit(self, corpusFile):
        """
        Training of Mallet Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """

        # Output model folder and training file for the corpus
        if not corpusFile.is_file():
            self._logger.error(
                f'-- -- Provided corpus Path does not exist -- Stop')
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('malletModel')
        modelFolder.mkdir()

        ##################################################
        # Importing Data to mallet
        self._logger.info('-- -- Mallet Corpus Generation: Mallet Data Import')

        corpusMallet = modelFolder.joinpath('corpus.mallet')

        cmd = self._mallet_path.as_posix() + \
            ' import-file --preserve-case --keep-sequence ' + \
            '--remove-stopwords --token-regex "' + self._token_regexp + \
            '" --input %s --output %s'
        cmd = cmd % (corpusFile, corpusMallet)

        try:
            self._logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error(
                '-- -- Mallet failed to import data. Revise command')

        ##################################################
        # Mallet Topic model training
        configMallet = modelFolder.joinpath('mallet.config')

        with configMallet.open('w', encoding='utf8') as fout:
            fout.write('input = ' + corpusMallet.resolve().as_posix() + '\n')
            fout.write('num-topics = ' + str(self._ntopics) + '\n')
            fout.write('alpha = ' + str(self._alpha) + '\n')
            fout.write('optimize-interval = ' +
                       str(self._optimize_interval) + '\n')
            fout.write('num-threads = ' + str(self._num_threads) + '\n')
            fout.write('num-iterations = ' + str(self._num_iterations) + '\n')
            fout.write('doc-topics-threshold = ' +
                       str(self._doc_topic_thr) + '\n')
            # fout.write('output-state = ' + os.path.join(self._outputFolder, 'topic-state.gz') + '\n')
            fout.write('output-doc-topics = ' +
                       modelFolder.joinpath('doc-topics.txt').resolve().as_posix() + '\n')
            fout.write('word-topic-counts-file = ' +
                       modelFolder.joinpath('word-topic-counts.txt').resolve().as_posix() + '\n')
            fout.write('diagnostics-file = ' +
                       modelFolder.joinpath('diagnostics.xml ').resolve().as_posix() + '\n')
            fout.write('xml-topic-report = ' +
                       modelFolder.joinpath('topic-report.xml').resolve().as_posix() + '\n')
            fout.write('output-topic-keys = ' +
                       modelFolder.joinpath('topickeys.txt').resolve().as_posix() + '\n')
            fout.write('inferencer-filename = ' +
                       modelFolder.joinpath('inferencer.mallet').resolve().as_posix() + '\n')
            # fout.write('output-model = ' + \
            #    self._outputFolder.joinpath('mallet_output').joinpath('modelo.bin').as_posix() + '\n')
            # fout.write('topic-word-weights-file = ' + \
            #    self._outputFolder.joinpath('mallet_output').joinpath('topic-word-weights.txt').as_posix() + '\n')

        cmd = str(self._mallet_path) + \
            ' train-topics --config ' + str(configMallet)

        try:
            self._logger.info(
                f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Model training failed. Revise command')
            return

        ##################################################
        # Create TMmodel object

        tm = self._createTMmodel(modelFolder)
        tm.save_npz(corpusFile.parent.joinpath('model.npz'))

        return


class ProdLDATrainer(Trainer):
    """
    Wrapper for the ProdLDA Topic Model Training. Implements the
    following functionalities
    - Transformation of the corpus to the ProdLDA internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(self, n_components=10, model_type='prodLDA',
                hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                solver='adam', num_epochs=100, reduce_on_plateau=False,
                topic_prior_mean=0.0, topic_prior_variance=None, num_samples=10, num_data_loader_workers=mp.cpu_count(), thetas_thr=0.003, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        n_components : int (default=10)
            Number of topic components
        model_type : string (default='prodLDA')
            Type of the model that is going to be trained, 'prodLDA' or 'LDA'
        hidden_sizes : tuple, length = n_layers (default=(100,100))
            Size of the hidden layer
        activation : string (default='softplus')
            Activation function to be used, chosen from 'softplus', 'relu', 'sigmoid', 'leakyrelu', 'rrelu', 'elu',
            'selu' or 'tanh'
        dropout : float (default=0.2)
            Percent of neurons to drop out.
        learn_priors : bool, (default=True)
            If true, priors are made learnable parameters
        batch_size : int (default=64)
            Size of the batch to use for training
        lr: float (defualt=2e-3)
            Learning rate to be used for training
        momentum: folat (default=0.99)
            Momemtum to be used for training
        solver: string (default='adam')
            NN optimizer to be used, chosen from 'adagrad', 'adam', 'sgd', 'adadelta' or 'rmsprop' 
        num_epochs: int (default=100)
            Number of epochs to train for
        reduce_on_plateau: bool (default=False)
            If true, reduce learning rate by 10x on plateau of 10 epochs 
        topic_prior_mean: double (default=0.0)
            Mean parameter of the prior
        topic_prior_variance: double (default=None)
            Variance parameter of the prior
        num_samples: int (default=10)
            Number of times the theta needs to be sampled
        num_data_loader_workers: int (default=0)
            Number of subprocesses to use for data loading
        verbose: bool
            If True, additional logs are displayed
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)

        self._n_components = n_components
        self._model_type = model_type
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._dropout = dropout
        self._learn_priors = learn_priors
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = momentum
        self._solver = solver
        self._num_epochs = num_epochs
        self._reduce_on_plateau = reduce_on_plateau
        self._topic_prior_mean = topic_prior_mean
        self._topic_prior_variance = topic_prior_variance
        self._num_samples = num_samples
        self._num_data_loader_workers = num_data_loader_workers
        self._thetas_thr = thetas_thr

        return
    
    def _createTMmodel(self, modelFolder, avitm):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained using ProdLDA topic modeling and whose
        output is available at the provided folder

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel

        """

        # Get thetas
        thetas32 = np.asarray(
            avitm.get_doc_topic_distribution(avitm.train_data))  # .T

        # Sparsification of thetas matrix
        self.logger.debug('-- -- Sparsifying doc-topics matrix')
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32)
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._sparse_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Calculate beta matrix
        betas = avitm.get_topic_word_distribution()

        # Create vocabulary files and calculate beta matrix
        betas = avitm.get_topic_word_distribution()
        vocab_size = betas.shape[1]
        vocab = []
        term_freq = np.zeros((vocab_size,))

        for top in np.arange(self._n_components):
            for idx, word in self._id2token.items():
                vocab.append(word)
                cnt = betas[top][idx]
                term_freq[idx] += cnt  # @TODO: This is not the freq
        # Save vocabulary and frequencies
        vocabfreq_file = modelFolder.joinpath('vocab_freq_prod.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.debug('-- -- ProdLDA training: Vocabulary file generated')

        tm = TMmodel(betas=betas, thetas=thetas32, alphas=alphas,
                     vocabfreq_file=vocabfreq_file)

        return tm

    def fit(self, corpusFile):
        """
        Training of ProdLDA Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """

        # Output model folder and training file for the corpus
        if not corpusFile.is_file():
            self._logger.error(
                f'-- -- Provided corpus Path does not exist -- Stop')
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('prodLDAModel')
        modelFolder.mkdir()

        # Generating the corpus in the input format required by ProdLDA
        self.logger.info('-- -- ProdLDA Corpus Generation: BOW Dataset object')
        df = pd.read_parquet(corpusFile)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]       

        self._corpus = [el for el in df_lemas]
        self._train_dataset, self._val_dataset, self._bow_size, self._id2token = \
            prepare_dataset(self._corpus)
       
        data = [self._train_dataset, self._val_dataset, self._bow_size, self._id2token]
        bowdataset_file = modelFolder.joinpath('bowdataset.pickle')
        with open(bowdataset_file, 'wb') as f:
            pickle.dump(data, f)
        
        avitm = AVITM(logger=self.logger,
                      input_size=self._bow_size,
                      n_components=self._n_components,
                      model_type=self._model_type,
                      hidden_sizes=self._hidden_sizes,
                      activation=self._activation,
                      dropout=self._dropout,
                      learn_priors=self._learn_priors,
                      batch_size=self._batch_size,
                      lr=self._lr,
                      momentum=self._momentum,
                      solver=self._solver,
                      num_epochs=self._num_epochs,
                      reduce_on_plateau=self._reduce_on_plateau,
                      topic_prior_mean=self._topic_prior_mean,
                      topic_prior_variance=self._topic_prior_variance,
                      num_samples=self._num_samples,
                      num_data_loader_workers=self._num_data_loader_workers)

        avitm.fit(self._train_dataset, self._val_dataset)

        # Create TMmodel object
        tm = self._createTMmodel(modelFolder, avitm)
        tm.save_npz(corpusFile.parent.joinpath('model.npz'))

        return

class CTMTrainer(Trainer):
    """
    Wrapper for the CTM Topic Model Training. Implements the
    following functionalities
    - Transformation of the corpus to the CTM internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(self, inference_type="combined", n_components=10, model_type='prodLDA',
                hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99, 
                solver='adam', num_epochs=100, num_samples=10, reduce_on_plateau=False,
                topic_prior_mean=0.0, topic_prior_variance=None,
                num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None,
                thetas_thr=0.003, sbert_model_to_load='paraphrase-distilroberta-base-v1',
                logger=None, ):
        """
        Initilization Method

        Parameters
        ----------
        inference_type: string (default='combined')
            Type of inference network to be used. It can be either 'combined' or 'contextual'
        n_components : int (default=10)
            Number of topic components
        model_type : string (default='prodLDA')
            Type of the model that is going to be trained, 'prodLDA' or 'LDA'
        hidden_sizes : tuple, length = n_layers (default=(100,100))
            Size of the hidden layer
        activation : string (default='softplus')
            Activation function to be used, chosen from 'softplus', 'relu', 'sigmoid', 'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'
        dropout : float (default=0.2)
            Percent of neurons to drop out.
        learn_priors : bool, (default=True)
            If true, priors are made learnable parameters
        batch_size : int (default=64)
            Size of the batch to use for training
        lr: float (defualt=2e-3)
            Learning rate to be used for training
        momentum: folat (default=0.99)
            Momemtum to be used for training
        solver: string (default='adam')
            NN optimizer to be used, chosen from 'adagrad', 'adam', 'sgd', 'adadelta' or 'rmsprop' 
        num_epochs: int (default=100)
            Number of epochs to train for
        num_samples: int (default=10)
            Number of times the theta needs to be sampled
        reduce_on_plateau: bool (default=False)
            If true, reduce learning rate by 10x on plateau of 10 epochs 
        topic_prior_mean: double (default=0.0)
            Mean parameter of the prior
        topic_prior_variance: double (default=None)
            Variance parameter of the prior
        num_data_loader_workers: int (default=0)
            Number of subprocesses to use for data loading
        label_size: int (default=0)
            Number of total labels
        loss_weights: dict (default=None)
            It contains the name of the weight parameter (key) and the weight (value) for each loss.
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        logger: Logger object
            To log object activity
        sbert_model_to_load: str (default='paraphrase-distilroberta-base-v1')
            Model to be used for calculating the embeddings
        """

        super().__init__(logger)

        self._inference_type = inference_type
        self._n_components = n_components
        self._model_type = model_type
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._dropout = dropout
        self._learn_priors = learn_priors
        self._batch_size = batch_size
        self._lr = lr
        self._momentum = momentum
        self._solver = solver
        self._num_epochs = num_epochs
        self._reduce_on_plateau = reduce_on_plateau
        self._topic_prior_mean = topic_prior_mean
        self._topic_prior_variance = topic_prior_variance
        self._num_samples = num_samples
        self._num_data_loader_workers = num_data_loader_workers
        self._label_size = label_size
        self._sbert_model_to_load = sbert_model_to_load
        self._loss_weights = loss_weights
        self._thetas_thr = thetas_thr

        return

    def _createTMmodel(self, modelFolder, ctm):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained using ProdLDA topic modeling and whose
        output is available at the provided folder

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel
        """

        # Get thetas
        thetas32 = np.asarray(ctm.get_doc_topic_distribution(ctm.train_data))

        # Sparsification of thetas matrix
        self.logger.debug('-- -- Sparsifying doc-topics matrix')
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32)
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._sparse_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Calculate beta matrix
        betas = ctm.get_topic_word_distribution()

        # Create vocabulary files and calculate beta matrix
        vocab_size = betas.shape[1]
        vocab = []
        term_freq = np.zeros((vocab_size,))

        for top in np.arange(self._n_components):
            for idx, word in self._id2token.items():
                vocab.append(word)
                cnt = betas[top][idx]
                term_freq[idx] += cnt  # Cuántas veces aparece una palabra

        vocabfreq_file = modelFolder.joinpath('vocab_freq_ctm.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.debug('-- -- CTM training: Vocabulary file generated')

        tm = TMmodel(betas=betas, thetas=thetas32, alphas=alphas,
                     vocabfreq_file=vocabfreq_file)

        return tm

    def fit(self, corpusFile):
        """
        Training of CTM Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """

        # Output model folder and training file for the corpus
        if not corpusFile.is_file():
            self._logger.error(
                f'-- -- Provided corpus Path does not exist -- Stop')
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('ctmModel')
        modelFolder.mkdir()

        # Generating the corpus in the input format required by ProdLDA
        self.logger.info('-- -- CTM Corpus Generation: BOW Dataset object')
        df = pd.read_parquet(corpusFile)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]
        df_raw = df[["all_rawtext"]].values.tolist()
        df_raw = [doc[0].split() for doc in df_raw]
        self._corpus = [el for el in df_lemas]
        self._unpreprocessed_corpus = [el for el in df_raw]

        # Generate the corpus in the input format required by CTM
        self._train_dataset, self._val_dataset, self._input_size, self._id2token, self._qt, self.embeddings_train, custom_embeddings, self.docs_train = \
            prepare_ctm_dataset(corpus=self._corpus,
                                unpreprocessed_corpus=self._unpreprocessed_corpus,
                                sbert_model_to_load=self._sbert_model_to_load)
        
        data = [self._train_dataset, self._val_dataset, self._input_size, self._id2token]
        ctm_dataset_file = modelFolder.joinpath('ctm_dataset.pickle')
        with open(ctm_dataset_file, 'wb') as f:
            pickle.dump(data, f)
        
        ctm = CombinedTM(
            logger=self.logger,
            input_size=self._input_size,
            contextual_size=768,
            n_components=self._n_components,
            model_type=self._model_type,
            hidden_sizes=self._hidden_sizes,
            activation=self._activation,
            dropout=self._dropout,
            learn_priors=self._learn_priors,
            batch_size=self._batch_size,
            lr=self._lr,
            momentum=self._momentum,
            solver=self._solver,
            num_epochs=self._num_epochs,
            num_samples=self._num_samples,
            reduce_on_plateau=self._reduce_on_plateau,
            topic_prior_mean=self._topic_prior_mean,
            topic_prior_variance=self._topic_prior_variance,
            num_data_loader_workers=self._num_data_loader_workers)

        ctm.fit(self._train_dataset, self._val_dataset)

        # Create TMmodel object
        tm = self._createTMmodel(modelFolder, ctm)
        tm.save_npz(corpusFile.parent.joinpath('model.npz'))

        return

##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Topic modeling utilities')
    parser.add_argument('--spark', action='store_true', default=False,
                        help='Indicate that spark cluster is available',
                        required=False)
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Preprocess training data according to config file")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train Topic Model according to config file")
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")
    args = parser.parse_args()

    if args.spark:

        # Spark imports and session generation
        import pyspark.sql.functions as F
        from pyspark.ml.feature import (CountVectorizer, StopWordsRemover,
                                        Tokenizer)
        from pyspark.sql import SparkSession

        spark = SparkSession\
            .builder\
            .appName("Topicmodeling")\
            .getOrCreate()

    else:
        spark = None

    # If the preprocessing flag is activated, we need to check availability of
    # configuration file, and run the preprocessing of the training data using
    # the textPreproc class
    if args.preproc:
        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                train_config = json.load(fin)

            """
            Data preprocessing This part of the code will preprocess all the
            documents that are available in the training dataset and generate
            also the necessary objects for preprocessing objects during inference
            """

            tPreproc = textPreproc(stw_files=train_config['Preproc']['stopwords'],
                                   eq_files=train_config['Preproc']['equivalences'],
                                   min_lemas=train_config['Preproc']['min_lemas'],
                                   no_below=train_config['Preproc']['no_below'],
                                   no_above=train_config['Preproc']['no_above'],
                                   keep_n=train_config['Preproc']['keep_n'])

            # Create a Dataframe with all training data
            trDtFile = Path(train_config['TrDtSet'])
            with trDtFile.open() as fin:
                trDtSet = json.load(fin)

            if args.spark:
                # Read all training data and configure them as a spark dataframe
                for idx, DtSet in enumerate(trDtSet['Dtsets']):
                    df = spark.read.parquet(f"file://{DtSet['parquet']}")
                    if len(DtSet['filter']):
                        # To be implemented
                        # Needs a spark command to carry out the filtering
                        # df = df.filter ...
                        pass
                    df = (
                        df.withColumn("all_lemmas", F.concat_ws(
                            ' ', *DtSet['lemmasfld']))
                          .withColumn("all_rawtext", F.concat_ws(' ', *DtSet['rawtxtfld']))
                          .withColumn("source", F.lit(DtSet["source"]))
                          .select("id", "source", "all_lemmas", "all_rawtext")
                    )
                    if idx == 0:
                        trDF = df
                    else:
                        trDF = trDF.union(df).distinct()

                # We preprocess the data and save the CountVectorizer Model used to obtain the BoW
                trDF = tPreproc.preprocBOW(trDF)
                tPreproc.saveCntVecModel(configFile.parent.resolve())
                trDataFile = tPreproc.exportTrData(trDF=trDF,
                                                   dirpath=configFile.parent.resolve(),
                                                   tmTrainer=train_config['trainer'])
                sys.stdout.write(trDataFile.as_posix())

            else:
                # TBD Preprocess training data with Gensim
                pass
        else:
            sys.exit('You need to provide a valid configuration file')

    # If the training flag is activated, we need to check availability of
    # configuration file, and run the topic model training
    if args.train:
        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                train_config = json.load(fin)

                if train_config['trainer'] == 'mallet':
                    MallTr = MalletTrainer(mallet_path=train_config['LDAparam']['mallet_path'],
                                           ntopics=train_config['LDAparam']['ntopics'],
                                           alpha=train_config['LDAparam']['alpha'],
                                           optimize_interval=train_config['LDAparam']['optimize_interval'],
                                           num_threads=train_config['LDAparam']['num_threads'],
                                           num_iterations=train_config['LDAparam']['num_iterations'],
                                           doc_topic_thr=train_config['LDAparam']['doc_topic_thr'],
                                           thetas_thr=train_config['LDAparam']['thetas_thr'],
                                           token_regexp=train_config['LDAparam']['token_regexp'])
                    MallTr.fit(
                        corpusFile=configFile.parent.joinpath('corpus.txt'))
                elif train_config['trainer'] == 'sparkLDA':
                    if not args.spark:
                        sys.stodout.write(
                            "You need access to a spark cluster to run sparkLDA")
                        sys.exit(
                            "You need access to a spark cluster to run sparkLDA")
                    sparkLDATr = sparkLDATrainer()
                    sparkLDATr.fit(
                        configFile.parent.joinpath('corpus.parquet'))
                elif train_config['trainer'] == 'prodLDA':
                    ProdLDATr = ProdLDATrainer(
                                n_components=train_config['PRODparam']['n_components'],
                                model_type=train_config['PRODparam']['model_type'],
                                hidden_sizes=train_config['PRODparam']['hidden_sizes'],
                                activation=train_config['PRODparam']['activation'],
                                dropout=train_config['PRODparam']['dropout'],
                                learn_priors=train_config['PRODparam']['learn_priors'],
                                batch_size=train_config['PRODparam']['batch_size'],
                                lr=train_config['PRODparam']['lr'],
                                momentum=train_config['PRODparam']['momentum'],
                                solver=train_config['PRODparam']['solver'],
                                num_epochs=train_config['PRODparam']['num_epochs'],
                                reduce_on_plateau=train_config['PRODparam']['reduce_on_plateau'],
                                topic_prior_mean=train_config['PRODparam']['topic_prior_mean'],
                                topic_prior_variance=train_config['PRODparam']['topic_prior_variance'],
                                num_samples=train_config['PRODparam']['num_samples'],
                                num_data_loader_workers=train_config['PRODparam']['num_data_loader_workers'],
                                thetas_thr=train_config['PRODparam']['thetas_thr'])
                    ProdLDATr.fit()
                elif train_config['trainer'] == 'ctm':
                    CTMr = CTMTrainer(
                            inference_type=train_config['CTMparam']['inference_type'],
                            n_components=train_config['CTMparam']['n_components'],
                            model_type=train_config['CTMparam']['model_type'],
                            hidden_sizes=train_config['CTMparam']['hidden_sizes'],
                            activation=train_config['CTMparam']['activation'],
                            dropout=train_config['CTMparam']['dropout'],
                            learn_priors=train_config['CTMparam']['learn_priors'],
                            batch_size=train_config['CTMparam']['batch_size'],
                            lr=train_config['CTMparam']['lr'],
                            momentum=train_config['CTMparam']['momentum'],
                            solver=train_config['CTMparam']['solver'],
                            num_epochs=train_config['CTMparam']['num_epochs'],
                            num_samples=train_config['CTMparam']['num_samples'],
                            reduce_on_plateau=train_config['CTMparam']['reduce_on_plateau'],
                            topic_prior_mean=train_config['CTMparam']['topic_prior_mean'],
                            topic_prior_variance=train_config['CTMparam']['topic_prior_variance'],
                            num_data_loader_workers=train_config['CTMparam']['num_data_loader_workers'],
                            label_size=train_config['CTMparam']['label_size'],
                            loss_weights=train_config['CTMparam']['loss_weights'],
                            thetas_thr=train_config['CTMparam']['thetas_thr'],
                            sbert_model_to_load=train_config['CTMparam']['sbert_model_to_load'])
                    CTMr.fit()

        else:
            sys.exit('You need to provide a valid configuration file')
