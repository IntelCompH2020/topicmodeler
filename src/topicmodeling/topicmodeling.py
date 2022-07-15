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
import os
import pickle
import shutil
import sys
from abc import abstractmethod
from pathlib import Path
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy import sparse
from sklearn.preprocessing import normalize
from gensim import corpora

from neural_models.contextualized_topic_models.ctm_network.ctm import CombinedTM, ZeroShotTM
from neural_models.contextualized_topic_models.utils.data_preparation import prepare_ctm_dataset
from neural_models.pytorchavitm.utils.data_preparation import prepare_dataset
from neural_models.pytorchavitm.avitm_network.avitm import AVITM

from manageModels import TMmodel as newTMmodel

"""
# from scipy.spatial.distance import jensenshannon
# import pyLDAvis
import logging
import regex as javare
import re
from tqdm import tqdm
import ipdb
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
                 keep_n=100000, cntVecModel=None,
                 GensimDict=None, logger=None):
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
        GensimDict : gensim.corpora.Dictionary
            Optimized Gensim Dictionary Object
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
        self._GensimDict = GensimDict

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
        if isinstance(trDF, dd.DataFrame):

            def tkz_clean_str(rawtext):
                """Function to carry out tokenization and cleaning of text

                Parameters
                ----------
                rawtext: str
                    string with the text to lemmatize

                Returns
                -------
                cleantxt: str
                    Cleaned text
                """
                if rawtext == None or rawtext == '':
                    return ''
                else:
                    # lowercase and tokenization (similar to Spark tokenizer)
                    cleantext = rawtext.lower().split()
                    # remove stopwords
                    cleantext = [
                        el for el in cleantext if el not in self._stopwords]
                    # replacement of equivalent words
                    cleantext = [self._equivalents[el] if el in self._equivalents else el
                                 for el in cleantext]
                return cleantext

            # Compute tokens, clean them, and filter out documents
            # with less than minimum number of lemmas
            trDF['final_tokens'] = trDF['all_lemmas'].apply(
                tkz_clean_str, meta=('all_lemmas', 'object'))
            trDF = trDF.loc[trDF.final_tokens.apply(
                len, meta=('final_tokens', 'int64')) >= self._min_lemas]

            # Gensim dictionary creation. It persists the created Dataframe
            # to accelerate dictionary calculation
            # Filtering of words is carried out according to provided values
            self._logger.info('-- -- Gensim Dictionary Generation')

            with ProgressBar():
                DFtokens = trDF[['final_tokens']]
                DFtokens = DFtokens.compute(scheduler='processes')
            self._GensimDict = corpora.Dictionary(
                DFtokens['final_tokens'].values.tolist())

            # Remove words that appear in less than no_below documents, or in more than
            # no_above, and keep at most keep_n most frequent terms

            self._logger.info('-- -- Gensim Filter Extremes')

            self._GensimDict.filter_extremes(no_below=self._no_below,
                                             no_above=self._no_above, keep_n=self._keep_n)

            # We skip the calculation of the bow for each document, because Spark LDA will
            # not be used in this case. Note that this is different from what is done for
            # Spark preprocessing
            trDFnew = trDF

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

    def saveGensimDict(self, dirpath):
        """
        Saves a Gensim Dictionary to the specified path
        Saves also a text document with the corresponding
        vocabulary

        Parameters
        ----------
        dirpath: pathlib.Path
            The folder where the Gensim dictionary and the
            text file with the vocabulary will be saved

        Returns
        -------
        status: int
            - 1: If the files were generated sucessfully
            - 0: Error (Gensim dictionary does not exist)
        """
        if self._GensimDict:
            GensimFile = dirpath.joinpath('dictionary.gensim')
            if GensimFile.is_file():
                GensimFile.unlink()
            self._GensimDict.save_as_text(GensimFile)
            with dirpath.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
                fout.write(
                    '\n'.join([self._GensimDict[idx] for idx in range(len(self._GensimDict))]))
            return 1
        else:
            return 0

    def exportTrData(self, trDF, dirpath, tmTrainer):
        """
        Exports the training data in the provided dataset to the
        format required by the topic modeling trainer

        Parameters
        ----------
        trDF: Dask or Spark dataframe
            If Spark, the dataframe should contain a column "bow" that will
            be used to calculate the training data
            If Dask, it should contain a column "final_tokens"
        dirpath: pathlib.Path
            The folder where the data will be saved
        tmTrainer: string
            The output format [mallet|sparkLDA|prodLDA|ctm]

        Returns
        -------
        outFile: Path
            A path containing the location of the training data in the indicated format
        """

        self._logger.info(f'-- -- Exporting corpus to {tmTrainer} format')

        if isinstance(trDF, dd.DataFrame):
            # Dask dataframe

            # Remove words not in dictionary, and return a string
            vocabulary = set([self._GensimDict[idx]
                             for idx in range(len(self._GensimDict))])

            def tk_2_text(tokens):
                """Function to filter words not in dictionary, and
                return a string of lemmas 

                Parameters
                ----------
                tokens: list
                    list of "final_tokens"

                Returns
                -------
                lemmasstr: str
                    Clean text including only the lemmas in the dictionary
                """
                #bow = self._GensimDict.doc2bow(tokens)
                # return ''.join([el[1] * (self._GensimDict[el[0]]+ ' ') for el in bow])
                return ' '.join([el for el in tokens if el in vocabulary])

            trDF['cleantext'] = trDF['final_tokens'].apply(
                tk_2_text, meta=('final_tokens', 'str'))

            if tmTrainer == "mallet":

                outFile = dirpath.joinpath('corpus.txt')
                if outFile.is_file():
                    outFile.unlink()

                trDF['2mallet'] = trDF['id'].apply(
                    str, meta=('id', 'str')) + " 0 " + trDF['cleantext']

                with ProgressBar():
                    #trDF = trDF.persist(scheduler='processes')
                    DFmallet = trDF[['2mallet']]
                    DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                    compute_kwargs={'scheduler': 'processes'})

            elif tmTrainer == 'sparkLDA':
                self._logger.error(
                    '-- -- sparkLDA requires preprocessing with spark')
                return
            elif tmTrainer == "prodLDA":

                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = trDF[['id', 'cleantext']].rename(
                        columns={"cleantext": "bow_text"})
                    DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                                         'scheduler': 'processes'})

            elif tmTrainer == "ctm":
                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = trDF[['id', 'cleantext', 'all_rawtext']].rename(
                        columns={"cleantext": "bow_text"})
                    DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                                         'scheduler': 'processes'})

        else:
            # Spark dataframe
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
                outFile = dirpath.joinpath('corpus.parquet')
                lemas_df = (trDF.withColumn("bow_text", back2textUDF(
                    F.col("bow"))).select("id", "bow_text"))
                lemas_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")
            elif tmTrainer == "ctm":
                outFile = dirpath.joinpath('corpus.parquet')
                lemas_raw_df = (trDF.withColumn("bow_text", back2textUDF(
                    F.col("bow"))).select("id", "bow_text", "all_raw_text"))
                lemas_raw_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")

        return outFile


class TMmodel(object):
    # This class represents a Topic Model according to the LDA generative model
    # Essentially the TM is characterized by
    # _alphas: The weight of each topic
    # _betas: The weight of each word in the vocabulary
    # _thetas: The weight of each topic in each document
    #
    # The TM can be trained with Blei's LDA, Mallet, or any other toolbox that produces a model according to this representation

    # The following variables will store original values of matrices alphas, betas, thetas
    # They will be used to reset the model to original values
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
    _vocab = None
    _size_vocab = None

    def __init__(self, betas=None, thetas=None, alphas=None,
                 vocab=None, from_file=None, logger=None):
        """Topic model inititalization

        Inicializacion del model de topicos a partir de las matrices que lo caracterizan
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
        vocab:
            Vocabulary. List of words sorted according to betas matrix
        from_file:
            If not None, contains the name of a file from which the object can be initialized
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
        if from_file:
            from_file = Path(from_file)

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

    def get_vocab_id2w(self):
        return self._vocab_id2w

    def get_vocab_w2id(self):
        return self._vocab_w2id

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
        """Creates an object of class TMmodel hosting the topic model
        that has been trained and whose output is available at the
        provided folder

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
        print(thetas32.shape)
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Create vocabulary files and calculate beta matrix
        # A vocabulary is available with words provided by the Count Vectorizer object, but the new files need the order used by mallet
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
        vocabfreq_file = modelFolder.joinpath('vocab_freq.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.debug('-- -- Mallet training: Vocabulary file generated')

        tm = newTMmodel(modelFolder.parent.joinpath('TMmodel'))
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

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

        modelFolder = corpusFile.parent.joinpath('MalletFiles')
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
        self._logger.debug('-- -- Sparsifying doc-topics matrix')
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32, modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Calculate beta matrix
        betas = avitm.get_topic_word_distribution()
        self._logger.info(betas.shape)

        # Create vocabulary list and calculate beta matrix
        betas = avitm.get_topic_word_distribution()
        vocab = self._train_dataset.idx2token
        #for top in np.arange(self._n_components):
        #    for idx, word in self._id2token.items():
        #        vocab.append(word)
        self._logger.info(len(vocab))
            
        tm = newTMmodel(modelFolder.parent.joinpath('TMmodel'))
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

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
        if not os.path.exists(corpusFile):
            self._logger.error(
                f'-- -- Provided corpus Path does not exist -- Stop')
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir()

        # Generating the corpus in the input format required by ProdLDA
        self._logger.info(
            '-- -- ProdLDA Corpus Generation: BOW Dataset object')
        df = pd.read_parquet(corpusFile)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]

        self._corpus = [el for el in df_lemas]
        self._train_dataset, self._val_dataset, self._bow_size, self._id2token, self._docs_train = \
            prepare_dataset(self._corpus)

        # Save training corpus
        corpus_file = modelFolder.joinpath('corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as fout:
            id = 0
            for el in self._docs_train:
                fout.write(str(id) + ' 0 ' + ' '.join(el) + '\n')
                id += 1

        avitm = AVITM(logger=self._logger,
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

    def __init__(self, n_components=10, ctm_model_type='CombinedTM', model_type='prodLDA', hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_samples=10, reduce_on_plateau=False, topic_prior_mean=0.0, topic_prior_variance=None, num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None, thetas_thr=0.003, sbert_model_to_load='paraphrase-distilroberta-base-v1', logger=None):
        """
        Initilization Method

        Parameters
        ----------
        n_components : int (default=10)
            Number of topic components
        model_type : string (default='prodLDA')
            Type of the model that is going to be trained, 'prodLDA' or 'LDA'
        ctm_model_type : string (default='CombinedTM')
            CTM model that is going to used for training
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
        sbert_model_to_load: str (default='paraphrase-distilroberta-base-v1')
            Model to be used for calculating the embeddings
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)

        self._n_components = n_components
        self._model_type = model_type
        self._ctm_model_type = ctm_model_type
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
        self._logger.debug('-- -- Sparsifying doc-topics matrix')
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32, modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
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

        vocabfreq_file = modelFolder.joinpath('vocab_freq.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.debug('-- -- CTM training: Vocabulary file generated')

        tm = TMmodel(betas=betas, thetas=thetas32, alphas=alphas,
                     vocabfreq_file=vocabfreq_file)

        return tm

    def fit(self, corpusFile, embeddingsFile=None):
        """
        Training of CTM Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """

        # Output model folder and training file for the corpus
        if not os.path.exists(corpusFile):
            self._logger.error(
                f'-- -- Provided corpus Path does not exist -- Stop')
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir()

        # Generating the corpus in the input format required by ProdLDA
        self._logger.info('-- -- CTM Corpus Generation: BOW Dataset object')
        df = pd.read_parquet(corpusFile)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]
        self._corpus = [el for el in df_lemas]

        if embeddingsFile is None:
            df_raw = df[["all_rawtext"]].values.tolist()
            df_raw = [doc[0].split() for doc in df_raw]
            self._unpreprocessed_corpus = [el for el in df_raw]
            self._embeddings = None
        else:
            if not embeddingsFile.is_file():
                self._logger.error(
                    f'-- -- Provided embeddings Path does not exist -- Stop')
                sys.exit()
            self._embeddings = np.load(embeddingsFile, allow_pickle=True)
            self._unpreprocessed_corpus = None

        # Generate the corpus in the input format required by CTM
        self._train_dts, self._val_dts, self._input_size, self._id2token, _, self._embeddings_train, _, self._docs_train = \
            prepare_ctm_dataset(corpus=self._corpus,
                                unpreprocessed_corpus=self._unpreprocessed_corpus,
                                custom_embeddings=self._embeddings,
                                sbert_model_to_load=self._sbert_model_to_load)

        # Save embeddings
        embeddings_file = modelFolder.joinpath('embeddings.npy')
        np.save(embeddings_file, self._embeddings_train)

        # Save training corpus
        corpus_file = modelFolder.joinpath('corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as fout:
            id = 0
            for el in self._docs_train:
                fout.write(str(id) + ' 0 ' + ' '.join(el) + '\n')
                id += 1

        if self._ctm_model_type == 'ZeroShotTM':
            ctm = ZeroShotTM(
                logger=self._logger,
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
        else:
            ctm = CombinedTM(
                logger=self._logger,
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
                num_data_loader_workers=self._num_data_loader_workers,
                label_size=self._label_size,
                loss_weights=self._loss_weights)

        ctm.fit(self._train_dts, self._val_dts)

        # Create TMmodel object
        tm = self._createTMmodel(modelFolder, ctm)
        tm.save_npz(corpusFile.parent.joinpath('model.npz'))

        return


class HierarchicalTMManager(object):

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
            self._logger = logging.getLogger('HierarchicalTMManager')

        return

    def create_submodel_tr_corpus(self, TMmodel_path, configFile_f, configFile_c):
        """

        Parameters
        ----------
        TMmodel_path: str
            Path to the TModel object associated with the father model
        train_config_f: str
            Father model's configuration file' s path
        train_config_c: str
            Submodel's configuration file' s path
        """

        # Load TMmodel
        configFile_c = Path(configFile_c)
        configFile_f = Path(configFile_f)
        vocabFile = configFile_f.parent.joinpath(
            'modelFiles/vocab_freq.txt')
        tmmodel = TMmodel(vocabfreq_file=vocabFile,
                          from_file=TMmodel_path)

        # Read training configurations from father model and submodel
        with configFile_f.open('r', encoding='utf8') as fin:
            tr_config_f = json.load(fin)
        configFile_c = Path(configFile_c)
        with configFile_c.open('r', encoding='utf8') as fin:
            tr_config_c = json.load(fin)

        # Get father model's training corpus as dask dataframe
        if tr_config_f['trainer'] == "ctm" or  tr_config_f['trainer'] == "prodLDA":
            corpusFile = configFile_f.parent.joinpath('modelFiles/corpus.txt')
            self._logger.info(corpusFile)
        else:
            corpusFile = configFile_f.parent.joinpath('corpus.txt')
        corpus = [line.rsplit(' 0 ')[1].strip() for line in open(
            corpusFile, encoding="utf-8").readlines()]
        tr_data_df = pd.DataFrame(data=corpus, columns=['doc'])
        tr_data_df['id'] = range(1, len(tr_data_df) + 1)
        tr_data_ddf = dd.from_pandas(tr_data_df, npartitions=2)

        # Get embeddings if the trainer is CTM
        if tr_config_f['trainer'] == "ctm":
            embeddingsFile = configFile_f.parent.joinpath('modelFiles/embeddings.npy')
            embeddings = np.load(embeddingsFile, allow_pickle=True)

        # Get father model's thetas and betas and expansion topic
        thetas = tmmodel.get_thetas().toarray()  # (ndocs, ntopics)
        betas = tmmodel.get_betas()  # (ntopics, nwords)
        vocab_id2w = tmmodel.get_vocab_id2w()
        vocab_w2id = tmmodel.get_vocab_w2id()
        exp_tpc = int(tr_config_c['expansion_tpc'])

        if tr_config_c['htm-version'] == "htm-ws":
            self._logger.info(
                '-- -- -- Creating training corpus according to HTM-WS.')

            def get_htm_ws_corpus(row, thetas, betas, vocab_id2w, vocab_w2id, exp_tpc):
                """Function to carry out the selection of words according to HTM-WS.

                Parameters
                ----------
                row: pandas.Series
                    ndarray representation of the document
                thetas: ndarray
                    Document-topic distribution
                betas: ndarray
                    Word-topic distribution
                vocab_id2w: dict
                    Dictionary in the form {i: word_i}
                exp_tpc: int
                    Expansion topic

                Returns
                -------
                reduced_doc_str: str
                    String representation of the words to keep in the document given by row
                """

                id_doc = int(row["id"]) - 1
                doc = row["doc"].split()
                thetas_d = thetas[id_doc, :]

                # ids of words in d
                words_doc_idx = [vocab_w2id[word]
                                 for word in doc if word in vocab_w2id]

                # ids of words in d assigned to exp_tpc
                words_exp_idx = [idx_w for idx_w in words_doc_idx if np.argmax(
                    np.multiply(thetas_d, betas[:, idx_w])) == exp_tpc]

                #words_exp_idx = [idx_w for idx_w in range(betas.shape[1]) if #np.nonzero(np.random.multinomial(len(betas), np.multiply#(thetas_d, betas[:, idx_w])))[0][0] == exp_tpc]

                # Only words generated by exp_tpc are kept
                reduced_doc = [vocab_id2w[str(id_word)]
                               for id_word in words_exp_idx]
                reduced_doc_str = ' '.join([el for el in reduced_doc])

                return reduced_doc_str

            tr_data_ddf['reduced_doc'] = tr_data_ddf.apply(
                get_htm_ws_corpus, axis=1, meta=('x', 'object'), args=(thetas, betas, vocab_id2w, vocab_w2id, exp_tpc))

            if tr_config_c['trainer'] == "mallet":

                outFile = configFile_c.parent.joinpath('corpus.txt')
                if outFile.is_file():
                    outFile.unlink()

                tr_data_ddf['2mallet'] = tr_data_ddf['id'].apply(
                    str, meta=('id', 'str')) + " 0 " + tr_data_ddf['reduced_doc']

                with ProgressBar():
                    DFmallet = tr_data_ddf[['2mallet']]
                    DFmallet.to_csv(
                        outFile, index=False,
                        header=False, single_file=True,
                        compute_kwargs={'scheduler': 'processes'})

            elif tr_config_c['trainer'] == 'sparkLDA':
                pass

            elif tr_config_c['trainer'] == "prodLDA" or tr_config_c['trainer'] == "ctm":

                outFile = configFile_c.parent.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = tr_data_ddf[['id', 'reduced_doc']].rename(
                        columns={"reduced_doc": "bow_text"})
                    DFparquet.to_parquet(
                        outFile, write_index=False,
                        compute_kwargs={'scheduler': 'processes'})

        elif tr_config_c['htm-version'] == "htm-ds":
            self._logger.info(
                '-- -- -- Creating training corpus according to HTM-DS.')

            # Get ids of documents that meet the condition of having a representation of the expansion topic larger than thr
            thr = float(tr_config_c['thr'])
            doc_ids_to_keep = \
                [idx for idx in range(thetas.shape[0])
                 if thetas[idx, exp_tpc] > thr]

            # Keep selected documents from the father's corpus
            tr_data_ddf = tr_data_ddf.loc[doc_ids_to_keep, :]

            # Save corpus file in the format required by each trainer
            if tr_config_c['trainer'] == "mallet":

                outFile = configFile_c.parent.joinpath('corpus.txt')
                if outFile.is_file():
                    outFile.unlink()

                tr_data_ddf['2mallet'] = tr_data_ddf['id'].apply(
                    str, meta=('id', 'str')) + " 0 " + tr_data_ddf['doc']

                with ProgressBar():
                    DFmallet = tr_data_ddf[['2mallet']]
                    DFmallet.to_csv(outFile, index=False, header=False, single_file=True, compute_kwargs={
                                    'scheduler': 'processes'})

            elif tr_config_c['trainer'] == 'sparkLDA':
                pass

            elif tr_config_c['trainer'] == "prodLDA" or tr_config_c['trainer'] == "ctm":

                outFile = configFile_c.parent.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = tr_data_ddf[['id', 'doc']].rename(
                        columns={"doc": "bow_text"})
                    DFparquet.to_parquet(
                        outFile, write_index=False,
                        compute_kwargs={'scheduler': 'processes'})

            if tr_config_c['trainer'] == "ctm":
                embeddings = embeddings[doc_ids_to_keep, :]

        else:
            self._logger.error(
                '-- -- -- The specified HTM version is not available.')
        
                # If the trainer is CTM, keep embeddings related to the selected documents t
        
        if tr_config_c['trainer'] == "ctm":
            # Save embeddings
            embeddings_file = configFile_c.parent.joinpath('embeddings.npy')
            np.save(embeddings_file, embeddings)

        
        return


##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Topic modeling utilities')
    parser.add_argument('--spark', action='store_true', default=False,
                        help='Indicate that spark cluster is available',
                        required=False)
    parser.add_argument('--path_models', type=str, default=None,
                        help="path to topic models folder")
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Preprocess training data according to config file")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train Topic Model according to config file")
    parser.add_argument('--hierarchical', action='store_true', default=False,
                        help='Create submodel training data according to config files', required=False)
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")
    parser.add_argument('--config_child', type=str, default=None,
                        help="Path to submodel's config file", required=False)
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
                # Read all training data and configure them as a dask dataframe
                for idx, DtSet in enumerate(trDtSet['Dtsets']):
                    df = dd.read_parquet(DtSet['parquet']).fillna("")
                    if len(DtSet['filter']):
                        # To be implemented
                        # Needs a dask command to carry out the filtering
                        # df = df.filter ...
                        pass
                    # Concatenate text fields
                    for idx2, col in enumerate(DtSet['lemmasfld']):
                        if idx2 == 0:
                            df["all_lemmas"] = df[col]
                        else:
                            df["all_lemmas"] += " " + df[col]
                    for idx2, col in enumerate(DtSet['rawtxtfld']):
                        if idx2 == 0:
                            df["all_rawtext"] = df[col]
                        else:
                            df["all_rawtext"] += " " + df[col]
                    df["source"] = DtSet["source"]
                    df = df[["id", "source", "all_lemmas", "all_rawtext"]]

                    # Concatenate dataframes
                    if idx == 0:
                        trDF = df
                    else:
                        trDF = dd.concat([trDF, df])

                #trDF = trDF.drop_duplicates(subset=["id"], ignore_index=True)
                # We preprocess the data and save the Gensim Model used to obtain the BoW
                trDF = tPreproc.preprocBOW(trDF)
                tPreproc.saveGensimDict(configFile.parent.resolve())
                trDataFile = tPreproc.exportTrData(trDF=trDF,
                                                   dirpath=configFile.parent.resolve(),
                                                   tmTrainer=train_config['trainer'])
                sys.stdout.write(trDataFile.as_posix())

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
                    MallTr = MalletTrainer(
                        mallet_path=train_config['TMparam']['mallet_path'],
                        ntopics=train_config['TMparam']['ntopics'],
                        alpha=train_config['TMparam']['alpha'],
                        optimize_interval=train_config['TMparam']['optimize_interval'],
                        num_threads=train_config['TMparam']['num_threads'],
                        num_iterations=train_config['TMparam']['num_iterations'],
                        doc_topic_thr=train_config['TMparam']['doc_topic_thr'],
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        token_regexp=train_config['TMparam']['token_regexp'])
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
                        n_components=train_config['TMparam']['ntopics'],
                        model_type=train_config['TMparam']['model_type'],
                        hidden_sizes=tuple(
                            train_config['TMparam']['hidden_sizes']),
                        activation=train_config['TMparam']['activation'],
                        dropout=train_config['TMparam']['dropout'],
                        learn_priors=train_config['TMparam']['learn_priors'],
                        batch_size=train_config['TMparam']['batch_size'],
                        lr=train_config['TMparam']['lr'],
                        momentum=train_config['TMparam']['momentum'],
                        solver=train_config['TMparam']['solver'],
                        num_epochs=train_config['TMparam']['num_epochs'],
                        reduce_on_plateau=train_config['TMparam']['reduce_on_plateau'],
                        topic_prior_mean=train_config['TMparam']['topic_prior_mean'],
                        topic_prior_variance=train_config['TMparam']['topic_prior_variance'],
                        num_samples=train_config['TMparam']['num_samples'],
                        num_data_loader_workers=train_config['TMparam']['num_data_loader_workers'],
                        thetas_thr=train_config['TMparam']['thetas_thr'])
                    ProdLDATr.fit(
                        corpusFile=configFile.parent.joinpath('corpus.parquet'))

                elif train_config['trainer'] == 'ctm':
                    CTMr = CTMTrainer(
                        n_components=train_config['TMparam']['ntopics'],
                        model_type=train_config['TMparam']['model_type'],
                        ctm_model_type=train_config['TMparam']['ctm_model_type'],
                        hidden_sizes=tuple(
                            train_config['TMparam']['hidden_sizes']),
                        activation=train_config['TMparam']['activation'],
                        dropout=train_config['TMparam']['dropout'],
                        learn_priors=train_config['TMparam']['learn_priors'],
                        batch_size=train_config['TMparam']['batch_size'],
                        lr=train_config['TMparam']['lr'],
                        momentum=train_config['TMparam']['momentum'],
                        solver=train_config['TMparam']['solver'],
                        num_epochs=train_config['TMparam']['num_epochs'],
                        num_samples=train_config['TMparam']['num_samples'],
                        reduce_on_plateau=train_config['TMparam']['reduce_on_plateau'],
                        topic_prior_mean=train_config['TMparam']['topic_prior_mean'],
                        topic_prior_variance=train_config['TMparam']['topic_prior_variance'],
                        num_data_loader_workers=train_config['TMparam']['num_data_loader_workers'],
                        label_size=train_config['TMparam']['label_size'],
                        loss_weights=train_config['TMparam']['loss_weights'],
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        sbert_model_to_load=train_config['TMparam']['sbert_model_to_load'])

                    if Path(train_config['embeddings']).is_file():
                        CTMr.fit(
                            corpusFile=configFile.parent.joinpath('corpus.parquet'),
                            embeddingsFile=Path(train_config['embeddings']))
                    else:
                        CTMr.fit(
                            corpusFile=configFile.parent.joinpath('corpus.parquet'))

        else:
            sys.exit('You need to provide a valid configuration file')

    if args.hierarchical:
        if not args.config_child:
            sys.exit('You need to provide a configuration file for the submodel')
        else:
            configFile_f = Path(args.config)
            if not configFile_f.is_file():
                sys.exit('You need to provide a valid configuration file for the father model.')
            else:
                configFile_c = Path(args.config_child)
                if not configFile_c.is_file():
                    sys.exit('You need to provide a valid configuration file for the submodel.')
                else:
                    tMmodel_path = configFile_f.parent.joinpath('model.npz')
                    if not tMmodel_path.is_file():
                        sys.exit('There must exist a valid TMmodel file for the parent corpus')
                    
                    # Create hierarhicalTMManager object
                    hierarchicalTMManager = HierarchicalTMManager()
                    
                    # Create corpus
                    hierarchicalTMManager.create_submodel_tr_corpus(
                        tMmodel_path, args.config, args.config_child)
