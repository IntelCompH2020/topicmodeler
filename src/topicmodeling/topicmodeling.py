"""
* *IntelComp H2020 project*
* *Topic Modeling Toolbox*

Provides several classes for Topic Modeling
    - textPreproc: Preparation of datasets for training topic models, including
                   - string cleaning (stopword removal + equivalent terms)
                   - BoW calculation
    - Trainer:  Generic class for training a topic model from a given corpus and performing inference on a new unseen corpus. 
    The following classes, each of them representing a specific trainer, extend from it:
        * MalletTrainer
        * sparkLDATrainer
        * ProdLDATrainer
        * CTMTrainer
        * BERTopicTrainer
    - HierarchicalTMManager: Manages the creation of the corpus associated with a 2nd level hierarchical topic model
"""
import argparse
import ast
import gzip
import json
import os
import sys
from abc import abstractmethod
from pathlib import Path
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.preprocessing import normalize
from subprocess import check_output
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import pyarrow as pa
import shutil
from gensim import corpora
from bertopic import BERTopic
from bertopic.representation import (KeyBERTInspired, MaximalMarginalRelevance)
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from umap import UMAP
try:
    print(f"-- -- Importing from src")
    from src.topicmodeling.neural_models.pytorchavitm.avitm_network.avitm import AVITM
    from src.topicmodeling.neural_models.pytorchavitm.utils.data_preparation import prepare_dataset
    from src.topicmodeling.tm_utils import (file_lines, pickler, pickler_avitm_for_ewb_inferencer)
    from src.topicmodeling.neural_models.contextualized_topic_models.ctm_network.ctm import (
        CombinedTM, ZeroShotTM)
    from src.topicmodeling.neural_models.contextualized_topic_models.utils.data_preparation import \
        prepare_ctm_dataset
    from src.topicmodeling.manageModels import TMmodel
    DO_LOGGER = True
except:
    print(f"-- -- Importing from topicmodeling")
    from neural_models.pytorchavitm.avitm_network.avitm import AVITM
    from tm_utils import (file_lines, pickler, pickler_avitm_for_ewb_inferencer)
    from neural_models.contextualized_topic_models.ctm_network.ctm import (
        CombinedTM, ZeroShotTM)
    from neural_models.contextualized_topic_models.utils.data_preparation import \
        prepare_ctm_dataset
    from manageModels import TMmodel

def print_logs(message, do_logger=True, level="info", logger=None):
    if do_logger and logger:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
    else:
        print(message, flush=True)
    print (message,flush=True)

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

    def preprocBOW(self, trDF, nw=0):
        """
        Preprocesses the documents in the dataframe to carry
        out the following tasks
            - Filter out short documents (below min_lemas)
            - Cleaning of stopwords
            - Equivalent terms application
            - BoW calculation

        Parameters
        ----------
        trDF: Dask or Spark dataframe
            This routine works on the following column "all_lemmas"
            Other columns are left untouched
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

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
                    # remove stopwords again, in case equivalences introduced new stopwords
                    cleantext = [
                        el for el in cleantext if el not in self._stopwords]

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
            print_logs(f'-- -- Gensim Dictionary Generation',
                       do_logger=DO_LOGGER, level="info", logger=self._logger)

            with ProgressBar():
                DFtokens = trDF[['final_tokens']]
                if nw > 0:
                    DFtokens = DFtokens.compute(
                        scheduler='processes', num_workers=nw)
                else:
                    # Use Dask default (i.e., number of available cores)
                    DFtokens = DFtokens.compute(scheduler='processes')

            final_tokens = DFtokens['final_tokens'].values.tolist()
            if isinstance(final_tokens[0], str):
                final_tokens = [ast.literal_eval(doc) for doc in final_tokens]
            self._GensimDict = corpora.Dictionary(final_tokens)

            # Remove words that appear in less than no_below documents, or in more than
            # no_above, and keep at most keep_n most frequent terms
            print_logs(f'-- -- Gensim Filter Extremes',
                       do_logger=DO_LOGGER, level="info", logger=self._logger)

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
            else:
                # We need to create a copy of the tokens with the new name
                trDF = trDF.withColumn("final_tokens", trDF["clean_tokens"])

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

    def exportTrData(self, trDF, dirpath, tmTrainer, nw=0):
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
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

        Returns
        -------
        outFile: Path
            A path containing the location of the training data in the indicated format
        """

        print_logs(f'-- -- Exporting corpus to {tmTrainer} format',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)

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
                # bow = self._GensimDict.doc2bow(tokens)
                # return ''.join([el[1] * (self._GensimDict[el[0]]+ ' ') for el in bow])
                if isinstance(tokens, str):
                    tokens = ast.literal_eval(tokens)
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
                    # trDF = trDF.persist(scheduler='processes')
                    DFmallet = trDF[['2mallet']]
                    if nw > 0:
                        DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                        compute_kwargs={'scheduler': 'processes', 'num_workers': nw})
                    else:
                        # Use Dask default number of workers (i.e., number of cores)
                        DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                        compute_kwargs={'scheduler': 'processes'})

            elif tmTrainer == 'sparkLDA':
                print_logs(f'-- -- sparkLDA requires preprocessing with spark',
                           do_logger=DO_LOGGER, level="error", logger=self._logger)
                return

            elif tmTrainer == "prodLDA":

                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = trDF[['id', 'cleantext']].rename(
                        columns={"cleantext": "bow_text"})
                    if nw > 0:
                        DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                            'scheduler': 'processes', 'num_workers': nw})
                    else:
                        # Use Dask default number of workers (i.e., number of cores)
                        DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                            'scheduler': 'processes'})

            elif tmTrainer in ["ctm", "bertopic"]:
                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = trDF[['id', 'cleantext', 'embeddings']].rename(
                        columns={"cleantext": "bow_text"})

                    try:
                        schema = pa.schema([
                            ('id', pa.int64()),
                            ('bow_text', pa.string()),
                            ('embeddings', pa.list_(pa.float64()))
                        ])
                        if nw > 0:
                            DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                                'scheduler': 'processes', 'num_workers': nw})
                        else:
                            # Use Dask default number of workers (i.e., number of cores)
                            DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                                'scheduler': 'processes'})
                    except:
                        schema = pa.schema([
                            ('id', pa.int64()),
                            ('bow_text', pa.string()),
                            ('embeddings', pa.string())
                        ])
                        if nw > 0:
                            DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                                'scheduler': 'processes', 'num_workers': nw})
                        else:
                            # Use Dask default number of workers (i.e., number of cores)
                            DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
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
                # malletDF.coalesce(1).write.format("text").option("header", "false").save(f"file://{tempFolder.as_posix()}")
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
                    F.col("bow"))).select("id", "bow_text", "embeddings"))
                lemas_raw_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")

        return outFile


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

    def __init__(self, mallet_path, ntopics=25, alpha=5.0, optimize_interval=10, num_threads=4, num_iterations=1000, doc_topic_thr=0.0, thetas_thr=0.003, token_regexp=None, get_sims=False, open_api_key=None, do_labeller=True, logger=None):
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
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
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
        self._get_sims = get_sims
        self._open_api_key = open_api_key
        self._do_labeller = do_labeller

        if not self._mallet_path.is_file():
            print_logs(f'-- -- Provided mallet path is not valid -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
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
        print_logs('-- -- Sparsifying doc-topics matrix',
                   do_logger=DO_LOGGER, level="debug", logger=self._logger)
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
        print_logs('-- -- Mallet training: Vocabulary file generated',
                   do_logger=DO_LOGGER, level="debug", logger=self._logger)

        tm = TMmodel(TMfolder=modelFolder.parent.joinpath('TMmodel'),
                     get_sims=self._get_sims, open_api_key=self._open_api_key, do_labeller=self._do_labeller)
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

        # Remove doc-topics file. It is no longer needed and takes a lot of space
        thetas_file.unlink()

        return tm

    def _extract_pipe(self, modelFolder):
        """
        Creates a pipe based on a small amount of the training data to ensure that the holdout data that may be later inferred is compatible with the training data

        Parameters
        ----------
        modelFolder: Path
            Path to the model folder
        """

        # Get corpus file
        path_corpus = modelFolder.joinpath('corpus.mallet')
        if not path_corpus.is_file():
            print_logs('-- Pipe extraction: Could not locate corpus file',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            return

        # Create auxiliary file with only first line from the original corpus file
        path_txt = modelFolder.parent.joinpath('corpus.txt')
        with path_txt.open('r', encoding='utf8') as f:
            first_line = f.readline()
        path_aux = modelFolder.joinpath('corpus_aux.txt')
        with path_aux.open('w', encoding='utf8') as fout:
            fout.write(first_line + '\n')

        # We perform the import with the only goal to keep a small file containing the pipe
        print_logs('-- Extracting pipeline', do_logger=DO_LOGGER,
                   level="info", logger=self._logger)
        path_pipe = modelFolder.joinpath('import.pipe')

        cmd = self._mallet_path.as_posix() + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_corpus, path_aux, path_pipe)

        try:
            print_logs(
                f'-- Running command {cmd}', do_logger=DO_LOGGER, level="info", logger=self._logger)
            check_output(args=cmd, shell=True)
        except:
            print_logs('-- Failed to extract pipeline. Revise command',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)

        # Remove auxiliary file
        path_aux.unlink()

        return

    def fit(self, corpusFile):
        """
        Training of Mallet Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """
        
        t_start = time.perf_counter()

        # Output model folder and training file for the corpus
        if not corpusFile.is_file():
            print_logs(f'-- -- Provided corpus Path does not exist -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir(exist_ok=True)

        ##################################################
        # Importing Data to mallet
        print_logs('-- -- Mallet Corpus Generation: Mallet Data Import',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)

        corpusMallet = modelFolder.joinpath('corpus.mallet')

        cmd = self._mallet_path.as_posix() + \
            ' import-file --preserve-case --keep-sequence ' + \
            '--remove-stopwords --token-regex "' + self._token_regexp + \
            '" --input %s --output %s'
        cmd = cmd % (corpusFile, corpusMallet)

        try:
            print_logs(
                f'-- -- Running command {cmd}', do_logger=DO_LOGGER, level="info", logger=self._logger)
            check_output(args=cmd, shell=True)
        except Exception as e:
            print(e)
            print_logs(f'-- -- Mallet failed to import data. Revise command',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)

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
            fout.write('output-state = ' +
                       modelFolder.joinpath('topic-state.gz').resolve().as_posix() + '\n')
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
            print_logs(
                f'-- -- Training mallet topic model. Command is {cmd}', do_logger=DO_LOGGER, level="info", logger=self._logger)
            check_output(args=cmd, shell=True)
        except:
            print_logs('-- -- Model training failed. Revise command',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            return

        t_end = time.perf_counter() - t_start
        
        ##################################################
        # Create TMmodel object

        tm = self._createTMmodel(modelFolder)

        # Create pipe for future inference tasks
        self._extract_pipe(modelFolder)

        return t_end


class sparkLDATrainer(Trainer):
    """
    Wrapper for the Spark LDA Topic Model Training. Implements the
    following functionalities
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    """

    def __init__(self, ntopics=25, alpha=5.0, maxIter=20, optimizer='online',
                 optimizeDocConcentration=True, subsamplingRate=0.05, thetas_thr=0.003, get_sims=False, open_api_key=None, do_labeller=True, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        ntopics: int
            Number of topics for the model
        alpha: float
            Parameter for the Dirichlet prior on doc distribution
        maxIter: int
            Maximum number of iterations
        optimizer: str
            Optimiation algorithm 'online'|'em' (see Spark MLLIB documentation)
        optimizeDocConcentration: bool
            If True, the prior for the documents Dirichlet will be optimized
        subsamplingRate: float
            percentage of documents that will be used for every minibatch
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)

        self._ntopics = ntopics
        self._alpha = alpha
        self._maxIter = maxIter
        self._optimizer = optimizer
        self._optimizeDocConcentration = optimizeDocConcentration
        self._subsamplingRate = subsamplingRate
        self._thetas_thr = thetas_thr
        self._get_sims = get_sims
        self._open_api_key = open_api_key
        self._do_labeller = do_labeller

    def _createTMmodel(self, modelFolder, ldaModel, df):
        """Creates an object of class TMmodel hosting the topic model
        that has been trained using Spark LDA

        Parameters
        ----------
        modelFolder: Path
            the folder with the mallet output files
        ldaModel: Spark LDAModel
            Spark model already fitted with the training data
        df: Spark dataframe
            With training data

        Returns
        -------
        tm: TMmodel
            The topic model as an object of class TMmodel

        """

        # Calculation and Sparsification of thetas matrix
        print_logs('-- -- Spark LDA Sparsifying doc-topics matrix',
                   do_logger=DO_LOGGER, level="debug", logger=self._logger)

        # We calculate the full matrix. We need to do the following
        # * Obtain topic representation of documents
        # * Convert DenseVectors to arrays
        # * Generate a numpy matrix (thetas32)
        # * Threshold the matrix
        # This can be probably implemented in a more efficient manner
        df = (
            ldaModel.transform(df)
            .select("topicDistribution")
            .withColumn('topicDistribution', vector_to_array('topicDistribution'))
        )
        thetas32 = np.array([el[0] for el in df.toPandas().values.tolist()])
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32, modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Topics-vocabulary matrix
        betas = normalize(
            ldaModel.topicsMatrix().toArray().T, axis=1, norm='l1')

        # Load vocabulary file
        with Path(modelFolder.parent.joinpath('vocabulary.txt')).open('r', encoding='utf8') as fin:
            vocab = [el.strip() for el in fin.readlines()]

        tm = TMmodel(modelFolder.parent.joinpath(
            'TMmodel'), get_sims=self._get_sims, open_api_key=self._open_api_key, do_labeller=self._do_labeller)
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

        return tm

    def fit(self, corpusFile):
        """
        Training of Spark LDA Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to parquet file in mallet format
        """

        t_start = time.perf_counter()
        
        # Output model folder and training file for the corpus
        if not corpusFile.exists():
            print_logs(f'-- -- Provided corpus Path does not exist -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            sys.exit()

        # Train spark LDA model and obtain also the topic distribution of
        # documents
        ##################################################
        # Importing Data to mallet
        print_logs('-- -- Training LDA model with Spark',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)

        df = spark.read.parquet(f"file://{corpusFile.resolve().as_posix()}")
        lda = pysparkLDA(featuresCol="bow", maxIter=self._maxIter, k=self._ntopics,
                         optimizer=self._optimizer, subsamplingRate=self._subsamplingRate,
                         optimizeDocConcentration=self._optimizeDocConcentration,
                         docConcentration=self._ntopics*[self._alpha/self._ntopics])
        ldaModel = lda.fit(df)

        # Save model for future use
        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir(exist_ok=True)
        ldaModel.save(
            f"file://{modelFolder.joinpath('sparkLDAmodel').as_posix()}")

        t_end = time.perf_counter() - t_start
        
        ##################################################
        # Create TMmodel object

        tm = self._createTMmodel(modelFolder, ldaModel, df)

        return t_end


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
                 topic_prior_mean=0.0, topic_prior_variance=None, num_samples=10,
                 num_data_loader_workers=0, thetas_thr=0.003, get_sims=False, open_api_key=None, do_labeller=True, logger=None):
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
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
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
        self._get_sims = get_sims
        self._open_api_key = open_api_key
        self._do_labeller = do_labeller
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
        print_logs('-- -- Sparsifying doc-topics matrix',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)
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

        # Create vocabulary list and calculate beta matrix
        betas = avitm.get_topic_word_distribution()
        vocab = self._train_dataset.idx2token

        # Create TMmodel
        tm = TMmodel(modelFolder.parent.joinpath(
            'TMmodel'), get_sims=self._get_sims, open_api_key=self._open_api_key, do_labeller=self._do_labeller)
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

        t_start = time.perf_counter()
        
        # Output model folder and training file for the corpus
        if not os.path.exists(corpusFile):
            print_logs(f'-- -- Provided corpus Path does not exist -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir(exist_ok=True)

        # Generating the corpus in the input format required by ProdLDA
        print_logs('-- -- ProdLDA Corpus Generation: BOW Dataset object',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)
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
        
        t_end = time.perf_counter() - t_start

        # Create TMmodel object
        tm = self._createTMmodel(modelFolder, avitm)

        # Save avitm model for future inference
        model_file = modelFolder.joinpath('model.pickle')
        pickler_avitm_for_ewb_inferencer(
            path_model_infer=model_file,
            avitm_model=avitm)

        return t_end


class CTMTrainer(Trainer):
    """
    Wrapper for the CTM Topic Model Training. Implements the
    following functionalities
    - Transformation of the corpus to the CTM internal training format
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(self,
                 n_components=10,
                 contextual_size=768,
                 ctm_model_type='CombinedTM',
                 model_type='prodLDA',
                 hidden_sizes=(100, 100),
                 activation='softplus',
                 dropout_in=0.2,
                 dropout_out=0.2,
                 learn_priors=True,
                 batch_size=64,
                 lr=2e-3,
                 momentum=0.99,
                 solver='adam',
                 num_epochs=100,
                 num_samples=10,
                 reduce_on_plateau=False,
                 topic_prior_mean=0.0,
                 topic_prior_variance=None,
                 num_data_loader_workers=0,
                 label_size=0,
                 loss_weights=None,
                 thetas_thr=0.003, sbert_model_to_load='paraphrase-distilroberta-base-v1',
                 get_sims=False,
                 max_features=None,
                 open_api_key=None,
                 do_labeller=True,
                 logger=None):
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
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)
        self._n_components = n_components
        self._model_type = model_type
        self._ctm_model_type = ctm_model_type
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._dropout_in = dropout_in
        self._dropout_out = dropout_out
        self._learn_priors = learn_priors
        self._batch_size = batch_size
        self._contextual_size = contextual_size
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
        self._get_sims = get_sims
        self._max_features = max_features
        self._open_api_key = open_api_key
        self._do_labeller = do_labeller

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
        print_logs('-- -- Sparsifying doc-topics matrix',
                   do_logger=DO_LOGGER, level="debug", logger=self._logger)
        # Create figure to check thresholding is correct
        self._SaveThrFig(thetas32, modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Calculate beta matrix and vocab list
        betas = ctm.get_topic_word_distribution()
        vocab = self._qt.vocab

        # TODO: This should be removed in final version
        # Data pyldavis
        # Get LDAVIS data format
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyLDAvis as vis
        data_ldavis = ctm.get_ldavis_data_format(
            self._qt.vocab, self._train_dts, n_samples=2)
        ctm_pd = vis.prepare(**data_ldavis)
        file = modelFolder.joinpath('pyLDAvis.html').as_posix()
        vis.save_html(ctm_pd, file)

        # Create TMmodel
        tm = TMmodel(modelFolder.parent.joinpath(
            'TMmodel'), get_sims=self._get_sims, open_api_key=self._open_api_key, do_labeller=self._do_labeller)
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

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

        t_start = time.perf_counter()
        
        # Output model folder and training file for the corpus
        if not os.path.exists(corpusFile):
            print_logs(f'-- -- Provided corpus Path does not exist -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir(exist_ok=True)

        # Generating the corpus in the input format required by CTM
        print_logs('-- -- CTM Corpus Generation: BOW Dataset object',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)
        df = pd.read_parquet(corpusFile)
        df_lemas = df[["bow_text"]].values.tolist()
        df_lemas = [doc[0].split() for doc in df_lemas]
        self._corpus = [el for el in df_lemas]

        if embeddingsFile is None:
            if not "embeddings" in list(df.columns.values):
                df_raw = df[["all_rawtext"]].values.tolist()
                df_raw = [doc[0].split() for doc in df_raw]
                self._unpreprocessed_corpus = [el for el in df_raw]
                self._embeddings = None
            else:
                self._embeddings = df.embeddings.values
                if isinstance(self._embeddings[0], str):
                    self._embeddings = np.array(
                        [np.array(el.split(), dtype=np.float32) for el in self._embeddings])
                self._unpreprocessed_corpus = None
        else:
            if not embeddingsFile.is_file():
                print_logs(f'-- -- Provided embeddings Path does not exist -- Stop',
                           do_logger=DO_LOGGER, level="error", logger=self._logger)
                sys.exit()
            self._embeddings = np.load(embeddingsFile, allow_pickle=True)
            self._unpreprocessed_corpus = None

        # Generate the corpus in the input format required by CTM
        self._train_dts, self._val_dts, self._input_size, self._id2token, self._qt, self._embeddings_train, _, self._docs_train = \
            prepare_ctm_dataset(corpus=self._corpus,
                                unpreprocessed_corpus=self._unpreprocessed_corpus,
                                custom_embeddings=self._embeddings,
                                sbert_model_to_load=self._sbert_model_to_load,
                                max_features=self._max_features)

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
                contextual_size=self._contextual_size,
                n_components=self._n_components,
                model_type=self._model_type,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                dropout_in=self._dropout_in,
                dropout_out=self._dropout_out,
                learn_priors=self._learn_priors,
                batch_size=self._batch_size,
                lr=self._lr,
                momentum=self._momentum,
                solver=self._solver,
                num_epochs=self._num_epochs,
                reduce_on_plateau=self._reduce_on_plateau,
                num_data_loader_workers=self._num_data_loader_workers)
        else:
            ctm = CombinedTM(
                logger=self._logger,
                input_size=self._input_size,
                contextual_size=self._contextual_size,
                n_components=self._n_components,
                model_type=self._model_type,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                dropout_in=self._dropout_in,
                dropout_out=self._dropout_out,
                learn_priors=self._learn_priors,
                batch_size=self._batch_size,
                lr=self._lr,
                momentum=self._momentum,
                solver=self._solver,
                num_epochs=self._num_epochs,
                reduce_on_plateau=self._reduce_on_plateau,
                num_data_loader_workers=self._num_data_loader_workers,
                label_size=self._label_size,
                loss_weights=self._loss_weights)

        ctm.fit(self._train_dts, self._val_dts)

        # Save ctm model for future inference
        model_file = modelFolder.joinpath('model.pickle')
        pickler(model_file, ctm)
        
        t_end = time.perf_counter() - t_start

        # Create TMmodel object
        tm = self._createTMmodel(modelFolder, ctm)

        return t_end


class BERTopicTrainer(Trainer):
    """
    Wrapper for the BERTopic Topic Model Training. Implements the
    following functionalities
    - Training of the model
    - Creation and persistence of the TMmodel object for tm curation
    - Execution of some other time consuming tasks (pyLDAvis, ..)

    """

    def __init__(
        self,
        stopwords=None,
        no_below: int = 1,
        no_above: float = 1,
        ntopics=10,
        thetas_thr=0.003,
        get_sims=False,
        sbert_model: str = "paraphrase-distilroberta-base-v2",
        umap_n_components=5,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        umap_metric='cosine',
        hdbscan_min_cluster_size=10,
        hdbscan_metric='euclidean',
        hdbscan_cluster_selection_method='eom',
        hbdsan_prediction_data=True,
        open_api_key=None,
        do_labeller=True,
        logger=None
    ):
        """
        Initilization Method

        Parameters
        ----------
        stopwords: list
            List of stopwords to be used for the model
        no_below: int (default=1)
            Ignore all words which appear in less than no_below documents
        no_above: float (default=1)
            Ignore all words which appear in more than no_above documents
        ntopics : int (default=10)
            Number of topics
        thetas_thr: float
            Min value for sparsification of topic proportions after training
        get_sims: boolean
            Flag to detect if similarities are going to be calculated or not.
        sbert_model: str (default='paraphrase-distilroberta-base-v2')
            Model to be used for calculating the embeddings
        umap_n_components: int (default=5)
            Number of components to reduce the embeddings to
        umap_n_neighbors: int (default=15)
            Number of neighbors to consider for UMAP
        umap_min_dist: float (default=0.0)
            Minimum distance between points in the UMAP space
        umap_metric: str (default='cosine')
            Metric to be used for UMAP
        hdbscan_min_cluster_size: int (default=10)
            Minimum number of samples in a cluster
        hdbscan_metric: str (default='euclidean')
            Metric to be used for HDBSCAN
        hdbscan_cluster_selection_method: str (default='eom')
            Method to select the number of clusters
        hbdsan_prediction_data: bool (default=True)
            If True, the prediction data is used for HDBSCAN
        logger: Logger object
            To log object activity
        """

        super().__init__(logger)
        self.ntopics = ntopics
        self._thetas_thr = thetas_thr
        self._get_sims = get_sims
        self._open_api_key = open_api_key 
        self._do_labeller = do_labeller

        # Initialize specific parameters for BERTopic
        self.sbert_model = sbert_model
        self.stopwords = stopwords
        self.no_below = no_below
        self.no_above = no_above
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method
        self.hbdsan_prediction_data = hbdsan_prediction_data

        word_min_len = 2
        self.word_pattern = (
            f"(?<![a-zA-Z\u00C0-\u024F\d\-\_])"
            f"[a-zA-Z\u00C0-\u024F]"
            f"(?:[a-zA-Z\u00C0-\u024F]|(?!\d{{4}})[\d]|[\-\_\·\.'](?![\-\_\·\.'])){{{word_min_len - 1},}}"
            f"(?<![\-\_\·\.'])[a-zA-Z\u00C0-\u024F\d]?"
            f"(?![a-zA-Z\u00C0-\u024F\d])"
        )

        return

    def _createTMmodel(self, modelFolder):
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

        # Sparsification of thetas matrix
        print_logs('-- -- Sparsifying doc-topics matrix',
                   do_logger=DO_LOGGER, level="debug", logger=self._logger)
        # Create figure to check thresholding is correct
        self._SaveThrFig(self._model.thetas,
                         modelFolder.joinpath('thetasDist.pdf'))
        # Set to zeros all thetas below threshold, and renormalize
        thetas32 = self._model.thetas
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Calculate beta matrix and vocab list
        betas = self._model.betas
        vocab = self._model.vocab

        # Create TMmodel
        tm = TMmodel(modelFolder.parent.joinpath(
            'TMmodel'), get_sims=self._get_sims, open_api_key=self._open_api_key, do_labeller=self._do_labeller)
        tm.create(betas=betas, thetas=thetas32, alphas=alphas,
                  vocab=vocab)

        return tm

    def fit(
        self,
        corpusFile
    ):
        """
        Training of BERTopic Topic Model

        Parameters
        ----------
        corpusFile: Path
            Path to txt file in mallet format
            id 0 token1 token2 token3 ...
        """
        
        t_start = time.perf_counter()

        # Output model folder and training file for the corpus
        if not os.path.exists(corpusFile):
            print_logs(f'-- -- Provided corpus Path does not exist -- Stop',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)
            sys.exit()

        modelFolder = corpusFile.parent.joinpath('modelFiles')
        modelFolder.mkdir(exist_ok=True)

        # Generating the corpus in the input format required by BERTopic
        print_logs('-- -- BERTopic Corpus Generation: Reading documents from parquet file as list of strings',
                   do_logger=DO_LOGGER, level="info", logger=self._logger)

        df = pd.read_parquet(corpusFile)
        # If the raw text is available, use it, otherwise use the bow text (the recommended option would be to use the raw text)
        if "raw_text" in df.columns:
            texts = [el[0] for el in df[["raw_text"]].values.tolist()]
        else:
            texts = [el[0] for el in df[["bow_text"]].values.tolist()]
        embeddings = None
        if "embeddings" in df.columns:
            embeddings = df.embeddings.values.tolist()
            if isinstance(embeddings[0], str):
                embeddings = np.array(
                    [np.array(el.split(), dtype=np.float32) for el in embeddings])
            embeddings = np.array(embeddings)

        # Save training corpus
        corpus_file = modelFolder.joinpath('corpus.txt')
        with open(corpus_file, 'w', encoding='utf-8') as fout:
            id = 0
            for el in texts:
                fout.write(str(id) + ' 0 ' + el + '\n')
                id += 1

        # ------------------------------------------------
        # Put components together to create BERTopic model
        # ------------------------------------------------
        ##################################################
        # STEP 0 : Embedding model
        ##################################################
        if embeddings is not None:
            print_logs(f"-- -- Using pre-trained embeddings from the dataset...",
                       do_logger=DO_LOGGER, level="info", logger=self._logger)
            self._embedding_model = None
        else:
            print_logs(
                f"-- -- Creating SentenceTransformer model with {self.sbert_model}...", do_logger=DO_LOGGER, level="info", logger=self._logger)
            self._embedding_model = SentenceTransformer(
                self.sbert_model
            )
        ##################################################
        # STEP 1: Reduce dimensionality of embeddings
        ##################################################
        self._umap_model = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric
        )

        ##################################################
        # STEP 2: Cluster reduced embeddings
        ##################################################
        self._hdbscan_model = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            prediction_data=self.hbdsan_prediction_data
        )

        ##################################################
        # STEP 3: Tokenize topics
        ##################################################
        self._vectorizer_model = CountVectorizer(
            token_pattern=self.word_pattern,
            stop_words=self.stopwords,
            # max_df=self.no_below,
            # min_df=self.no_above,
        )

        ##################################################
        # STEP 4: Create topic representation
        ##################################################
        self._ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        ###################################################
        # STEP 5 (Optional): Add additional representations
        ###################################################
        self._representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(
                diversity=0.3,
                top_n_words=15
            )
        }

        self._model = BERTopic(
            language="multilingual",
            min_topic_size=1,
            nr_topics=self.ntopics,
            low_memory=False,
            # calculate_probabilities=True,
            embedding_model=self._embedding_model,
            # umap_model=KMeans(n_clusters=self.num_topics),#self._umap_model,
            hdbscan_model=self._hdbscan_model,
            vectorizer_model=self._vectorizer_model,
            # ctfidf_model=self._ctfidf_model,
            # representation_model=self._representation_model,
            verbose=True
        )

        # ------------------------------------------------
        # Train model
        # ------------------------------------------------
        # probs = The probability of the assigned topic per document.
        # If `calculate_probabilities` in BERTopic is set to True, then
        # it calculates the probabilities of all topics across all documents
        # instead of only the assigned topic.
        if embeddings is not None:
            _, probs = self._model.fit_transform(texts, embeddings)
        else:
            _, probs = self._model.fit_transform(texts)

        # ------------------------------------------------
        # Get distributions
        # ------------------------------------------------
        ##################################################
        # THETAS
        ##################################################
        # topic_distr = D x K matrix containing the topic distributions
        # for all input documents.
        thetas_approx, _ = self._model.approximate_distribution(texts)
        check_thetas = [(doc_id, thetas_approx[doc_id].shape) for doc_id in range(
            len(thetas_approx)) if thetas_approx[doc_id].shape != (self.ntopics,)]
        if len(check_thetas) > 0:
            print_logs(
                f"-- -- No all the thetas have the same shape: {check_thetas}", do_logger=DO_LOGGER, level="warning", logger=self._logger)
        self._model.thetas = thetas_approx

        ##################################################
        # BETAS
        ##################################################
        self._model.betas = self._model.c_tf_idf_.toarray()
        self._model.vocab = self._model.vectorizer_model.get_feature_names_out()

        ##################################################
        # TOPICS
        ##################################################
        # self._model.topics = dict()
        # for k, v in self._model.get_topics().items():
        #    self._model.topics[k] = [el[0] for el in v]

        # Save BERTopic model for future inference
        model_file = modelFolder.joinpath('model.pickle')
        pickler(model_file, self._model)

        t_end = time.perf_counter() - t_start
        
        # Create TMmodel object
        self._createTMmodel(modelFolder)

        return t_end


class HierarchicalTMManager(object):
    """
    Main class for the creation of hierarchical topic models. Implements the
    following functionalities
    - Generation of the corpus of a second-level submodel based on the chosen hierarchical algorithm and the specified first-level topic model
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

        # Read training configurations from father model and submodel
        configFile_c = Path(configFile_c)
        configFile_f = Path(configFile_f)
        with configFile_f.open('r', encoding='utf8') as fin:
            tr_config_f = json.load(fin)
        with configFile_c.open('r', encoding='utf8') as fin:
            tr_config_c = json.load(fin)

        # Get father model's trainin corpus as dask dataframe
        if tr_config_f['trainer'] == "ctm":
            corpusFile = configFile_f.parent.joinpath('modelFiles/corpus.txt')
        else:
            corpusFile = configFile_f.parent.joinpath('corpus.txt')
        corpus = [line.rsplit(' 0 ')[1].strip() for line in open(
            corpusFile, encoding="utf-8").readlines()]
        tr_data_df = pd.DataFrame(data=corpus, columns=['doc'])
        tr_data_df['id'] = range(1, len(tr_data_df) + 1)

        w_assignFile = configFile_f.parent.joinpath('w_assign.txt')
        if tr_config_c['htm-version'] == "htm-ws" and w_assignFile.is_file():
            w_assign = [line.strip() for line in open(
                w_assignFile, encoding="utf-8").readlines()]
            tr_data_df['w_assign'] = w_assign

        tr_data_ddf = dd.from_pandas(tr_data_df, npartitions=2)

        # Get embeddings if the trainer is CTM
        if tr_config_f['trainer'] == "ctm":
            embeddingsFile = configFile_f.parent.joinpath(
                'modelFiles/embeddings.npy')
            embeddings = np.load(embeddingsFile, allow_pickle=True)

        # Load father's TMmodel and get from it the information necessary for the submodel's corpus creation
        tmmodel = TMmodel(TMmodel_path)
        betas, thetas, vocab_w2id, vocab_id2w = \
            tmmodel.get_model_info_for_hierarchical()
        thetas = thetas.toarray()

        # Get expansion topic
        exp_tpc = int(tr_config_c['expansion_tpc'])

        if tr_config_c['htm-version'] == "htm-ws":
            print_logs('-- -- -- Creating training corpus according to HTM-WS.',
                       do_logger=DO_LOGGER, level="info", logger=self._logger)

            def get_htm_ws_corpus_base(row, thetas, betas, vocab_id2w, vocab_w2id, exp_tpc):
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
                assignments = []
                for idx_w in words_doc_idx:
                    p_z = np.multiply(thetas_d, betas[:, idx_w])
                    p_z_args = np.argsort(p_z)
                    if p_z[p_z_args[-1]] > 20*p_z[p_z_args[-2]]:
                        assignments.append(p_z_args[-1])
                    else:
                        sampling = np.random.multinomial(1, np.multiply(
                            thetas_d, betas[:, idx_w])/np.sum(np.multiply(thetas_d, betas[:, idx_w])))
                        assignments.append(int(np.nonzero(sampling)[0][0]))

                assignments_str = ' '.join([str(el) for el in assignments])

                return assignments_str

            def get_htm_ws_corpus_from_zs(row, thetas, betas, vocab_id2w, vocab_w2id, exp_tpc):

                doc = row["doc"].split()
                w_assign = row["w_assign"].split()

                reduced_doc = [el[0] for el in zip(
                    doc, w_assign) if el[1] == str(exp_tpc)]

                reduced_doc_str = ' '.join([el for el in reduced_doc])

                return reduced_doc_str

            if tr_config_c['trainer'] == "ctm":

                if not w_assignFile.is_file():
                    print("Generating assignments file...")
                    tr_data_ddf['w_assign'] = tr_data_ddf.apply(
                        get_htm_ws_corpus_base, axis=1, meta=('x', 'object'), args=(thetas, betas, vocab_id2w, vocab_w2id, exp_tpc))

                    with ProgressBar():
                        DFmallet = tr_data_ddf[['w_assign']]
                        DFmallet.to_csv(
                            w_assignFile, index=False,
                            header=False, single_file=True,
                            compute_kwargs={'scheduler': 'processes'})
                    print("Saved assignments file")

                tr_data_ddf['reduced_doc'] = tr_data_ddf.apply(
                    get_htm_ws_corpus_from_zs, axis=1, meta=('x', 'object'), args=(thetas, betas, vocab_id2w, vocab_w2id, exp_tpc))

            elif tr_config_c['trainer'] == "mallet":
                topic_state_model = configFile_f.parent.joinpath(
                    'modelFiles/topic-state.gz').as_posix()

                # 0 = document's id
                # 1 = document's name
                # 3
                # 4 = word
                # 5 = topic to which the word belongs

                with gzip.open(topic_state_model) as fin:
                    topic_state_df = pd.read_csv(fin, delim_whitespace=True,
                                                 names=['docid', 'NA1', 'NA2',
                                                        'NA3', 'word', 'tpc'],
                                                 header=None, skiprows=3)

                # Read the dictionary from the JSON file
                with open(configFile_f.parent.joinpath(
                        'TMmodel/corr.json').as_posix(), 'r') as json_file:
                    corr = json.load(json_file)
                corr = {np.int64(key): value for key, value in corr.items()}

                # Map Mallet topics to the new ID given by the TMmodel ordering
                topic_state_df['tpc'] = topic_state_df['tpc'].map(corr)

                topic_state_df.word.replace('nan', np.nan, inplace=True)
                topic_state_df.fillna('nan_value', inplace=True)

                topic_state_df_tpc = topic_state_df[topic_state_df['tpc'] == exp_tpc]
                topic_to_corpus = topic_state_df_tpc.groupby(
                    'docid')['word'].apply(list).reset_index(name='new')

            if tr_config_c['trainer'] == "mallet":

                outFile = configFile_c.parent.joinpath('corpus.txt')
                if outFile.is_file():
                    outFile.unlink()

                with open(outFile, 'w', encoding='utf-8') as fout:
                    for el in topic_to_corpus.values.tolist():
                        fout.write(str(el[0]) + ' 0 ' + ' '.join(el[1]) + '\n')

            elif tr_config_c['trainer'] == "ctm":
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
            print_logs('-- -- -- Creating training corpus according to HTM-DS.',
                       do_logger=DO_LOGGER, level="info", logger=self._logger)

            # Get ids of documents that meet the condition of having a representation of the expansion topic larger than thr
            thr = float(tr_config_c['thr'])
            doc_ids_to_keep = \
                [idx for idx in range(thetas.shape[0])
                 if thetas[idx, exp_tpc] > thr]

            # Keep selected documents from the father's corpus
            tr_data_ddf = tr_data_ddf.loc[doc_ids_to_keep]

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

            elif tr_config_c['trainer'] == "ctm":

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
                embeddings = embeddings[doc_ids_to_keep]

        else:
            print_logs('-- -- -- The specified HTM version is not available.',
                       do_logger=DO_LOGGER, level="error", logger=self._logger)

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
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Preprocess training data according to config file")
    parser.add_argument('--nw', type=int, required=False, default=0,
                        help="Number of workers when preprocessing data with Dask. Use 0 to use Dask default")
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train Topic Model according to config file")
    parser.add_argument('--hierarchical', action='store_true', default=False,
                        help='Create submodel training data according to config files', required=False)
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")
    parser.add_argument('--config_child', type=str, default=None,
                        help="Path to submodel's config file", required=False)
    parser.add_argument('--do_logger', type=bool, default=True, required=False)
    parser.add_argument('--open_ai_key', type=str, default=None, required=False)
    parser.add_argument('--open_ai_key_path', type=str, required=False,
                        default='/export/usuarios_ml4ds/lbartolome/Repos/my_repos/miniature-fishstick/.env')
    parser.add_argument('--do_labeller', type=bool, default=True,
                        help="Flag to active the labelling of the topics using OpenAI.", required=False)

    args = parser.parse_args()


    args = parser.parse_args()

    DO_LOGGER = args.do_logger

    if args.spark:
        # Spark imports and session generation
        import pyspark.sql.functions as F
        from pyspark.ml.clustering import LDA as pysparkLDA
        from pyspark.ml.feature import (CountVectorizer, StopWordsRemover,
                                        Tokenizer)
        from pyspark.ml.functions import vector_to_array
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
                          .withColumn("source", F.lit(DtSet["source"]))
                          .select("id", "source", "all_lemmas")
                    )
                    if idx == 0:
                        trDF = df
                    else:
                        trDF = trDF.union(df).distinct()

                # We preprocess the data and save the CountVectorizer Model used to obtain the BoW
                trDF = tPreproc.preprocBOW(trDF)
                tPreproc.saveCntVecModel(configFile.parent.resolve())

                # If the trainer is CTM, we also need the embeddings
                if train_config['trainer'] in ["ctm", "bertopic"]:
                    # We get full df containing the embeddings
                    for idx, DtSet in enumerate(trDtSet['Dtsets']):
                        df = spark.read.parquet(f"file://{DtSet['parquet']}")
                        
                        # If the trainer is BERTopic, we would need the raw texts
                        # To avoid an error here in case the raw texts are not available, we just try. If it fails, we continue
                        if train_config["trainer"] == "bertopic":
                            try:
                                df = df.select("id", "embeddings", "raw_text")
                            except:
                                print_logs("-- -- -- -- Raw texts not available in the dataset", do_logger=False)
                        else:
                            df = df.select("id", "embeddings")
                        if idx == 0:
                            eDF = df
                        else:
                            eDF = eDF.union(df).distinct()
                            
                    # We perform a left join to keep the embeddings of only those documents kept after preprocessing
                    trDF = (trDF.join(eDF, trDF.id == eDF.id, "left")
                            .drop(df.id))
                    
                # For sparkLDA, we need also a corpus.txt file only for coherence calculation
                if train_config['trainer'] == 'sparkLDA':
                    tPreproc.exportTrData(trDF=trDF,
                                          dirpath=configFile.parent.resolve(),
                                          tmTrainer='mallet')

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
                        df = df.rename(columns={DtSet['idfld']: 'id'})
                        if idx2 == 0:
                            df["all_lemmas"] = df[col]
                        else:
                            df["all_lemmas"] += " " + df[col]
                    df["source"] = DtSet["source"]
                    df = df[["id", "source", "all_lemmas"]]

                    # Concatenate dataframes
                    if idx == 0:
                        trDF = df
                    else:
                        trDF = dd.concat([trDF, df])

                # trDF = trDF.drop_duplicates(subset=["id"], ignore_index=True)
                # We preprocess the data and save the Gensim Model used to obtain the BoW
                trDF = tPreproc.preprocBOW(trDF, nw=args.nw)
                tPreproc.saveGensimDict(configFile.parent.resolve())

                # If the trainer is CTM, we also need the embeddings
                if train_config['trainer'] in ["ctm", "bertopic"]:
                    # We get full df containing the embeddings
                    for idx, DtSet in enumerate(trDtSet['Dtsets']):
                        df = dd.read_parquet(DtSet['parquet']).fillna("")

                        # Rename id column to "id"
                        df = df.rename(columns={DtSet['idfld']: 'id'})
                        
                        # If the trainer is BERTopic, we would need the raw texts
                        # To avoid an error here in case the raw texts are not available, we just try. If it fails, we continue
                        if train_config["trainer"] == "bertopic":
                            try:
                                df = df[["id", "embeddings", "raw_text"]]
                            except:
                                print_logs("-- -- -- -- Raw texts not available in the dataset", do_logger=False)
                        else:
                            df = df[["id", "embeddings"]]

                        # Concatenate dataframes
                        if idx == 0:
                            eDF = df
                        else:
                            eDF = dd.concat([trDF, df])

                    # We perform a left join to keep the embeddings of only those documents kept after preprocessing
                    trDF = trDF.merge(eDF, how="left", on=["id"])

                trDataFile = tPreproc.exportTrData(trDF=trDF,
                                                   dirpath=configFile.parent.resolve(),
                                                   tmTrainer=train_config['trainer'],
                                                   nw=args.nw)
                sys.stdout.write(trDataFile.as_posix())

        else:
            sys.exit('You need to provide a valid configuration file')

    # If the training flag is activated, we need to check availability of
    # configuration file, and run the topic model training
    if args.train:
        
        do_labeller = False
        if args.do_labeller:
            # Loading OpenAI API Key if given
            if args.open_ai_key:
                openai_api_key = args.open_ai_key
                do_labeller = True
                sys.stdout.write("#"*20)
                sys.stdout.write("Loading OPENAI_API_KEY passed as argument...")
                sys.stdout.write("#"*20)
            elif args.open_ai_key_path:
                try:
                    load_dotenv(args.open_ai_key_path)
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    do_labeller = True
                    sys.stdout.write("#"*20)
                    sys.stdout.write("Loading OPENAI_API_KEY from .env file...")
                    sys.stdout.write("#"*20)
                except Exception as e:
                    sys.stdout.write("#"*20)
                    sys.stdout.write("OPENAI_API_KEY could not be set from given .env file. You need to specify a valid OPENAI_API_KEY to use the TopicLabeller functinality. The model will be trained, but no labels will be generated...")
                    sys.stdout.write("#"*20)
                    sys.stdout.write(f"This is the error that occurred:{e}")
            else:
                try:
                    openai_api_key = os.environ["OPENAI_API_KEY"]
                except Exception as e:
                    sys.stdout.write("#"*20)
                    sys.stdout.write("You need to specify an OPENAI_API_KEY to use the TopicLabeller functinality. The model will be trained, but no labels will be generated....")
                    sys.stdout.write("#"*20)
                    sys.stdout.write(f"This is the error that occurred:{e}")
        
        sys.stdout.write("#"*20)
        sys.stdout.write(f"Doing labelling: {do_labeller}")
        sys.stdout.write("#"*20)
        
        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                train_config = json.load(fin)

                get_sims = train_config['TMparam']["get_sims"] if "get_sims" in train_config['TMparam'].keys(
                ) else False

                if train_config['trainer'] == 'mallet':
                    # Create a MalletTrainer object with the parameters specified in the configuration file
                    MallTr = MalletTrainer(
                        mallet_path=train_config['TMparam']['mallet_path'],
                        ntopics=train_config['TMparam']['ntopics'],
                        alpha=train_config['TMparam']['alpha'],
                        optimize_interval=train_config['TMparam']['optimize_interval'],
                        num_threads=train_config['TMparam']['num_threads'],
                        num_iterations=train_config['TMparam']['num_iterations'],
                        doc_topic_thr=train_config['TMparam']['doc_topic_thr'],
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        token_regexp=train_config['TMparam']['token_regexp'],
                        get_sims=get_sims,
                        open_api_key=openai_api_key,
                        do_labeller=do_labeller)

                    # Train the Mallet topic model with the specified corpus
                    MallTr.fit(
                        corpusFile=configFile.parent.joinpath('corpus.txt'))

                elif train_config['trainer'] == 'sparkLDA':
                    if not args.spark:
                        sys.stodout.write(
                            "You need access to a spark cluster to run sparkLDA")
                        sys.exit(
                            "You need access to a spark cluster to run sparkLDA")

                    sparkLDATr = sparkLDATrainer(
                        ntopics=train_config['TMparam']['ntopics'],
                        alpha=train_config['TMparam']['alpha'],
                        maxIter=train_config['TMparam']['maxIter'],
                        optimizer=train_config['TMparam']['optimizer'],
                        optimizeDocConcentration=train_config['TMparam']['optimizeDocConcentration'],
                        subsamplingRate=train_config['TMparam']['subsamplingRate'],
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        get_sims=get_sims,
                        open_api_key=openai_api_key,
                        do_labeller=do_labeller)

                    sparkLDATr.fit(
                        corpusFile=configFile.parent.joinpath('corpus.parquet'))

                elif train_config['trainer'] == 'prodLDA':
                    # Create a ProdLDATrainer object with the parameters specified in the configuration file
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
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        get_sims=get_sims,
                        open_api_key=openai_api_key,
                        do_labeller=do_labeller)

                    # Train the ProdLDA topic model with the specified corpus
                    ProdLDATr.fit(
                        corpusFile=configFile.parent.joinpath('corpus.parquet'))

                elif train_config['trainer'] == 'ctm':
                    # Create a CTMTrainer object with the parameters specified in the configuration file
                    if 'dropout_in' in train_config['TMparam'].keys() and 'dropout_out' in train_config['TMparam'].keys():
                        dropout_in_ = train_config['TMparam']['dropout_in']
                        dropout_out_ = train_config['TMparam']['dropout_out']
                    elif 'dropout' in train_config['TMparam'].keys():
                        dropout_in_ = dropout_out_ = train_config['TMparam']['dropout']
                    else:
                        dropout_in_ = dropout_out_ = 0.2
                    max_features = int(train_config['TMparam']['max_features']) if 'max_features' in train_config['TMparam'].keys(
                    ) else None
                    CTMr = CTMTrainer(
                        n_components=train_config['TMparam']['ntopics'],
                        contextual_size=train_config['TMparam']['contextual_size'],
                        model_type=train_config['TMparam']['model_type'],
                        hidden_sizes=tuple(
                            train_config['TMparam']['hidden_sizes']),
                        activation=train_config['TMparam']['activation'],
                        dropout_in=dropout_in_,
                        dropout_out=dropout_out_,
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
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        get_sims=get_sims,
                        max_features=max_features,
                        open_api_key=openai_api_key,
                        do_labeller=do_labeller)

                    # Train the CTM topic model with the specified corpus
                    corpusFile = configFile.parent.joinpath('corpus.parquet')
                    if not corpusFile.is_dir() and not corpusFile.is_file():
                        sys.exit(
                            "The corpus file 'corpus.parquet' does not exist.")
                    else:
                        if train_config['hierarchy-level'] == 0:
                            CTMr.fit(corpusFile=corpusFile)
                        elif train_config['hierarchy-level'] == 1:
                            embbeddingsFile = configFile.parent.joinpath(
                                'embeddings.npy')
                            if not embbeddingsFile.is_file():
                                sys.exit(
                                    "The embeddings file 'embeddings.npy' does not exist.")
                            else:
                                CTMr.fit(corpusFile=corpusFile,
                                         embeddingsFile=embbeddingsFile)

                elif train_config['trainer'] == 'bertopic':
                    # Create a BERTopicTrainer object with the parameters specified in the configuration file
                    BERTopicTr = BERTopicTrainer(
                        # stopwords=train_config['TMparam']['stopwords'],
                        no_below=train_config['TMparam']['no_below'],
                        no_above=train_config['TMparam']['no_above'],
                        ntopics=train_config['TMparam']['ntopics'],
                        thetas_thr=train_config['TMparam']['thetas_thr'],
                        sbert_model=train_config['TMparam']['sbert_model'],
                        umap_n_components=train_config['TMparam']['umap_n_components'],
                        umap_n_neighbors=train_config['TMparam']['umap_n_neighbors'],
                        umap_min_dist=train_config['TMparam']['umap_min_dist'],
                        umap_metric=train_config['TMparam']['umap_metric'],
                        hdbscan_min_cluster_size=train_config['TMparam']['hdbscan_min_cluster_size'],
                        hdbscan_metric=train_config['TMparam']['hdbscan_metric'],
                        hdbscan_cluster_selection_method=train_config[
                            'TMparam']['hdbscan_cluster_selection_method'],
                        hbdsan_prediction_data=train_config['TMparam']['hbdsan_prediction_data'],
                        get_sims=get_sims,
                        open_api_key=openai_api_key,
                        do_labeller=do_labeller)

                    # Train the Mallet topic model with the specified corpus
                    corpusFile = configFile.parent.joinpath('corpus.parquet')
                    if not corpusFile.is_dir() and not corpusFile.is_file():
                        sys.exit(
                            "The corpus file 'corpus.parquet' does not exist.")
                    BERTopicTr.fit(corpusFile=corpusFile)

        else:
            sys.exit('You need to provide a valid configuration file')

    if args.hierarchical:
        if not args.config_child:
            sys.exit('You need to provide a configuration file for the submodel')
        else:
            configFile_f = Path(args.config)
            if not configFile_f.is_file():
                sys.exit(
                    'You need to provide a valid configuration file for the father model.')
            else:
                configFile_c = Path(args.config_child)
                if not configFile_c.is_file():
                    sys.exit(
                        'You need to provide a valid configuration file for the submodel.')
                else:
                    tMmodel_path = configFile_f.parent.joinpath('TMmodel')
                    if not os.path.isdir(tMmodel_path):
                        sys.exit(
                            'There must exist a valid TMmodel folder for the parent corpus')

                    # Create hierarhicalTMManager object
                    hierarchicalTMManager = HierarchicalTMManager()

                    # Create corpus
                    hierarchicalTMManager.create_submodel_tr_corpus(
                        tMmodel_path, args.config, args.config_child)
