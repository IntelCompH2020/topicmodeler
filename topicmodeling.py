"""
*** IntelComp H2020 project ***
*** Topic Modeling Toolbox  ***

Provides several classes for Topic Modeling
    - stwEQcleaner: For string cleaning (stopword removal + equivalent terms)
    - TMmodel: To represent a trained topic model + edition functions
    - MalletTrainer: To train a topic model from a given corpus
"""

import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse
#from scipy.spatial.distance import jensenshannon
#import pyLDAvis
import matplotlib.pyplot as plt
import argparse
import configparser
import logging
from pathlib import Path
import regex as javare
import re
import sys
import pandas as pd
from tqdm import tqdm
import ipdb
from gensim import corpora
from gensim.utils import check_output, tokenize
logging.getLogger("gensim").setLevel(logging.WARNING)


def file_lines(fname):
    #Count number of lines in file
    with fname.open('r',encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


class stwEQcleaner (object):

    """Simpler version of the english lemmatizer
    It only provides stopword removal and application of equivalences
    ====================================================
    Public methods:
    - cleanstr: Apply stopwords and equivalences on provided string
    =====================================================
    """

    def __init__(self, stw_files=[], dict_eq_file='', logger=None):
        """
        Initilization Method
        Stopwwords and the dictionary of equivalences will be loaded
        during initialization
        :stw_file: List of files of stopwords
        :dict_eq_file: Dictionary of equivalent words A : B means A will be replaced by B

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
            self.__unigramdictio, self.__pattern_unigrams = self.__loadEQFile(dict_eq_file)
            if len(self.__unigramdictio):
                self.__useunigrams = True
        else:
            self.logger.info('-- -- Equivalence file not found')

        return

    def cleanstr(self, rawtext):
        """Function to remove stopwords and apply equivalences
        :param rawtext: string with the text to lemmatize
        """
        if rawtext==None or rawtext=='':
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
        :param file: The file to read the stopwords from
        """
        with open(file, encoding='utf-8') as f:
            stopw = f.read().splitlines()

        return [word.strip() for word in stopw if word]

    def __loadEQFile(self, file):
        """Function to load equivalences from a file. The equivalence file
        will contain an equivalence per line in the format original : target
        where original will be changed to target after lemmatization
        :param file: The file to read the equivalences from
        """
        unigrams = []
        with open(file, 'r', encoding='utf-8') as f:
            unigramlines = f.readlines()
        unigramlines = [el.strip() for el in unigramlines]
        unigramlines = [x.split(' : ') for x in unigramlines]
        unigramlines = [x for x in unigramlines if len(x) == 2]

        if len(unigramlines):
            #This dictionary contains the necessary replacements to carry out
            unigramdictio = dict(unigramlines)
            unigrams = [x[0] for x in unigramlines]
            #Regular expression to find the tokens that need to be replaced
            pattern_unigrams = re.compile(r'\b(' + '|'.join(unigrams) + r')\b')
            return unigramdictio, pattern_unigrams
        else:
            return None, None

    def __removeSTW(self, tokens):
        """Removes stopwords from the provided list
        :param tokens: Input list of string to be cleaned from stw
        """
        return [el for el in tokens if el not in self.__stopwords]


class MalletTrainer(object):
    
    def __init__(self, cf, modelFolder):
        """Object initializer
        Initializes relevant variables from config file
        """

        logging.basicConfig(level='INFO')
        self.logger = logging.getLogger('MalletTrainer')
        
        #Settings for text preprocessing
        self._min_lemas = int(cf['Preproc']['min_lemas'])
        self._no_below = int(cf['Preproc']['no_below'])
        self._no_above = float(cf['Preproc']['no_above'])
        self._keep_n = int(cf['Preproc']['keep_n'])
        #Append stopwords and equivalences files only if they exist
        #Several stopword files can be used, but only one with equivalent terms
        self._stw_file = []
        for f in cf['Preproc']['stw_file'].split(','):
            if not Path(f).is_file():
                self.logger.warning(f'-- -- Stopword file {f} does not exist -- Ignored')
            else:
                self._stw_file.append(Path(f))
        f = cf['Preproc']['eq_file']
        if not Path(f).is_file():
            self.logger.warning(f'-- -- Equivalence file {f} does not exist -- Ignored')
        else:
            self._eq_file = Path(f)
        #Initialize string cleaner
        self._stwEQ = stwEQcleaner(stw_files=self._stw_file, dict_eq_file=self._eq_file,
                             logger=self.logger)

        #Settings for Mallet training
        self._token_regexp_str = cf['Training']['token_regexp']
        self._token_regexp = javare.compile(cf['Training']['token_regexp'])
        self._mallet_path = Path(cf['Training']['mallet_path'])
        if not self._mallet_path.is_file():
            self.logger.error(f'-- -- Provided mallet path is not valid -- Stop')
            sys.exit()
        self._numTopics = int(cf['Training']['ntopics'])
        self._alpha = float(cf['Training']['alpha'])
        self._optimizeInterval = int(cf['Training']['optimize_interval'])
        self._numThreads = int(cf['Training']['num_threads'])
        self._numIterations = int(cf['Training']['num_iterations'])
        self._docTopicsThreshold = float(cf['Training']['doc_topic_thr'])
        self._sparse_thr = float(cf['Training']['thetas_thr'])

        #Output model folder and training files for the corpus
        self._modelFolder = modelFolder
        if not self._modelFolder.is_dir():
            self.logger.error(f'-- -- Provided model folder is not valid -- Stop')
            sys.exit()
        self._corpusFiles = []
        for f in cf['Training']['training_files'].split(','):
            if not Path(f).is_file():
                self.logger.warn(f'-- -- Corpus file {f} does not exist -- Ignored')
            else:
                self._corpusFiles.append(Path(f))

        self.logger.info(f'-- -- Initialization of MalletTrainer variables completed')

        return

    def fit(self):
        """To fit the model we need to preprocess training data, and then
        carry out the training itself"""
        self._preproc()
        self._train()
        return

    def _SaveThrFig(self, thetas32):
        """
        Creates a figure to illustrate the effect of thresholding
        The distribution of thetas is plotted, together with the value
        that the trainer is programmed to use for the thresholding
        """
        allvalues = np.sort(thetas32.flatten())
        step = int(np.round(len(allvalues)/1000))
        plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
        plt.semilogx([self._sparse_thr, self._sparse_thr], [0,100], 'r')
        plot_file = self._modelFolder.joinpath('thetas_dist.pdf')
        plt.savefig(plot_file)
        plt.close()

    def _preproc(self):
        """Preprocessing of files
        For the training we have access (in self._corpusFiles) to a number of lemmatized
        documents. This function:
        1) Carries out a first set of cleaning and homogeneization tasks
        2) Allow to reduce the size of the vocabulary (removing very rare or common terms)
        3) Import the training corpus into Mallet format
        """

        #Identification of words that are too rare or common that need to be 
        #removed from the dictionary. 
        self.logger.info('-- -- Mallet Corpus Generation: Vocabulary generation')
        dictionary = corpora.Dictionary()

        #We iterate over all CSV files
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
            #Keep only relevant fields
            id_lemas = df[["id", "lemmas"]].values.tolist()
            #Apply regular expression to identify tokens
            id_lemas = [[el[0], ' '.join(self._token_regexp.findall(el[1].replace('\n',' ').strip()))]
                                for el in id_lemas]
            #Apply stopwords and equivalence files
            id_lemas = [[el[0], self._stwEQ.cleanstr(el[1])] for el in id_lemas]
            #Apply gensim tokenizer
            id_lemas = [[el[0], list(tokenize(el[1], lowercase=True, deacc=True))] for el in id_lemas]
            #Retain only documents with minimum extension
            id_lemas = [el for el in id_lemas if len(el[1])>=self._min_lemas]
            #Add to dictionary
            all_lemas = [el[1] for el in id_lemas]
            dictionary.add_documents(all_lemas)

        #Remove words that appear in less than no_below documents, or in more than
        #no_above, and keep at most keep_n most frequent terms, keep track of removed
        #words for debugging purposes
        all_words = [dictionary[idx] for idx in range(len(dictionary))]
        dictionary.filter_extremes(no_below=self._no_below, no_above=self._no_above, keep_n=self._keep_n)
        kept_words = set([dictionary[idx] for idx in range(len(dictionary))])
        rmv_words = [el for el in all_words if el not in kept_words]
        #Save extreme words that will be removed
        self.logger.info(f'-- -- Saving {len(rmv_words)} extreme words to file')
        rmv_file = self._modelFolder.joinpath('commonrare_words.txt')
        with rmv_file.open('w', encoding='utf-8') as fout:
            [fout.write(el+'\n') for el in sorted(rmv_words)]
        #Save dictionary to file
        self.logger.info(f'-- -- Saving dictionary to file. Number of words: {len(kept_words)}')
        vocab_txt = self._modelFolder.joinpath('vocabulary.txt')
        with vocab_txt.open('w', encoding='utf-8') as fout:
            [fout.write(el+'\n') for el in sorted(kept_words)]
        #Save also in gensim text format
        vocab_gensim = self._modelFolder.joinpath('vocabulary.gensim')
        dictionary.save_as_text(vocab_gensim)

        ##################################################
        #Create corpus txt files
        self.logger.info('-- -- Mallet Corpus generation: TXT file')
        
        corpus_file = self._modelFolder.joinpath('training_data.txt')

        fcorpus = corpus_file.open('w', encoding='utf-8')
        
        print('Processing files for training dataset creation')
        pbar = tqdm(self._corpusFiles)
        for csvFile in pbar:
            #Document preprocessing is same as before, but now we apply an additional filter
            #and keep only words in the vocabulary
            df = pd.read_csv(csvFile, escapechar="\\", on_bad_lines="skip")
            id_lemas = df[["id", "lemmas"]].values.tolist()
            id_lemas = [[el[0], ' '.join(self._token_regexp.findall(el[1].replace('\n',' ').strip()))]
                                for el in id_lemas]
            id_lemas = [[el[0], self._stwEQ.cleanstr(el[1])] for el in id_lemas]
            id_lemas = [[el[0], list(tokenize(el[1], lowercase=True, deacc=True))] for el in id_lemas]
            id_lemas = [[el[0], [tk for tk in el[1] if tk in kept_words]] for el in id_lemas]
            id_lemas = [el for el in id_lemas if len(el[1])>=self._min_lemas]
            #Write to corpus file
            [fcorpus.write(el[0] + ' 0 ' + ' '.join(el[1]) + '\n') for el in id_lemas]
            
        fcorpus.close()
        
        ##################################################
        #Importing Data to mallet
        self.logger.info('-- -- Mallet Corpus Generation: Mallet Data Import')

        corpus_mallet = self._modelFolder.joinpath('training_data.mallet')

        cmd = self._mallet_path.as_posix() + \
              ' import-file --preserve-case --keep-sequence ' + \
              '--remove-stopwords --token-regex "' + self._token_regexp_str + \
              '" --input %s --output %s'
        cmd = cmd % (corpus_file, corpus_mallet)

        try:
            self.logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Mallet failed to import data. Revise command')

        return

    def _train(self):
        """Mallet training. It does the following:
        1) Trains a Mallet model using the settings provided by the user
        2) It sparsifies thetas matrix and save a figure to report the effect
        3) It saves model matrices: alphas, betas, thetas (sparse)
        """
        config_file = self._modelFolder.joinpath('mallet.config')
        corpus_mallet = self._modelFolder.joinpath('training_data.mallet')

        with config_file.open('w', encoding='utf8') as fout:
            fout.write('input = ' + corpus_mallet.as_posix() + '\n')
            fout.write('num-topics = ' + str(self._numTopics) + '\n')
            fout.write('alpha = ' + str(self._alpha) + '\n')
            fout.write('optimize-interval = ' + str(self._optimizeInterval) + '\n')
            fout.write('num-threads = ' + str(self._numThreads) + '\n')
            fout.write('num-iterations = ' + str(self._numIterations) + '\n')
            fout.write('doc-topics-threshold = ' + str(self._docTopicsThreshold) + '\n')
            #fout.write('output-state = ' + os.path.join(self._outputFolder, 'topic-state.gz') + '\n')
            fout.write('output-doc-topics = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('doc-topics.txt').as_posix() + '\n')
            fout.write('word-topic-counts-file = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('word-topic-counts.txt').as_posix() + '\n')
            fout.write('diagnostics-file = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('diagnostics.xml ').as_posix() + '\n')
            fout.write('xml-topic-report = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('topic-report.xml').as_posix() + '\n')
            fout.write('output-topic-keys = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('topickeys.txt').as_posix() + '\n')
            fout.write('inferencer-filename = ' + \
                self._modelFolder.joinpath('mallet_output').joinpath('inferencer.mallet').as_posix() + '\n')
            #fout.write('output-model = ' + \
            #    self._outputFolder.joinpath('mallet_output').joinpath('modelo.bin').as_posix() + '\n')
            #fout.write('topic-word-weights-file = ' + \
            #    self._outputFolder.joinpath('mallet_output').joinpath('topic-word-weights.txt').as_posix() + '\n')

        cmd = str(self._mallet_path) + ' train-topics --config ' + str(config_file)

        try:
            self.logger.info(f'-- -- Training mallet topic model. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Model training failed. Revise command')
            return

        thetas_file = self._modelFolder.joinpath('mallet_output').joinpath('doc-topics.txt')
        
        cols = [k for k in np.arange(2,self._numTopics+2)]

        #Sparsification of thetas matrix
        self.logger.debug('-- -- Sparsifying doc-topics matrix')
        thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32, usecols=cols)
        #thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32)[:,2:]
        #Create figure to check thresholding is correct
        self._SaveThrFig(thetas32)
        #Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32<self._sparse_thr] = 0
        thetas32 = normalize(thetas32,axis=1,norm='l1')
        thetas32 = sparse.csr_matrix(thetas32, copy=True)

        #Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32,axis=0)).ravel()

        #Create vocabulary files and calculate beta matrix
        #A vocabulary is available with words in alphabetic order,
        #but the new files will use the order used by mallet
        wtcFile = self._modelFolder.joinpath('mallet_output').joinpath('word-topic-counts.txt')
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self._numTopics,vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))

        with wtcFile.open('r', encoding='utf8') as fin:
            for i,line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc,i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas,axis=1,norm='l1')
        #save vocabulary and frequencies
        with self._modelFolder.joinpath('vocab_freq_mallet.txt').open('w', encoding='utf8') as fout:
            [fout.write(el[0]+'\t'+str(int(el[1]))+'\n') for el in zip(vocab,term_freq)]
        self.logger.debug('-- -- Mallet training: Vocabulary file generated')

        #We end by saving the model for future use
        modelVarsDir = self._modelFolder.joinpath('model_vars')
        modelVarsDir.mkdir()
        np.save(modelVarsDir.joinpath('alpha_orig.npy'), alphas)
        np.save(modelVarsDir.joinpath('beta_orig.npy'), betas)
        np.savez(modelVarsDir.joinpath('thetas_orig.npz'),
            thetas_data=thetas32.data, thetas_indices=thetas32.indices,
            thetas_indptr=thetas32.indptr, thetas_shape=thetas32.shape)

        #Remove doc-topics file. It is no longer needed and takes a lot of space
        thetas_file.unlink()

        return


if __name__ == "__main__":

    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False,
                        help="Train a Topic Model according to config file")
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")
    args = parser.parse_args()
    
    #If the training flag is activated, we need to check availability of
    #configuration file, and run the training using class MalletTrainer
    if args.train:
        configFile = Path(args.config)
        if configFile.is_file():
            cf = configparser.ConfigParser()
            cf.read(configFile)
            if cf['Training']['trainer'] == 'mallet':
                MallTr = MalletTrainer(cf, modelFolder=configFile.parent)
                MallTr.fit()

        else:
            print('You need to provide a valid configuration file')

