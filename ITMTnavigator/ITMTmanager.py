"""
*** IntelComp H2020 project ***

Task Manager for the Interactive Topic Model Trainer App
It implements the functions needed to
  - Generate training datasets from datalake collections
  - Train topic models
  - Curate topic models
  - Do inference with topic models
"""

import shutil
import configparser
import logging
import datetime as DT
import json
#import numpy as np
#import time
#import re
#import regex as javare
import sys
from pathlib import Path
#from gensim import corpora
from gensim.utils import check_output
#from sklearn.preprocessing import normalize
from utils.misc import query_options, var_num_keyboard, request_confirmation
#from utils.misc import printgr, printred, printmag

#from topicmodeler.topicmodeling import MalletTrainer, TMmodel

class TaskManager(object):
    """
    Main class to manage functionality of the Topic Model Interactive Trainer

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'cfReady'     : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    # This is a dictionary that contains a list to all subdirectories
    # that should exist in the project folder
    _dir_struct = {
        'datasets': Path('datasets'),
        'LDAmodels': Path('LDAmodels')
        }

    _config_fname = 'config.cf'

    def __init__(self, p2p):
        """
        Initiates the Task Manager object

        We declare here all variables that will be used by the TaskManager
        If no value is available, they are set to None and will be
        initialized later

        Args:

        : p2p: String containing the name of the project
               (also the name of the folder that contains the project)
        """

        # Important directories for the project
        self.p2p = Path(p2p)

        # Configuration file
        self.p2config = self.p2p.joinpath(self._config_fname)

        # State variables that will be loaded from the metadata file when
        # when the project was loaded.
        self.state = {
            'isProject': False,     # True if the project exist.
            'cfReady': False}       # True if config file could be loaded

        # Other class variables
        self.cf = None         # Handler to the config file

        # Logger variables
        self.logformat = None
        self.logger = None

        # Other class variables
        self.ready2setup = False  # True after create() or load() are called
        print('-- Task Manager object succesfully initialized')

        return

    def create(self):
        """
        Creates a project instance for the Topic Model Trainer
        To do so, it defines the main folder structure, and creates (or cleans)
        the project folder, specified in self.p2p
        """

        # Check and clean project folder location
        if self.p2p.exists():

            # Remove current backup folder, if it exists
            old_p2p = Path(str(self.p2p) + '_old')
            if old_p2p.exists():
                shutil.rmtree(old_p2p)

            # Copy current project folder to the backup folder.
            shutil.move(self.p2p, old_p2p)
            print(f'-- -- Existing project with same name moved to {old_p2p}')

        # Create project folder and subfolders
        self.p2p.mkdir()
        for folder in self._dir_struct:
            self.p2p.joinpath(
                self._dir_struct[folder]).mkdir(parents=True)

        # Place a copy of a default configuration file in the project folder.
        shutil.copyfile('config.cf.default', self.p2config)

        # Update the state of the project.
        self.state['isProject'] = True
        print(f'-- Project {self.p2p} has been created')

        self._setup()

        return

    def load(self):
        """
        Loads an existing Interactive Topic Modeling Trainer project
        """

        # Check and clean project folder location
        if not self.p2p.exists():
            print(f'-- Project {self.p2p} not found. Create it before loading')
            return

        else:
            # Check integrity of the project structure
            valid = True
            if not self.p2config.exists():
                valid = False

            for folder in self._dir_struct:
                if not self.p2p.joinpath(self._dir_struct[folder]).exists():
                    valid = False

        if not valid:
            print(f'-- Project {self.p2p} folder structure incorrect. Stopping')
            sys.exit()
        else:
            print(f'-- Project {self.p2p} successfully loaded')
            self.state['isProject'] = True
            self._setup()

        return

    def _setup(self):
        """
        Sets up the project. To do so:
            - Loads the configuration file
        """

        # Loads configuration file
        self.cf = configparser.ConfigParser()
        self.cf.read(self.p2config)
        self.state['cfReady'] = True

        # Set up the logging format
        self._set_logs()
        self.state['configReady'] = True
        self.logger.info('Project setup finished')

        return

    def _set_logs(self):
        """
        Configure logging messages.
        """

        self.logformat = {
            'filename': self.cf.get('logformat', 'filename'),
            'datefmt': self.cf.get('logformat', 'datefmt'),
            'file_format': self.cf.get('logformat', 'file_format'),
            'file_level': self.cf.get('logformat', 'file_level'),
            'cons_level': self.cf.get('logformat', 'cons_level'),
            'cons_format': self.cf.get('logformat', 'cons_format')}

        # Log to file and console
        fpath = self.p2p / self.logformat['filename']

        logging.basicConfig(
            level=self.logformat['file_level'],
            format=self.logformat['file_format'],
            datefmt=self.logformat['datefmt'], filename=str(fpath),
            filemode='w')

        # Define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(self.logformat['cons_level'])

        # Set a simple format for console use
        formatter = logging.Formatter(fmt=self.logformat['cons_format'],
                                      datefmt=self.logformat['datefmt'])

        # Tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.info('Logs activated')

        # This is a logger objet, that can be used by specific modules
        self.logger = logging.getLogger('')

        return

    def generateCorpus(self):
        """
        Generate a training corpus for topic modeling or other tasks

        This needs to be linked with the Data mediator
        """

        ############################################################
        ## IMT Interface: New Dataset POPUP
        ############################################################

        # We need the user to specify table, fields to include, filtering conditions

        parquet_table = "parquet.`/export/ml4ds/IntelComp/Datalake/SemanticScholar/20220201/papers.parquet`"
        pt = input(f"Parquet table [{parquet_table}]: ")
        if len(pt):
            parquet_table = pt

        selectFields = ["id", "lemmas"]
        sf = input(f"Fields to include in dataset [{selectFields}]: ")
        if len(sf):
            selectFields = sf.split(",")

        filterCondition = "array_contains(fieldsOfStudy, 'Computer Science')"
        fc = input(f"Filter to apply [{filterCondition}]: ")
        if len(fc):
            filterCondition = fc

        # We also need a name for the dataset
        dtsName = ""
        while not len(dtsName):
            dtsName = input('Introduce a name for the dataset: ')
        path_dataset = self.p2p.joinpath(
            self._dir_struct['datasets']).joinpath(dtsName)
        path_dataset.mkdir(parents=True, exist_ok=True)

        #
        #Now, we can call the "fake Data Mediator"
        query = "SELECT " + (",").join(selectFields) + \
                " FROM " + parquet_table
        if len(filterCondition.strip()):
            query += " WHERE " + filterCondition

        datasetMeta = {
            "name"          : dtsName,
            "query"         : query,
            "validfor"      : ["TM"],
            "date"          : DT.datetime.now()
            }

        with path_dataset.joinpath('config.json').open('w', encoding='utf-8') as outfile:
            json.dump(datasetMeta, outfile, ensure_ascii=False, indent=2, default=str)

        cmd = '/export/usuarios_ml4ds/jarenas/script-spark/script-spark ' + \
              '-C /export/usuarios_ml4ds/jarenas/script-spark/tokencluster.json ' + \
              '-c 4 -N 10 -S "generateCorpus.py --p ' + \
              path_dataset.as_posix() + '"'
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Generation of script failed')

        return

        
    def removeCorpus(self):
        """
        Remove a training corpus from the Interactive Topic Model Trainer
        """
        ############################################################
        ## IMT Interface: Datasets window - remove selected datasets
        ############################################################
        dtSets = self.p2p.joinpath(self._dir_struct['datasets']).iterdir()
        for el in dtSets:
            if el.is_dir():
                Y_or_N = input(f"Remove Training Set {el.name} [Y/N]?:")
                if Y_or_N.upper() == "Y":
                    if request_confirmation(msg='Model ' + el.name + ' will be deleted. Proceed?'):
                        shutil.rmtree(el)
        

    def corpus2JSON(self):
        """
        Remove a training corpus from the Interactive Topic Model Trainer

        This is linked to the ITMTrainer only tentatively, since it should
        be part of WP4 tools

        Right now, it only runs the generation of the JSON files that are
        needed for ingestion of the corpus in Solr
        """

        print('corpus2JSON')

        cmd = '/export/usuarios_ml4ds/jarenas/script-spark/script-spark ' + \
              '-C /export/usuarios_ml4ds/jarenas/script-spark/tokencluster.json ' + \
              '-c 4 -N 10 -S "corpus2JSON.py"'
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Generation of script failed')

        return


    def trainTM(self, trainer):
        """
        Topic modeling trainer. Initial training of a topic model

        Args:
        :param trainer: TM optimizer [mallet/ctm]
        """
        
        ############################################################
        ## IMT Interface: Topic Model Trainer Window
        ############################################################

        self.logger.info(f'-- Topic Model Training')
        
        #First thing to do is to select a corpus
        #Ask user which dataset should be used for model training
        dtSets = self.p2p.joinpath(self._dir_struct['datasets']).iterdir()
        dtSets = sorted([d for d in dtSets if d.is_dir()])
        display_dtSets = [d.name for d in dtSets]
        selection = query_options(display_dtSets, "Select Training Dataset")
        path_dtSet = dtSets[selection]
        #Retrieve all CSV files that form the selected dataset
        dtSetCSV = sorted([f for f in path_dtSet.joinpath("CSV").iterdir()
                                if f.name.endswith(".csv")])
        self.logger.info(f'-- -- Selected corpus is {path_dtSet.name}')
        
        #Retrieve parameters for training
        if trainer=="mallet":
            #Default values are read from config file
            min_lemas = int(self.cf.get('MalletTM', 'min_lemas'))
            no_below = int(self.cf.get('MalletTM','no_below'))
            no_above = float(self.cf.get('MalletTM','no_above'))
            keep_n = int(self.cf.get('MalletTM','keep_n'))
            token_regexp = self.cf.get('MalletTM','token_regexp')
            mallet_path = self.cf.get('MalletTM','mallet_path')
            stw_file = [self.cf.get('MalletTM', 'default_stw_file')]
            eq_file = [self.cf.get('MalletTM', 'default_eq_file')]

            #Some of them can be confirmed/modified by the user
            min_lemas = var_num_keyboard('int', min_lemas, 
                'Enter minimum number of lemas for the documents in the training set')
            no_below = var_num_keyboard('int', no_below, 
                'Minimum number occurrences to keep words in vocabulary')
            no_above = var_num_keyboard('float', no_above, 
                'Maximum proportion of documents to keep a word in vocabulary')
            keep_n = var_num_keyboard('int', keep_n, 
                'Maximum vocabulary size')
            tk = input(f'Regular expresion for tokenizer [{token_regexp}]: ')
            if len(tk):
                token_regexp = tk
            print(f'Stopwords will be read from file {stw_file[0]}')
            swf = input('Enter the path of additional files of stopwords (separated by commas): ')
            if len(swf):
                for f in swf.split(','):
                    if Path(f.strip()).is_file():
                        stw_file.append(f.strip())
            print(f'Word equivalences will be read from file {eq_file[0]}')
            eq = input('Enter the path of additional files of equivalences (separated by commas): ')
            if len(eq):
                for f in eq.split(','):
                    if Path(f.strip()).is_file():
                        eq_file.append(f.strip())

            # Create folder for corpus, backup existing folder if necessary
            modelname = input('Enter a name to save the new model: ')
            modeldir = self.p2p.joinpath(self._dir_struct['LDAmodels']).joinpath(modelname)
            if modeldir.exists():

                # Remove current backup folder, if it exists
                old_model_dir = Path(str(modeldir) + '_old/')
                if old_model_dir.exists():
                    shutil.rmtree(old_model_dir)

                # Copy current project folder to the backup folder.
                shutil.move(modeldir, old_model_dir)
                self.logger.info(f'-- -- Creating backup of existing model in {old_model_dir}')

            # Create corpus_folder and save model training configuration
            modeldir.mkdir()
            configFile = modeldir.joinpath('train.config')
            with configFile.open('w', encoding='utf8') as fout:
                fout.write('[Preproc]\n')
                fout.write('min_lemas = ' + str(min_lemas) + '\n')
                fout.write('no_below = ' + str(no_below) + '\n')
                fout.write('no_above = ' + str(no_above) + '\n')
                fout.write('keep_n = ' + str(keep_n) + '\n')
                fout.write('stw_file = ' + ','.join(stw_file) + '\n')
                fout.write('eq_file = ' + ','.join(eq_file) + '\n')
                fout.write('\n[Training]\n')
                fout.write('trainer = mallet\n')
                fout.write('token_regexp = ' + str(token_regexp) + '\n')
                fout.write('mallet_path = ' + mallet_path + '\n')
                fout.write('training_files = ' + ','.join([el.as_posix() for el in dtSetCSV])+'\n')

            #############################################################
            ## END IMT Interface: Next, the actual training should happen
            #############################################################

            # Run command for training model
            cmd = f'python topicmodeling.py --train --config {configFile.as_posix()}'
            try:
                self.logger.info(f'-- -- Running command {cmd}')
                check_output(args=cmd, shell=True)
            except:
                self.logger.error('-- -- Command execution failed')

        if trainer=="ctm":
            #Other trainers will be available
            pass

        return
        

        #Initialize lemmatizer
        stwEQ = stwEQcleaner(stw_files=[stw_file, corpus_stw], dict_eq_file=corpus_eqs,
                             logger=self.logger)

        #Identification of words that are too rare or common that need to be 
        #removed from the dictionary. 
        self.logger.info('-- -- Corpus Generation: Vocabulary generation')
        dictionary = corpora.Dictionary()

        #Read reference table, and important columns for the specific corpus
        ref_table = self.cf.get(corpus, 'ref_table')
        ref_col = self.cf.get(corpus, 'ref_col')
        meta_fields = self.cf.get(corpus, 'meta_fields').split(',')
        meta_fields = [el for el in meta_fields if len(el)]

        selectOptions = ref_col + ',LEMAS'
        filterOptions = 'LEMAS IS NOT NULL'

        for df in self.DMs[corpus].readDBchunks(ref_table, ref_col, chunksize=chunksize,
                                    selectOptions=selectOptions, limit=None,
                                    filterOptions=filterOptions, verbose=True):
            id_lemas = df.values.tolist()
            id_lemas = [[el[0], ' '.join(token_regexp.findall(el[1].replace('\n',' ').strip()))]
                            for el in id_lemas]
            id_lemas = [[el[0], stwEQ.cleanstr(el[1]).split()] for el in id_lemas]
            id_lemas = [el for el in id_lemas if len(el[1])>=min_lemas]
            #id_lemas = [[el[0], el[1].replace('\n',' ').strip().split()] for el in id_lemas]
            #id_lemas = [el for el in id_lemas if len(el[1])>=min_lemas]
            all_lemas = [el[1] for el in id_lemas]
            dictionary.add_documents(all_lemas)

        #Remove words that appear in less than no_below documents, or in more than
        #no_above, and keep at most keep_n most frequent terms, keep track of removed
        #words for debugging purposes
        all_words = [dictionary[idx] for idx in range(len(dictionary))]
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        kept_words = set([dictionary[idx] for idx in range(len(dictionary))])
        rmv_words = [el for el in all_words if el not in kept_words]
        #Save extreme words that will be removed
        self.logger.info(f'-- -- Saving {len(rmv_words)} extreme words to file')
        rmv_file = corpus_dir.joinpath(corpus + '_commonrare_words.txt')
        with rmv_file.open('w', encoding='utf-8') as fout:
            [fout.write(el+'\n') for el in sorted(rmv_words)]
        #Save dictionary to file
        self.logger.info(f'-- -- Saving dictionary to file. Number of words: {len(kept_words)}')
        vocab_txt = corpus_dir.joinpath(corpus + '_vocabulary.txt')
        with vocab_txt.open('w', encoding='utf-8') as fout:
            [fout.write(el+'\n') for el in sorted(kept_words)]
        #Save also in gensim text format
        vocab_gensim = corpus_dir.joinpath(corpus + '_vocabulary.gensim')
        dictionary.save_as_text(vocab_gensim)
        
        ##################################################
        #Create corpus and metadata files
        self.logger.info('-- -- Corpus generation: Corpus and Metadata files')
        
        meta_file = corpus_dir.joinpath(corpus + '_metadata.csv')
        corpus_file = corpus_dir.joinpath(corpus + '_corpus.txt')
        corpus_mallet = corpus_dir.joinpath(corpus + '_corpus.mallet')

        if len(meta_fields):
            firstLineMeta = ref_col + ',' + ','.join(meta_fields)
        else:
            firstLineMeta = ref_col
        self.logger.debug(f'-- -- Heading of metadata file: {firstLineMeta}')

        selectOptions = firstLineMeta + ',LEMAS'

        fmeta = meta_file.open('w', encoding='utf-8')
        fmeta.write(firstLineMeta + '\n')
        fcorpus = corpus_file.open('w', encoding='utf-8')
        
        for df in self.DMs[corpus].readDBchunks(ref_table, ref_col, chunksize=chunksize,
                            selectOptions=selectOptions, limit=None,
                            filterOptions=filterOptions, verbose=True):
            id_lemas = df.applymap(str).values.tolist()
            id_lemas = [el[:-1] + [token_regexp.findall(el[-1].replace('\n',' ').strip())]
                            for el in id_lemas]
            id_lemas = [el[:-1] + [[tk for tk in el[-1] if tk in kept_words]]
                            for el in id_lemas]
            id_lemas = [el[:-1] + [stwEQ.cleanstr(' '.join(el[-1])).split()]
                            for el in id_lemas]
            id_lemas = [el for el in id_lemas if len(el[-1])>=min_lemas]
            #Write to corpus file
            [fcorpus.write(el[0] + ' 0 ' + ' '.join(el[-1]) + '\n') for el in id_lemas]
            #Write to metadata file
            if len(meta_fields):
                #Remove commas and new lines
                id_meta = [[el2.replace(',',' ').replace('\n',' ') for el2 in el[:-1]]
                            for el in id_lemas]
                [fmeta.write(','.join(el)+'\n') for el in id_meta]
            else:
                [fmeta.write(el[0]+'\n') for el in id_lemas]

        fmeta.close()
        fcorpus.close()
        
        ##################################################
        #Importing Data to mallet
        self.logger.info('-- -- Corpus Generation: Mallet Data Import')

        mallet_regexp=self.cf.get('TM','mallet_regexp')
        
        cmd = str(mallet_path) + \
              ' import-file --preserve-case --keep-sequence ' + \
              '--remove-stopwords --token-regex "' + mallet_regexp + '" ' + \
              '--input %s --output %s'
        cmd = cmd % (corpus_file, corpus_mallet)

        try:
            self.logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Mallet failed to import data. Revise command')

        return

    def trainTM_old(self, corpus):

        #A proper corpus with BoW, vocabulary, etc .. should exist
        path_corpus = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_corpus = path_corpus.joinpath(corpus).joinpath(corpus+'_corpus.mallet')
        if path_corpus.is_file():
            self.logger.info(f'-- Training topic model on corpus: {corpus}')
        else:
            self.logger.error(f'-- Corpus {corpus} does not exist')
            return

        #Read default values for some parameters
        mallet_path = Path(self.cf.get('TM', 'mallet_path'))
        num_threads = int(self.cf.get('TM', 'num_threads'))
        num_iterations = int(self.cf.get('TM', 'num_iterations'))
        doc_topic_thr = float(self.cf.get('TM', 'doc_topic_thr'))
        thetas_thr = float(self.cf.get('TM', 'thetas_thr'))
        sparse_block = int(self.cf.get('TM', 'sparse_block'))

        #Ask user to introduce manually the name for the Model
        givenName = input('Introduce a name for the model: ')
        #An automatic model_name will be generated according to existing
        #final models and the model name provided by the user
        #Final models are enumerated as corpusN_givenName
        path_model = self.p2p.joinpath(corpus).joinpath(self._dir_struct['modtm'])
        path_model = path_model.joinpath(givenName.replace(' ', '_'))
        if path_model.exists():
            if request_confirmation(msg='A model with that name already exists. ¿Overwrite?'):
                shutil.rmtree(path_model)
            else:
                return
        path_model.mkdir()

        #Create Topic model
        MallTr = MalletTrainer(corpusFile=path_corpus, outputFolder=path_model,
                        mallet_path=mallet_path, numThreads=num_threads,
                        numIterations=num_iterations, docTopicsThreshold=doc_topic_thr,
                        sparse_thr=thetas_thr, sparse_block=sparse_block,
                        logger = self.logger)
        MallTr.adj_settings()
        MallTr.fit()
        
        return

    def extractPipe(self, corpus):

        #A proper corpus with BoW, vocabulary, etc .. should exist
        path_corpus = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_corpus = path_corpus.joinpath(corpus).joinpath(corpus+'_corpus.mallet')
        if not path_corpus.is_file():
            self.logger.error('-- Pipe extraction: Could not locate corpus file')
            return

        #Create auxiliary file with only first line from the original corpus file
        path_txt = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_txt = path_txt.joinpath(corpus).joinpath(corpus+'_corpus.txt')
        with path_txt.open('r', encoding='utf8') as f:
            first_line = f.readline()

        path_aux = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_aux = path_aux.joinpath(corpus).joinpath('corpus_aux.txt')
        with path_aux.open('w', encoding='utf8') as fout:
            fout.write(first_line+'\n')

        ##################################################
        #We perform the import with the only goal to keep a small
        #file containing the pipe
        self.logger.info('-- Extracting pipeline')
        mallet_path = Path(self.cf.get('TM','mallet_path'))
        path_pipe = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_pipe = path_pipe.joinpath(corpus).joinpath('import.pipe')
        cmd = str(mallet_path) + \
              ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_corpus, path_aux, path_pipe)

        try:
            self.logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- Failed to extract pipeline. Revise command')


        #Remove auxiliary file
        path_aux.unlink()

        return

    def inference(self, corpus):

        #A proper corpus should exist with the corresponding ipmortation pipe
        path_pipe = self.p2p.joinpath(corpus).joinpath(self._dir_struct['corpus'])
        path_pipe = path_pipe.joinpath(corpus).joinpath('import.pipe')
        if not path_pipe.is_file():
            self.logger.error('-- Inference error. Importation pipeline not found')
            return

        #Ask user which model should be used for inference
        #Final models are enumerated as corpus_givenName
        path_model = self.p2p.joinpath(corpus).joinpath(self._dir_struct['modtm'])
        models = sorted([d for d in path_model.iterdir() if d.is_dir()])
        display_models = [' '.join(d.name.split('_')) for d in models]
        selection = query_options(display_models, 'Select model for the inference')
        path_model = models[selection]
        inferencer = path_model.joinpath('inferencer.mallet')

        #Ask user to provide a valid text file for performing inference
        #Format of the text will be one document per line, only text
        #Note all created files will be hosted in same directory, so a good idea
        #would be to put the file into an empty directory for this purpose
        while True:
            txt_file = input('Introduce complete path to file with texts for the inference: ')
            txt_file = Path(txt_file)
            if not txt_file.is_file():
                print('Please provide a valid file name')
                continue
            else:
                break

        #The following files will be generated in the same folder
        corpus_file = Path(str(txt_file) + '_corpus.txt') #lemmatized texts
        corpus_mallet_inf = Path(str(txt_file) + '_corpus.mallet') #mallet serialized
        doc_topics_file = Path(str(txt_file) + '_doc-topics.txt')  #Topic proportions
        #Reorder topic proportions in numpy format
        doc_topics_file_npy = Path(str(txt_file) + '_doc-topics.npy')
        
        #Start processing pipeline
        
        #========================
        # 1. Lemmatization
        #========================
        self.logger.info('-- Inference: Lemmatizing Titles and Abstracts ...')
        lemmas_server = self.cf.get('Lemmatizer', 'server')
        stw_file = Path(self.cf.get('Lemmatizer', 'default_stw_file'))
        dict_eq_file = Path(self.cf.get('Lemmatizer', 'default_dict_eq_file'))
        POS = self.cf.get('Lemmatizer', 'POS')
        concurrent_posts = int(self.cf.get('Lemmatizer', 'concurrent_posts'))
        removenumbers = self.cf.get('Lemmatizer', 'removenumbers') == 'True'
        keepSentence = self.cf.get('Lemmatizer', 'keepSentence') == 'True'

        #Initialize lemmatizer
        ENLM = ENLemmatizer(lemmas_server=lemmas_server, stw_file=stw_file,
                    dict_eq_file=dict_eq_file, POS=POS, removenumbers=removenumbers,
                    keepSentence=keepSentence, logger=self.logger)
        with txt_file.open('r', encoding='utf8') as fin:
            docs = fin.readlines()
        docs = [[el.split()[0], ' '.join(el.split()[1:])] for el in docs]
        docs = [[el[0], clean_utf8(el[1])] for el in docs]
        lemasBatch = ENLM.lemmatizeBatch(docs, processes=concurrent_posts)
        #Remove entries that where not lemmatized correctly
        lemasBatch = [[el[0], clean_utf8(el[1])] for el in lemasBatch if len(el[1])]
            
        #========================
        # 2. Tokenization and application of specific stopwords
        #    and equivalences for the corpus
        #========================
        self.logger.info('-- Inference: Applying corpus specific stopwords and equivalences')
        token_regexp=javare.compile(self.cf.get('CorpusGeneration','token_regexp'))
        corpus_stw = Path(self.cf.get(corpus,'stw_file'))
        corpus_eqs = Path(self.cf.get(corpus,'eq_file'))

        #Initialize Cleaner
        stwEQ = stwEQcleaner(stw_files=[stw_file,corpus_stw], dict_eq_file=corpus_eqs,
                             logger=self.logger)
        #tokenization with regular expression
        id_lemas = [[el[0], ' '.join(token_regexp.findall(el[1]))]
                            for el in lemasBatch]
        #stopwords and equivalences
        id_lemas = [[el[0], stwEQ.cleanstr(el[1])] for el in id_lemas]
        #No need to apply other transformations, because only known words
        #in the vocabulary will be used by Mallet for the topic-inference
        with corpus_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + ' 0 ' + el[1] + '\n') for el in id_lemas]

        #========================
        #3. Importing Data to mallet
        #========================
        self.logger.info('-- Inference: Mallet Data Import')
        mallet_path = Path(self.cf.get('TM','mallet_path'))
        
        cmd = str(mallet_path) + \
              ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_pipe, corpus_file, corpus_mallet_inf)

        try:
            self.logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- Mallet failed to import data. Revise command')
            return
        
        #========================
        #4. Get topic proportions
        #========================
        self.logger.info('-- Inference: Inferring Topic Proportions')
        num_iterations = int(self.cf.get('TM', 'num_iterations_inf'))
        doc_topic_thr = float(self.cf.get('TM', 'doc_topic_thr'))
        
        cmd = str(mallet_path) + \
              ' infer-topics --inferencer %s --input %s --output-doc-topics %s ' + \
              ' --doc-topics-threshold ' + str(doc_topic_thr) + \
              ' --num-iterations ' + str(num_iterations)
        cmd = cmd % (inferencer, corpus_mallet_inf, doc_topics_file)
        
        try:
            self.logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- Mallet inference failed. Revise command')
            return

        #========================
        #5. Apply model editions
        #========================
        self.logger.info('-- Inference: Applying model edition transformations')
        #Load thetas file, apply model edition actions, and save as a numpy array
        #We need to read the number of topics, e.g. from train_config file
        train_config = path_model.joinpath('train.config')
        with train_config.open('r', encoding='utf8') as fin:
            num_topics = [el for el in fin.readlines() if el.startswith('num-topics')][0]
            num_topics = int(num_topics.strip().split(' = ')[1])
        cols = [k for k in np.arange(2,num_topics+2)]
        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t', dtype=np.float32, usecols=cols)
        model_edits = path_model.joinpath('model_edits.txt')
        if model_edits.is_file():
            with model_edits.open('r', encoding='utf8') as fin:
                for line in fin:
                    line_els = line.strip().split()
                    if line_els[0]=='s':
                        idx = [int(el) for el in line_els[1:]]
                        thetas32 = thetas32[:,idx]
                    elif line_els[0]=='d':
                        tpc = int(line_els[1])
                        ntopics = thetas32.shape[1]
                        tpc_keep = [k for k in range(ntopics) if k!=tpc]
                        thetas32 = thetas32[:,tpc_keep]
                        thetas32 = normalize(thetas32,axis=1,norm='l1')
                    elif line_els[0]=='f':
                        tpcs = [int(el) for el in line_els[1:]]
                        thet = np.sum(thetas32[:,tpcs],axis=1)
                        thetas32[:,tpcs[0]] = thet
                        thetas32 = np.delete(thetas32,tpcs[1:],1)

        thetas32 = normalize(thetas32,axis=1,norm='l1')
        np.save(doc_topics_file_npy,thetas32)
        
        return

    def editTM(self, corpus):

        # Select model for edition
        #Final models are enumerated as corpus_givenName
        path_model = self.p2p.joinpath(corpus).joinpath(self._dir_struct['modtm'])
        models = sorted([d for d in path_model.iterdir() if d.is_dir()])
        display_models = [' '.join(d.name.split('_')) for d in models]
        selection = query_options(
            display_models, 'Select the topic model that you want to edit:')
        path_model = models[selection]
        tm = TMmodel(from_file=path_model.joinpath('modelo.npz'), logger=self.logger)

        corpus_size = tm.get_thetas().shape[0]
        var_exit2 = False
        modified = False
        print('\n---------')
        print('You are editing topic model:', path_model)
        print('---------\n')

        options = ['Salir',
                   'Visualizar tópicos',
                   'Visualizar las palabras de los tópicos',
                   'Visualizar palabras de tópicos "basura" vs otros tópicos',
                   'Exportar Visualización pyLDAvis',
                   'Anotación de Stopwords y Equivalencias',
                   'Etiquetado automático de los tópicos del modelo',
                   'Etiquetado manual de tópicos',
                   'Eliminar un tópico del modelo',
                   'Tópicos similares por coocurrencia',
                   'Tópicos similares por palabras',
                   'Fusionar dos tópicos del modelo',
                   'Ordenar tópicos por importancia',
                   'Resetear al modelo original']

        while not var_exit2:

            msg = 'Available options'
            selection = options[query_options(options, msg)]

            if selection == 'Salir':
                var_exit2 = True

            elif selection == 'Visualizar tópicos':
                tm.muestra_descriptions()

            elif selection == 'Visualizar las palabras de los tópicos':
                tm.muestra_descriptions(simple=True)

            elif selection == 'Visualizar palabras de tópicos "basura" vs otros tópicos':

                ntopics = tm.get_ntopics()
                msg = '\nIntroduce el ID de los tópicos "basura": '
                r = input(msg)
                try:
                    tpcsGbg = [int(n) for n in r.split(',')
                               if int(n) >= 0 and int(n) < ntopics]
                except:
                    tpcsGbg = []
                if len(tpcsGbg):
                    tpcsOth = [k for k in range(ntopics) if k not in tpcsGbg]
                else:
                    print('No se ha introdido ningún id de tópico válido')
                    return

                # Ahora seleccionamos el número de palabras a seleccionar de los tópicos
                # basura, y el número de palabras del resto de tópicos
                msg = '\nIntroduzca el peso máximo de las palabras de los tópicos basura'
                weighWordsGbg = var_num_keyboard('float', 0.001, msg)
                msg = '\nIntroduzca ahora el peso máximo para las palabras de otros tópicos'
                weighWordsOth = var_num_keyboard('float', 0.01, msg)

                #Y por último seleccionamos las palabras y hacemos la intersección
                wordsOth = []
                for tpc in tpcsOth:
                    wordstpc = tm.most_significant_words_per_topic(n_palabras=10000,tfidf=True,tpc=[tpc])
                    if wordstpc[0][-1][1] / wordstpc[0][0][1] > weighWordsOth:
                        printred('Se supera el límite preestablecido de palabras para el tópico ' + str(tpc))
                    else:
                        wordstpc = [el[0] for el in wordstpc[0] if el[1]/wordstpc[0][0][1]>weighWordsOth]
                        wordsOth += wordstpc
                wordsOth = set(wordsOth)

                for tpc in tpcsGbg:
                    wordstpc = tm.most_significant_words_per_topic(n_palabras=10000,tfidf=True,tpc=[tpc])
                    if wordstpc[0][-1][1] / wordstpc[0][0][1] > weighWordsGbg:
                        printred('Se supera el límite preestablecido de palabras para el tópico ' + str(tpc))
                    else:
                        wordstpc = [el[0] for el in wordstpc[0] if el[1]/wordstpc[0][0][1]>weighWordsGbg]
                        printgr(40 * '=')
                        printgr('Tópico ' + str(tpc))
                        printgr('Seleccionadas ' + str(len(wordstpc)) + ' palabras')
                        printgr(40 * '=')
                        stwCandidates = [el for el in wordstpc if el not in wordsOth]
                        printmag('Candidatas a StopWord (' + str(len(stwCandidates)) + '):')
                        print(stwCandidates)
                        nonStwCandidates = [el for el in wordstpc if el in wordsOth]
                        printmag('Coincidentes con otros tópicos (' + str(len(nonStwCandidates)) + '):')
                        print(nonStwCandidates)

            elif selection == 'Exportar Visualización pyLDAvis':
                tm.pyLDAvis(path_model.joinpath('pyLDAvis.html').as_posix(),
                    ndocs=int(self.cf.get('TMedit', 'LDAvis_ndocs')),
                    njobs=int(self.cf.get('TMedit', 'LDAvis_njobs')))

            elif selection == 'Anotación de Stopwords y Equivalencias':
                n_palabras = int(self.cf.get('TMedit', 'n_palabras'))
                round_size = int(self.cf.get('TMedit', 'round_size'))
                words = tm.most_significant_words_per_topic(
                    n_palabras=n_palabras, tfidf=True, tpc=None)
                words = [[wd[0] for wd in el] for el in words]
                # Launch labelling application
                stw, eqs = tagfilter.stw_eq_tool(words, round_size=round_size)
                eqs = [el[0] + ' : ' + el[1] for el in eqs]
                # Save stopwords and equivalences in corpus specific files
                corpus_stw = Path(self.cf.get(corpus, 'stw_file'))
                corpus_eqs = Path(self.cf.get(corpus, 'eq_file'))
                if corpus_stw.is_file():
                    with corpus_stw.open('r', encoding='utf8') as fin:
                        current_stw = fin.readlines()
                    current_stw = [el.strip() for el in current_stw]
                    stw = stw + current_stw
                stw = sorted(list(set(stw)))
                with corpus_stw.open('w', encoding='utf8') as fout:
                    [fout.write(wd + '\n') for wd in stw]
                # Same for equivalent words
                if corpus_eqs.is_file():
                    with corpus_eqs.open('r', encoding='utf8') as fin:
                        current_eqs = fin.readlines()
                    current_eqs = [el.strip() for el in current_eqs]
                    eqs = eqs + current_eqs
                eqs = sorted(list(set(eqs)))
                with corpus_eqs.open('w', encoding='utf8') as fout:
                    [fout.write(eq + '\n') for eq in eqs]

                return

            elif selection == 'Etiquetado automático de los tópicos del modelo':
                tm.automatic_topic_labeling(self.cf.get('TM', 'pathlabeling'),
                    workers=int(self.cf.get('TMedit', 'NETLworkers')))
                modified = True

            elif selection == 'Etiquetado manual de tópicos':
                descriptions = tm.get_descriptions()
                word_description = tm.get_topic_word_descriptions()
                for desc, (tpc, worddesc) in zip(descriptions, word_description):
                    print('=' * 5)
                    print('Topic ID:', tpc)
                    print('Current description:', desc)
                    print('Word description:', worddesc)
                    r = input('\nIntroduce the description for the topic, write "wd" use Word Description,\n' + \
                               'or press enter to keep current:\n')
                    if r=='wd':
                        tm.set_description(worddesc, tpc)
                    elif r!='':
                        tm.set_description(r, tpc)
                modified = True

            elif selection == 'Eliminar un tópico del modelo':
                lista = [(ndoc, desc) for ndoc, desc in zip(tm.ndocs_active_topic(), tm.get_topic_word_descriptions())]
                for k in sorted(lista, key=lambda x: -x[0]):
                    print('='*5)
                    perc = '('+str(round(100*float(k[0])/corpus_size)) + ' %)'
                    print('ID del tópico:', k[1][0])
                    print('Número de documentos en los que está activo:', k[0], perc)
                    print('Palabras más significativas:', k[1][1])
                msg = '\nIntroduce el ID de los tópicos que deseas eliminar separados por comas'
                msg += '\no presiona ENTER si no deseas eliminar ningún tópico\n'
                r = input(msg)
                try:
                    tpcs = [int(n) for n in r.split(',')]
                    #Eliminaremos los tópicos en orden decreciente
                    tpcs.sort(reverse=True)
                    for tpc in tpcs:
                        tm.delete_topic(tpc)
                    modified = True
                except:
                    print('Debes introducir una lista de topic ids')

            elif selection == 'Tópicos similares por coocurrencia':
                msg = '\nIntroduzca el número de pares de tópicos a mostrar'
                npairs = var_num_keyboard('int', 5, msg)
                msg = '\nIntroduzca número de palabras para mostrar'
                nwords = var_num_keyboard('int', 10, msg)
                selected = tm.get_similar_corrcoef(npairs)
                for pair in selected:
                    msg = 'Correlación de los tópicos {0:d} y {1:d}: {2:.2f}%'.format(pair[0], pair[1], 100*pair[2])
                    printmag(msg)
                    printmag(20*'=')
                    tm.muestra_perfiles(n_palabras=nwords,tpc=[pair[0], pair[1]])
                printred(20*'=')
                printred('Cuidado: los ids de los tópicos cambian tras la fusión o eliminación')
                printred(20*'=')

            elif selection == 'Tópicos similares por palabras':
                msg = '\nIntroduzca umbral para selección de palabras'
                thr = var_num_keyboard('float', 1e-3, msg)
                msg = '\nIntroduzca el número de pares de tópicos a mostrar'
                npairs = var_num_keyboard('int', 5, msg)
                msg = '\nIntroduzca número de palabras para mostrar'
                nwords = var_num_keyboard('int', 10, msg)
                selected = tm.get_similar_JSdist(npairs, thr)
                for pair in selected:
                    msg = 'Similitud de los tópicos {0:d} y {1:d}: {2:.2f}%'.format(pair[0], pair[1], 100*pair[2])
                    printmag(msg)
                    printmag(20*'=')
                    tm.muestra_perfiles(n_palabras=nwords,tpc=[pair[0], pair[1]])
                printred(20*'=')
                printred('Cuidado: los ids de los tópicos cambian tras la fusión o eliminación')
                printred(20*'=')

            elif selection == 'Fusionar dos tópicos del modelo':
                tm.muestra_descriptions()
                msg = '\nIntroduce el ID de los tópicos que deseas fusionar separados por comas'
                msg += '\no presiona ENTER si no deseas fusionar ningún tópico\n'
                r = input(msg)
                try:
                    tpcs = [int(n) for n in r.split(',')]
                    if len(tpcs)>=2:
                        tm.fuse_topics(tpcs)
                    modified = True
                except:
                    print('Debes introducir una lista de IDs')

            elif selection == 'Ordenar tópicos por importancia':
                tm.sort_topics()
                modified = True

            elif selection == 'Resetear al modelo original':
                tm.reset_model()
                modified = True

        if modified:
            if request_confirmation(msg='Save modified model?'):
                tm.save_npz(path_model.joinpath('modelo.npz'))

        return

    def compute_all_sim_graphs(self, corpus_name):
        """
        Computes all similarity graphs from the available topic models for a
        given corpus, and save them in a supergraph structure, to be used
        later in validation processes.

        Parameters
        ----------
        corpus : str
            Corpus (S24Ever: Semantic Scholar, Crunch4Ever: Crunchbase)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.compute_all_sim_graphs()

        return

    def compute_reference_graph(self, corpus_name):
        """
        Computes a reference graph for a given corpus, based on metadata.

        Parameters
        ----------
        corpus_name : str
            Name of the corpus
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.compute_reference_graph()

        return

    def validate_topic_models(self, corpus_name):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        corpus_name: str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.validate_topic_models()

        return

    def show_validation_results(self, corpus_name):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        corpus_name: str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.show_validation_results()

        return

    def analyze_variability(self, corpus_name):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated from the analisys of the variability
        of node relationships in the graph

        Parameters
        ----------
        corpus : str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.analyze_variability()

        return

    def show_variability_results(self, corpus_name):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        corpus_name: str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.show_variability_results()

        return

    def analyze_scalability(self, corpus_name):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        corpus_name: str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.analyze_scalability()

        return

    def show_scalability_results(self, corpus_name):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        corpus_name: str
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        # Path to the validation folder for the given corpus
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.show_scalability_results()

        return

    def validate_subtrain_models(self):
        """
        Validates topics models obtained using a reduced corpus, using a
        gold standard based o a large corpus
        """

        # Path to the validation folder for the given corpus
        corpus_name = 'S24Ever'
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.validate_subtrain_models()

        return

    def show_subtrain_results(self):
        """
        Shows the results of the topic model validation in
        self.validate_subtrain_models()
        """

        # Path to the validation folder for the given corpus
        corpus_name = 'S24Ever'
        path2val = self.p2p / corpus_name / self._dir_struct['val']
        # Output path to the given corpus
        path2out = self.p2p / corpus_name / self._dir_struct['valoutput']
        # Path to the topic models folder
        model2val = self.models_2_validate[corpus_name]
        V = Validator(corpus_name, self.DMs, model2val, path2val, path2out,
                      **self.val_params)
        V.show_subtrain_results()

        return
