"""
* *IntelComp H2020 project*

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
import pandas as pd
import pyarrow.parquet as pt
#import numpy as np
#import time
#import re
#import regex as javare
import sys
from pathlib import Path
#from gensim import corpora
#from gensim.utils import check_output
import subprocess
from subprocess import check_output
#from sklearn.preprocessing import normalize
from .utils.misc import query_options, var_num_keyboard, request_confirmation
from .utils.misc import printgr, printred, printmag
#from topicmodeler.topicmodeling import MalletTrainer, TMmodel

class TaskManager(object):
    """
    Main class to manage functionality of the Topic Model Interactive Trainer

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'cfReady'     : If True, config file succesfully loaded. Datamanager activated.
    """

    # This is a dictionary that contains a list to all subdirectories
    # that should exist in the project folder
    _dir_struct = {
        'datasets': Path('datasets'),
        'LDAmodels': Path('LDAmodels')
        }

    _config_fname = 'config.cf'

    def __init__(self, p2p, p2parquet):
        """
        Initiates the Task Manager object

        We declare here all variables that will be used by the TaskManager
        If no value is available, they are set to None and will be
        initialized later

        Parameters
        ----------
        p2p : pathlib.Path
            Path to the application project

        p2parquet : pathlib.Path
            Path to the folder hosting the parquet datasets
        """

        # Important directories for the project
        self.p2p = p2p
        self.p2parquet = p2parquet

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
        self.cf.optionxform = str #Preserves case of keys in config file
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

    def fromHDFS(self):
        """
        This method simulates the download of a corpus from the IntelComp data space
        
        In the version that will be taken to production, this method will not be necessary
        and data will be directly retrieved from the data datalogue using IntelComp mediators

        This needs to be linked with the Data mediator
        """

        displaytext = """
        *************************************************************************************
        This method simulates the download of a corpus from the IntelComp data space
        
        In the version that will be taken to production, this method will not be necessary
        and data will be directly retrieved from the data catalogue using IntelComp mediators

        This needs to be linked with the Data mediator
        *************************************************************************************
        """
        printred(displaytext)

        # We need the user to specify table, fields to include, filtering conditions
        # Available tables in HDFS are read from config file
        tables = {}
        for key in self.cf['HDFS']:
            tables[key] = self.cf['HDFS'][key]
        tables_list = [el for el in tables.keys()]
        table_opt = query_options(tables_list, 'Select the dataset you wish to download')
        parquet_table = tables[tables_list[table_opt]]

        #Select fields to include
        print('\nReference to available fields: https://intelcomp-uc3m.atlassian.net/wiki/spaces/INTELCOMPU/pages/884737/Status+of+UC3M+data+sets+for+WP2')
        selectFields = "fieldsOfStudy, year, ... (id not necessary)"
        sf = ''
        while not len(sf):
            sf = input(f"Fields to include in dataset [{selectFields}]: ")
        selectFields = ",".join([el.strip() for el in sf.split(',')])

        filterCondition = "array_contains(fieldsOfStudy, 'Computer Science')"
        filterCondition = input(f"Filter to apply [{filterCondition}]: ")
        # This is not very smart. Used for being able to send arguments with
        # "'" or " " to the spark job
        filterCondition = filterCondition.replace(' ','SsS').replace("'","XxX")
        
        # We need a name for the dataset
        dtsName = ""
        while not len(dtsName):
            dtsName = input('Introduce a name for the dataset: ')
        if not dtsName.endswith('.parquet'):
            dtsName += '.parquet'
        path_dataset = self.p2parquet.joinpath(dtsName)
        path_dataset.mkdir(parents=True, exist_ok=True)

        # Introduce a description for the dataset
        dtsDesc = ""
        while not len(dtsDesc):
            dtsDesc = input('Introduce a description for the dataset: ')

        #Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the dataset')
        privacy = privacy[opt]


        #printgr('Parquet_table: ' + parquet_table)
        #printgr('SelectFields: ' + selectFields)
        #printgr('filterCondition: '  + filterCondition)
        #printgr('Pathdataset: ' +path_dataset.resolve().as_posix())
        options = '"-p ' + parquet_table + ' -s ' + selectFields + \
                  ' -d ' + path_dataset.resolve().as_posix()
        if len(filterCondition):
            options = options + ' -f ' + filterCondition + '"'
        else:
            options = options + '"'
        #=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # This fragment of code creates a sparck cluster and submits the task
        # This function is dependent on UC3M local deployment infrastructure
        # and will not work in BSC production environment
        # In any case, this function will be replaced by the DataCatalogue
        # import functionalities, so no need to worry about setting it right,
        # it will not get into production
        #=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        script_spark = self.cf.get('Spark', 'script_spark')
        token_spark = self.cf.get('Spark', 'token_spark')
        script_path = '/export/usuarios_ml4ds/jarenas/github/IntelComp/ITMT/topicmodeler/aux/fromHDFS/fromHDFS.py'
        cmd = script_spark + ' -C ' + token_spark + \
              ' -c 4 -N 10 -S ' + script_path + ' -P ' + options
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
        #=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        datasetMeta = {
            "name"          : dtsName,
            "description"   : dtsDesc,
            "visibility"    : privacy,
            "download_date" : DT.datetime.now(),
            #"records"       : len(pd.read_parquet(path_dataset, columns=[])),
            "records"       : sum([pt.read_table(el, columns=[]).num_rows 
                                        for el in path_dataset.iterdir()
                                        if el.name.endswith('.parquet')]),
            "source"        : tables_list[table_opt],
            "schema"        : pt.read_schema([el for el in path_dataset.iterdir()
                                        if el.name.endswith('.parquet')][0]).names
            }

        with path_dataset.joinpath('datasetMeta.json').open('w', encoding='utf-8') as outfile:
            json.dump(datasetMeta, outfile, ensure_ascii=False, indent=2, default=str)

        return
        
    def listDownloaded(self):
        """
        This method shows all Datasets that have been retrieved from HDFS
        and are available for the Model Trainer

        This is a extremely simple method for the taskmanager that does not
        require any user interaction

        """
        cmd = 'python src/manageCorpus/manageCorpus.py --listDownloaded --parquet '
        cmd = cmd + self.p2parquet.resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            allDtsets = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        allDtsets = json.loads(allDtsets)
        for Dts in allDtsets.keys():
            printmag('\nDataset ' + allDtsets[Dts]['name'])
            print('\tSource:', allDtsets[Dts]['source'])
            print('\tDescription:', allDtsets[Dts]['description'])
            print('\tFields:', ', '.join([el for el in allDtsets[Dts]['schema']]))
            print('\tNumber of docs:', allDtsets[Dts]['records'])
            print('\tDownload date:', allDtsets[Dts]['download_date'])
            print('\tVisibility:', allDtsets[Dts]['visibility'])
        
        return allDtsets

    def createTMCorpus(self):
        """
        This method creates a training dataset for Topic Modeling
        """

        # We need first to download all available datasets
        cmd = 'python src/manageCorpus/manageCorpus.py --listDownloaded --parquet '
        cmd = cmd + self.p2parquet.resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            allDtsets = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        allDtsets = json.loads(allDtsets)

        # Now we start user interaction to gather datasets
        displaytext = """
        *************************************************************************************
        Generation of Training corpus for Topic Modeling

        You need to select one or more data sets; for each data set you need to select:

            1 - The column that will be used as the id
            2 - The columns that will be used for the rawtext
            3 - The columns that contain the lemmas
            4 - Any additional filtering condition in Spark format (advanced users only)
                (e.g.: array_contains(fieldsOfStudy, 'Computer Science'))
            5 - A domain selection model (To be implemented)
            6 - A set of level three FOS codes (To be implemented)
        
        *************************************************************************************
        """
        printgr(displaytext)
        
        Dtsets = [el for el in allDtsets.keys()]
        options = [allDtsets[el]['name'] for el in Dtsets] + ['Finish selection']
        TM_Dtset = []
        exit = False
        while not exit:
            opt = query_options(options, '\nSelect a corpus for the training dataset')
            if opt == len(options)-1:
                exit = True
            else:
                Dtset_loc = Dtsets.pop(opt)
                Dtset_source = allDtsets[Dtset_loc]['source']
                options.pop(opt)
                print('\nProcessing dataset', allDtsets[Dtset_loc]['name'])
                print('Available columns:', allDtsets[Dtset_loc]['schema'])
                
                #id fld
                Dtset_idfld = ''
                while Dtset_idfld not in allDtsets[Dtset_loc]['schema']:
                    Dtset_idfld = input('Select the field to use as identifier: ')
                
                #lemmas fields
                Dtset_lemmas_fld = input('Select fields for lemmas (separated by commas): ')
                Dtset_lemmas_fld = [el.strip() for el in Dtset_lemmas_fld.split(',')]
                Dtset_lemmas_fld = [el for el in Dtset_lemmas_fld
                                        if el in allDtsets[Dtset_loc]['schema']]
                print('Selected:', ', '.join(Dtset_lemmas_fld))
                
                #rawtext fields
                Dtset_rawtext_fld = input('Select fields for rawtext (separated by commas): ')
                Dtset_rawtext_fld = [el.strip() for el in Dtset_rawtext_fld.split(',')]
                Dtset_rawtext_fld = [el for el in Dtset_rawtext_fld
                                        if el in allDtsets[Dtset_loc]['schema']]
                print('Selected:', ', '.join(Dtset_rawtext_fld))

                #Spark clause for filtering (advanced users only)
                Dtset_filter = input('Introduce a filtering condition for Spark clause (advanced users): ')
                
                TM_Dtset.append({'parquet'    : Dtset_loc,
                                 'source'     : Dtset_source,
                                 'idfld'      : Dtset_idfld,
                                 'lemmasfld'  : Dtset_lemmas_fld,
                                 'rawtxtfld'  : Dtset_rawtext_fld,
                                 'filter'     : Dtset_filter
                    })

        # We need a name for the dataset
        dtsName = ""
        while not len(dtsName):
            dtsName = input('Introduce a name for the training dataset: ')

        # Introduce a description for the dataset
        dtsDesc = ""
        while not len(dtsDesc):
            dtsDesc = input('Introduce a description: ')

        #Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the dataset')
        privacy = privacy[opt]

        Dtset = {'name'         : dtsName,
                 'description'  : dtsDesc,
                 'valid_for'    : "TM",
                 'visibility'   : privacy,
                 'Dtsets'       : TM_Dtset
        }

        cmd = 'echo "' + json.dumps(Dtset).replace('"', '\\"') + '"'
        cmd = cmd + '| python src/manageCorpus/manageCorpus.py --saveTrDtset --path_datasets '
        cmd = cmd + self.p2p.joinpath(self._dir_struct['datasets']).resolve().as_posix()
        
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            status = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        return status

    def listTMCorpus(self):
        """
        This method shows all (logical) Datasets available for training 
        Topic Models

        This is a extremely simple method for the taskmanager that does not
        require any user interaction

        """
        cmd = 'python src/manageCorpus/manageCorpus.py --listTrDtsets --path_datasets '
        cmd = cmd + self.p2p.joinpath(self._dir_struct['datasets']).resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            allTrDtsets = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        allTrDtsets = json.loads(allTrDtsets)
        for TrDts in allTrDtsets.keys():
            printmag('\nTraining Dataset ' + allTrDtsets[TrDts]['name'])
            print('\tDescription:', allTrDtsets[TrDts]['description'])
            print('\tValid for:', allTrDtsets[TrDts]['valid_for'])
            print('\tCreation date:', allTrDtsets[TrDts]['creation_date'])
            print('\tVisibility:', allTrDtsets[TrDts]['visibility'])
        
        return allTrDtsets

    def deleteTMCorpus(self):
        """
        Delete Training Corpus from the Interactive Topic Model Trainer
        dataset folder
        """

        allTrDtsets = self.listTMCorpus()

        for TrDts in allTrDtsets.keys():
            Y_or_N = input(f"\nRemove Training Set {allTrDtsets[TrDts]['name']} [Y/N]?: ")
            if Y_or_N.upper() == "Y":
                if request_confirmation(msg='Training Dataset ' + allTrDtsets[TrDts]['name'] + ' will be deleted. Proceed?'):
                    cmd = 'python src/manageCorpus/manageCorpus.py --deleteTrDtset --path_TrDtset '
                    cmd = cmd + TrDts
                    printred(cmd)
                    try:
                        self.logger.info(f'-- -- Running command {cmd}')
                        status = check_output(args=cmd, shell=True)
                        if status.decode('utf8') == '1':
                            print('The training set was deleted')
                        else:
                            print('Deletion failed')
                    except:
                        self.logger.error('-- -- Execution of script failed')
                        print('Deletion failed')

        return
        

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
            self.logger.error('-- -- Execution of script failed')

        return


    def trainTM(self, trainer):
        """
        Topic modeling trainer. Initial training of a topic model

        Parameters
        ----------
        trainer:
            TM optimizer [mallet/ctm]
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

        #We also need the user to select/confirm number of topics
        ntopics = int(self.cf.get('TM','ntopics'))
        ntopics = var_num_keyboard('int', ntopics, 
                'Please, select the number of topics')

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
            alpha = float(self.cf.get('MalletTM','alpha'))
            optimize_interval = int(self.cf.get('MalletTM','optimize_interval'))
            num_threads = int(self.cf.get('MalletTM','num_threads'))
            num_iterations = int(self.cf.get('MalletTM','num_iterations'))
            doc_topic_thr = float(self.cf.get('MalletTM','doc_topic_thr'))
            thetas_thr = float(self.cf.get('MalletTM','thetas_thr'))

            #The following settings will only be accessed in the "advance settings panel"
            Y_or_N = input(f"Do you wish to access the advance settings panel [Y/N]?:")
            if Y_or_N.upper() == "Y":
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
                eq = input('Enter the path of an alternative file with equivalent terms: ')
                if len(eq):
                    if Path(eq.strip()).is_file():
                        eq_file = eq.strip()
                alpha = var_num_keyboard('float', alpha, 
                    'Prior parameter for the Dirichlet for doc generation')
                optimize_interval = var_num_keyboard('int', optimize_interval, 
                    'Iterations between Dirichlet priors optimization')
                num_threads = var_num_keyboard('int', num_threads, 
                    'Number of threads for mallet parallelization')
                num_iterations = var_num_keyboard('int', num_iterations, 
                    'Number of Gibbs Sampling iterations')
                doc_topic_thr = var_num_keyboard('float', doc_topic_thr, 
                    'Threshold for topic activation in a doc (mallet training)')
                thetas_thr = var_num_keyboard('float', thetas_thr, 
                    'Threshold for topic activation in a doc (sparsification)')

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
                fout.write('ntopics = ' + str(ntopics) + '\n')
                fout.write('alpha = ' + str(alpha) + '\n')
                fout.write('optimize_interval = ' + str(optimize_interval) + '\n')
                fout.write('num_threads = ' + str(num_threads) + '\n')
                fout.write('num_iterations = ' + str(num_iterations) + '\n')
                fout.write('doc_topic_thr = ' + str(doc_topic_thr) + '\n')
                fout.write('thetas_thr = ' + str(thetas_thr) + '\n')
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
        `self.validate_subtrain_models()`
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
