"""
* *IntelComp H2020 project*

Task Manager for the Interactive Topic Model Trainer App
It implements the functions needed to

    - Generate training datasets from datalake collections
    - Train topic models
    - Curate topic models
    - Do inference with topic models
"""

# import configparser
import datetime as DT
import json
import pathlib
# import logging
import shutil
# from gensim import corpora
# import numpy as np
# import time
# import re
# import regex as javare
# import sys
from pathlib import Path
from subprocess import check_output

# import pandas as pd
import pyarrow.parquet as pt
# from sklearn.preprocessing import normalize
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMessageBox
from src.gui.utils.utils import clearQTreeWidget, get_model_xml, printTree
from src.utils.misc import (printgr, printmag, printred, query_options,
                            request_confirmation, var_num_keyboard)

from ..gui.utils.constants import Constants
from .base_task_manager import BaseTaskManager


class ITMTTaskManager(BaseTaskManager):
    """
    This class extends the functionality of the baseTaskManager class for a
    specific example application
    This class inherits from the baseTaskManager class, which provides the
    basic method to create, load and set up an application project.
    The behavior of this class might depend on the state of the project, in
    dictionary self.state, with the following entries:
    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file successfully loaded.
    """

    def __init__(self, p2p, p2parquet, p2wdlist, config_fname='config.cf',
                 metadata_fname='metadata.yaml'):
        """
        Initializes an ITMTTaskManager object.

        Parameters
        ----------
        p2p : pathlib.Path
            Path to the application project
        p2parquet : pathlib.Path
            Path to the folder hosting the parquet datasets
        p2wdlist : pathlib.Path
            Path to the folder hosting the wordlists (stopwords, keywords, etc)
        config_fname : str, optional (default='config.cf')
            Name of the configuration file
        metadata_fname : str or None, optional (default=metadata.yaml)
            Name of the project metadata file.
            If None, no metadata file is used.
        """

        # Attributes that will be initialized in the base class
        self.p2p = None
        self.path2metadata = None
        self.p2config = None
        self.p2parquet = None
        self.p2wdlist = None
        self.metadata_fname = None
        self.cf = None
        self.state = None
        self.metadata = None
        self.ready2setup = None
        self.logger = None

        # Attributes to load project's associated datasets and WordLists, etc
        self.allDtsets = None
        self.allTrDtsets = None
        self.allWdLists = None
        self.allTMmodels = None

        super().__init__(p2p, p2parquet, p2wdlist, config_fname=config_fname,
                         metadata_fname=metadata_fname)

        # This is a dictionary that contains a list to all subdirectories
        # that should exist in the project folder
        self._dir_struct = {'datasets': 'datasets',
                            'LDAmodels': 'LDAmodels'}

        return

    def load(self):
        """
        Extends the load method from the parent class to load into execution time the datasets that have been
        retrieved from HDFS and are available in the p2parquet provided by the user, as well as all the (logical)
        Datasets available for the training of topic models which were created in previous executions.
        """

        super().load()

        self.load_listDownloaded()
        self.load_listTrDtsets()
        self.load_listWdLists()
        self.load_listTMmodels()

        return

    def load_listDownloaded(self):
        """
        This method loads all the datasets that have been retrieved from HDFS and are available for the Model Trainer
        into the ITMTTaskManager's 'allDtsets' attribute as a dictionary object, which is characterized by one dictionary
        entry per dataset, the key and the value being the absolute path to the dataset and a dictionary with the
        corresponding metadata, respectively. To do so, it invokes the script from the folder 'src/manageCorpus' with
        the option 'listDownloaded'.
        """

        # @TODO: Check whether it is better to utilized subprocess.pipe to invoke this and following functions' cmds.

        cmd = 'python src/manageCorpus/manageCorpus.py --listDownloaded --parquet '
        cmd = cmd + self.p2parquet.resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            self.allDtsets = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        self.logger.info("Downloaded datasets loaded")
        return

    def load_listTrDtsets(self):
        """
        This method loads all the (logical) datasets that are available for the training of topic models into the
        ITMTTaskManager's 'allTrDtsets' attribute as a dictionary object, which is characterized by one dictionary
        entry per dataset, the key and the value being the absolute path to the dataset and a dictionary with the
        corresponding metadata, respectively. To do so, it invokes the script from the folder 'src/manageCorpus' with
        the option 'listTrDtsets'.
        """

        cmd = 'python src/manageCorpus/manageCorpus.py --listTrDtsets --path_datasets '
        cmd = cmd + \
            self.p2p.joinpath(
                self._dir_struct['datasets']).resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            self.allTrDtsets = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        self.logger.info("Logical datasets loaded")
        return

    def load_listWdLists(self):
        """
        This method loads all the wordlists that are available into the
        ITMTTaskManager's 'allWdLists' attribute as a dictionary object,
        which is characterized by one dictionary entry per wordlist,
        the key and the value being the absolute path to the wordlist
        and a dictionary with the corresponding metadata, respectively.
        To do so, it invokes the script from the folder 'src/manageLists' with
        the option 'listWordLists'.
        """

        cmd = 'python src/manageLists/manageLists.py --listWordLists --path_wordlists '
        cmd = cmd + self.p2wdlist.resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            self.allWdLists = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        self.logger.info("All available wordlists were loaded")
        return

    def load_listTMmodels(self):
        """
        This method loads all the available Topic Models previously created by the user ITMTTaskManager's 'allTMmodels' attribute as a dictionary object, which is characterized by one dictionary entry per Topic Model, the key and the value being the absolute path to the model and a dictionary with the corresponding metadata, respectively. To do so, it invokes the script from the folder 'src topicmodeling' with the option 'listTMmodels'.
        """

        cmd = 'python src/topicmodeling/topicmodeling.py --listTMmodels --path_models '
        cmd = cmd + \
            self.p2p.joinpath(
                self._dir_struct['LDAmodels']).resolve().as_posix()
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            self.allTMmodels = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        self.logger.info("Logical datasets loaded")
        return

    def save_TrDtset(self, dt_set):
        """
        This method saves the (logical) training dataset specified by 'dt_set' in the dataset folder contained within
        the project folder. To do so, it invokes the script from the folder 'src/manageCorpus' with the option
        'saveTrDtset'.

        Parameters
        ----------
        dt_set :
            Dictionary with Training Dataset information

        Returns
        -------
        status : int
            - 0 if the dataset could not be created
            - 1 if the dataset was created successfully
            - 2 if the dataset replaced an existing dataset
        """

        cmd = 'echo "' + json.dumps(dt_set).replace('"', '\\"') + '"'
        cmd = cmd + '| python src/manageCorpus/manageCorpus.py --saveTrDtset --path_datasets '
        cmd = cmd + \
            self.p2p.joinpath(
                self._dir_struct['datasets']).resolve().as_posix()

        try:
            self.logger.info(f'-- -- Running command {cmd}')
            status = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        # Reload the list of training datasets to consider the new ones added during the current execution
        self.load_listTrDtsets()

        return status

    def delete_TrDtset(self, dt_set):
        """
        This method deletes the (logical) training dataset specified by 'dt_set' from the dataset folder contained
        within the project folder. To do so, it invokes the script from the folder 'src/manageCorpus' with the option
        'deleteTrDtset'.

        Parameters
        ----------
        dt_set : str
            String representation of the path to the json file with the training dataset information

        Returns
        -------
        status : int
            - 0 if the dataset could not be deleted
            - 1 if the dataset was deleted successfully
        """

        cmd = 'python src/manageCorpus/manageCorpus.py --deleteTrDtset --path_TrDtset '
        cmd = cmd + dt_set
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

        # Reload the list of training datasets to consider the ones deleted during the current execution
        self.load_listTrDtsets()

        return status

    def create_List(self, new_list):
        """
        This method saves the stopword list 'stw_list' in the folder self.p2wdlist
        To do so, it invokes the script from the folder 'src/manageLists'
        with the option 'createWordList'.

        Parameters
        ----------
        new_list :
            Dictionary with information for the new list

        Returns
        -------
        status : int
            - 0 if the list could not be created
            - 1 if the list was created successfully
            - 2 if the list replaced an existing dataset
        """

        cmd = 'echo "' + json.dumps(new_list).replace('"', '\\"') + '"'
        cmd = cmd + '| python src/manageLists/manageLists.py --createWordList --path_wordlists '
        cmd = cmd + self.p2wdlist.resolve().as_posix()

        try:
            self.logger.info(f'-- -- Running command {cmd}')
            status = check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- -- Execution of script failed')
            return

        # Reload the list of word lists to consider the new ones added during the current execution
        self.load_listWdLists()

        return status

    def delete_WdLst(self, wd_list):
        """
        This method deletes the wordlist specified by 'wd_list' from the wordlist folder.

        To do so, it invokes the script from the folder 'src/manageLists' with the option
        'deleteWordList'.

        Parameters
        ----------
        wd_list : str
            String representation of the path to the json file with the wordlist

        Returns
        -------
        status : int
            - 0 if the wordlist could not be deleted
            - 1 if the wordlist was deleted successfully
        """

        cmd = 'python src/manageLists/manageLists.py --deleteWordList --path_WdList '
        cmd = cmd + wd_list
        printred(cmd)
        try:
            self.logger.info(f'-- -- Running command {cmd}')
            status = check_output(args=cmd, shell=True)
            if status.decode('utf8') == '1':
                print('The word list was deleted')
            else:
                print('Deletion failed')
        except:
            self.logger.error('-- -- Execution of script failed')
            print('Deletion failed')

        # Reload the list of word lists to consider the one deleted during the current execution
        self.load_listWdLists()

        return status

    def trainTM(self, modelname, ModelDesc, privacy, trainer, TrDtSet, Preproc, training_params):
        """
        Topic modeling trainer. Initial training of a topic model

        Parameters
        ----------
        trainer : string
            Optimizer to use for training the topic model
            Possible values are mallet|sparkLDA|prodLDA|ctm
        """

        # 1. Create model directory
        modeldir = self.p2p.joinpath(
            self._dir_struct['LDAmodels']).joinpath(modelname)
        if modeldir.exists():

            # Remove current backup folder, if it exists
            old_model_dir = Path(str(modeldir) + '_old/')
            if old_model_dir.exists():
                shutil.rmtree(old_model_dir)

            # Copy current project folder to the backup folder.
            shutil.move(modeldir, old_model_dir)
            self.logger.info(
                f'-- -- Creating backup of existing model in {old_model_dir}')

        # 2. Create corpus_folder and save model training configuration
        modeldir.mkdir()
        configFile = modeldir.joinpath('trainconfig.json')

        train_config = {
            "name": modelname,
            "description": ModelDesc,
            "visibility": privacy,
            "trainer": trainer,
            "TrDtSet": TrDtSet,
            "Preproc": Preproc,
            "LDAparam": training_params,
            "creation_date": DT.datetime.now()
        }

        with configFile.open('w', encoding='utf-8') as outfile:
            json.dump(train_config, outfile,
                      ensure_ascii=False, indent=2, default=str)

        # 3. Topic Modeling starts
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # This fragment of code creates a spark cluster and submits the task
        # This function is dependent on UC3M local deployment infrastructure
        # and will not work in BSC production environment
        #
        # Needs to be modified with the BSC Spark Cluster and/or CITE SparkSubmit
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        # Step 1: Preprocessing of Training Data
        if self.cf.get('Spark', 'spark_available') == 'True':
            script_spark = self.cf.get('Spark', 'script_spark')
            token_spark = self.cf.get('Spark', 'token_spark')
            script_path = './src/topicmodeling/topicmodeling.py'
            options = '"--spark --preproc --config ' + configFile.resolve().as_posix() + '"'
            cmd = script_spark + ' -C ' + token_spark + \
                ' -c 4 -N 10 -S ' + script_path + ' -P ' + options
            printred(cmd)
            try:
                self.logger.info(f'-- -- Running command {cmd}')
                output = check_output(args=cmd, shell=True)
            except:
                self.logger.error('-- -- Execution of script failed')

        else:
            # Run command for corpus preprocessing using gensim
            cmd = f'python topicmodeling.py --preproc --config {configFile.as_posix()}'
            printred(cmd)
            try:
                self.logger.info(f'-- -- Running command {cmd}')
                output = check_output(args=cmd, shell=True)
            except:
                self.logger.error('-- -- Command execution failed')

        # Step 2: Training of Topic Model
        if trainer == "sparkLDA":
            if not self.cf.get('Spark', 'spark_available') == 'True':
                self.logger.error(
                    "-- -- sparkLDA requires access to a Spark cluster")
            else:
                script_spark = self.cf.get('Spark', 'script_spark')
                token_spark = self.cf.get('Spark', 'token_spark')
                script_path = './src/topicmodeling/topicmodeling.py'
                options = '"--spark --train --config ' + configFile.resolve().as_posix() + '"'
                cmd = script_spark + ' -C ' + token_spark + \
                    ' -c 4 -N 10 -S ' + script_path + ' -P ' + options
                printred(cmd)
                try:
                    self.logger.info(f'-- -- Running command {cmd}')
                    check_output(args=cmd, shell=True)
                except:
                    self.logger.error('-- -- Execution of script failed')

        else:
            # Other models do not require Spark
            cmd = f'python topicmodeling.py --train --config {configFile.as_posix()}'
            printred(cmd)
            try:
                self.logger.info(f'-- -- Running command {cmd}')
                output = check_output(args=cmd, shell=True)
            except:
                self.logger.error('-- -- Command execution failed')

        # Reload the list of topic models to consider the one created in the current execution
        self.load_listTMmodels()

        return


##############################################################################
#                          ITMTTaskManagerCMD                                #
##############################################################################
class ITMTTaskManagerCMD(ITMTTaskManager):
    """
    Provides extra functionality to the task manager, requesting parameters
    from users from a terminal.
    """

    def __init__(self, p2p, p2parquet, p2wdlist, config_fname='config.cf',
                 metadata_fname='metadata.yaml'):
        """
        Initializes an ITMTTaskManagerCMD object.

        Parameters
        ----------
        p2p : pathlib.Path
            Path to the application project
        p2parquet : pathlib.Path
            Path to the folder hosting the parquet datasets
        p2wdlist : pathlib.Path
            Path to the folder hosting the wordlists (stopwords, keywords, etc)
        config_fname : str, optional (default='config.cf')
            Name of the configuration file
        metadata_fname : str or None, optional (default=metadata.yaml)
            Name of the project metadata file.
            If None, no metadata file is used.
        """

        super().__init__(
            p2p, p2parquet, p2wdlist, config_fname=config_fname,
            metadata_fname=metadata_fname)

        # super().load()

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
        table_opt = query_options(
            tables_list, 'Select the dataset you wish to download')
        parquet_table = tables[tables_list[table_opt]]

        # Select fields to include
        print(
            '\nReference to available fields: https://intelcomp-uc3m.atlassian.net/wiki/spaces/INTELCOMPU/pages'
            '/884737/Status+of+UC3M+data+sets+for+WP2')
        selectFields = "fieldsOfStudy, year, ... (id not necessary)"
        sf = ''
        while not len(sf):
            sf = input(f"Fields to include in dataset [{selectFields}]: ")
        selectFields = ",".join([el.strip() for el in sf.split(',')])

        filterCondition = "array_contains(fieldsOfStudy, 'Computer Science')"
        filterCondition = input(f"Filter to apply [{filterCondition}]: ")
        # This is not very smart. Used for being able to send arguments with
        # "'" or " " to the spark job
        filterCondition = filterCondition.replace(
            ' ', 'SsS').replace("'", "XxX")

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

        # Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the dataset')
        privacy = privacy[opt]

        # printgr('Parquet_table: ' + parquet_table)
        # printgr('SelectFields: ' + selectFields)
        # printgr('filterCondition: '  + filterCondition)
        # printgr('Pathdataset: ' +path_dataset.resolve().as_posix())
        options = '"-p ' + parquet_table + ' -s ' + selectFields + \
                  ' -d ' + path_dataset.resolve().as_posix()
        if len(filterCondition):
            options = options + ' -f ' + filterCondition + '"'
        else:
            options = options + '"'
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
        # This fragment of code creates a spark cluster and submits the task
        # This function is dependent on UC3M local deployment infrastructure
        # and will not work in BSC production environment
        # In any case, this function will be replaced by the DataCatalogue
        # import functionalities, so no need to worry about setting it right,
        # it will not get into production
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
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
        # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

        datasetMeta = {
            "name": dtsName,
            "description": dtsDesc,
            "visibility": privacy,
            "download_date": DT.datetime.now(),
            # "records"       : len(pd.read_parquet(path_dataset, columns=[])),
            "records": sum([pt.read_table(el, columns=[]).num_rows
                            for el in path_dataset.iterdir()
                            if el.name.endswith('.parquet')]),
            "source": tables_list[table_opt],
            "schema": pt.read_schema([el for el in path_dataset.iterdir()
                                      if el.name.endswith('.parquet')][0]).names
        }

        path_datasetMeta = self.p2parquet.joinpath('datasetMeta.json')
        if path_datasetMeta.is_file():
            with path_datasetMeta.open('r', encoding='utf-8') as infile:
                allMeta = json.load(infile)
            allMeta[dtsName] = datasetMeta
            with path_datasetMeta.open('w', encoding='utf-8') as outfile:
                json.dump(allMeta, outfile, ensure_ascii=False,
                          indent=2, default=str)
        else:
            datasetMeta = {dtsName: datasetMeta}
            with path_datasetMeta.open('w', encoding='utf-8') as outfile:
                json.dump(datasetMeta, outfile, ensure_ascii=False,
                          indent=2, default=str)

        self.load_listDownloaded()

        return

    def listDownloaded(self):
        """
        This method shows all Datasets that have been retrieved from HDFS
        and are available for the Model Trainer in the command line window.

        This is an extremely simple method for the taskmanager that does not
        require any user interaction.
        """

        allDtsets = json.loads(self.allDtsets)
        for Dts in allDtsets.keys():
            printmag('\nDataset ' + allDtsets[Dts]['name'])
            print('\tSource:', allDtsets[Dts]['source'])
            print('\tDescription:', allDtsets[Dts]['description'])
            print('\tFields:', ', '.join(
                [el for el in allDtsets[Dts]['schema']]))
            print('\tNumber of docs:', allDtsets[Dts]['records'])
            print('\tDownload date:', allDtsets[Dts]['download_date'])
            print('\tVisibility:', allDtsets[Dts]['visibility'])

        return

    def createTMCorpus(self):
        """
        This method creates a training dataset for Topic Modeling
        """

        # We need first to get all available (downloaded) datasets
        allDtsets = json.loads(self.allDtsets)

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
        options = [allDtsets[el]['name']
                   for el in Dtsets] + ['Finish selection']
        TM_Dtset = []
        exit = False
        while not exit:
            opt = query_options(
                options, '\nSelect a corpus for the training dataset')
            if opt == len(options) - 1:
                exit = True
            else:
                Dtset_loc = Dtsets.pop(opt)
                Dtset_source = allDtsets[Dtset_loc]['source']
                options.pop(opt)
                print('\nProcessing dataset', allDtsets[Dtset_loc]['name'])
                print('Available columns:', allDtsets[Dtset_loc]['schema'])

                # id fld
                Dtset_idfld = ''
                while Dtset_idfld not in allDtsets[Dtset_loc]['schema']:
                    Dtset_idfld = input(
                        'Select the field to use as identifier: ')

                # lemmas fields
                Dtset_lemmas_fld = input(
                    'Select fields for lemmas (separated by commas): ')
                Dtset_lemmas_fld = [el.strip()
                                    for el in Dtset_lemmas_fld.split(',')]
                Dtset_lemmas_fld = [el for el in Dtset_lemmas_fld
                                    if el in allDtsets[Dtset_loc]['schema']]
                print('Selected:', ', '.join(Dtset_lemmas_fld))

                # rawtext fields
                Dtset_rawtext_fld = input(
                    'Select fields for rawtext (separated by commas): ')
                Dtset_rawtext_fld = [el.strip()
                                     for el in Dtset_rawtext_fld.split(',')]
                Dtset_rawtext_fld = [el for el in Dtset_rawtext_fld
                                     if el in allDtsets[Dtset_loc]['schema']]
                print('Selected:', ', '.join(Dtset_rawtext_fld))

                # Spark clause for filtering (advanced users only)
                Dtset_filter = input(
                    'Introduce a filtering condition for Spark clause (advanced users): ')

                TM_Dtset.append({'parquet': Dtset_loc,
                                 'source': Dtset_source,
                                 'idfld': Dtset_idfld,
                                 'lemmasfld': Dtset_lemmas_fld,
                                 'rawtxtfld': Dtset_rawtext_fld,
                                 'filter': Dtset_filter
                                 })

        # We need a name for the dataset
        dtsName = ""
        while not len(dtsName):
            dtsName = input('Introduce a name for the training dataset: ')

        # Introduce a description for the dataset
        dtsDesc = ""
        while not len(dtsDesc):
            dtsDesc = input('Introduce a description: ')

        # Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the dataset')
        privacy = privacy[opt]

        Dtset = {'name': dtsName,
                 'description': dtsDesc,
                 'valid_for': "TM",
                 'visibility': privacy,
                 'Dtsets': TM_Dtset
                 }

        return self.save_TrDtset(Dtset)

    def listTMCorpus(self):
        """
        This method shows all (logical) Datasets available for training 
        Topic Models in the command line window.

        This is an extremely simple method for the taskmanager that does not
        require any user interaction

        """

        allTrDtsets = json.loads(self.allTrDtsets)
        for TrDts in allTrDtsets.keys():
            printmag('\nTraining Dataset ' + allTrDtsets[TrDts]['name'])
            print('\tDescription:', allTrDtsets[TrDts]['description'])
            print('\tValid for:', allTrDtsets[TrDts]['valid_for'])
            print('\tCreation date:', allTrDtsets[TrDts]['creation_date'])
            print('\tVisibility:', allTrDtsets[TrDts]['visibility'])

        return

    def deleteTMCorpus(self):
        """
        Delete Training Corpus from the Interactive Topic Model Trainer
        dataset folder
        """

        # Show available training datasets
        self.listTMCorpus()

        allTrDtsets = json.loads(self.allTrDtsets)
        for TrDts in allTrDtsets.keys():
            Y_or_N = input(
                f"\nRemove Training Set {allTrDtsets[TrDts]['name']} [Y/N]?: ")
            if Y_or_N.upper() == "Y":
                if request_confirmation(
                        msg='Training Dataset ' + allTrDtsets[TrDts]['name'] + ' will be deleted. Proceed?'):
                    self.delete_TrDtset(TrDts)

        return

    def listAllWdLists(self):
        """
        This method shows all wordlists available for the project

        This is an extremely simple method for the taskmanager that does not
        require any user interaction

        """

        allWdLists = json.loads(self.allWdLists)
        for WdLst in allWdLists.keys():
            printmag('\nWordlist ' + allWdLists[WdLst]['name'])
            print('\tDescription:', allWdLists[WdLst]['description'])
            print('\tValid for:', allWdLists[WdLst]['valid_for'])
            print('\tCreation date:', allWdLists[WdLst]['creation_date'])
            print('\tVisibility:', allWdLists[WdLst]['visibility'])
            print('\tNumber of elements:', len(allWdLists[WdLst]['wordlist']))

        return

    def NewWdList(self, listType):
        """
        This method creates a New List of words that can be later used for
        corpus preprocessing

        Parameters
        ----------
        listType : string
            type of list that will be created [keywords|stopwords|equivalences]
        """

        displaytext = """
        *************************************************************************************
        Generation of a new List

            - Stopwords or keywords: Introduce the words separated by commas (stw1,stw2, ...)
            - Equivalences: Introduce equivalences separated by commas in the format
              orig:target (orig1:tgt1, orig2:tgt2, ...)
        *************************************************************************************
        """
        printgr(displaytext)

        # Obtain lists of words for the list
        if listType == 'keywords':
            wds = input('Introduce the keywords: ')
        elif listType == 'stopwords':
            wds = input('Introduce the stopwords: ')
        else:  # equivalences
            wds = input('Introduce the equivalences: ')
        wds = [el.strip() for el in wds.split(',') if len(el)]
        wds = sorted(list(set(wds)))

        # We need a name for the list
        ListName = ""
        while not len(ListName):
            ListName = input('Introduce a name for the new list: ')

        # Introduce a description for the dataset
        ListDesc = ""
        while not len(ListDesc):
            ListDesc = input('Introduce a description: ')

        # Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the list')
        privacy = privacy[opt]

        WdList = {'name': ListName,
                  'description': ListDesc,
                  'valid_for': listType,
                  'visibility': privacy,
                  'wordlist': wds
                  }

        return self.create_List(WdList)

    def EditWdList(self):
        """
        This method allows the edition of an existing list of words, i.e.
        adding new words or removing existing words
        """

        displaytext = """
        *************************************************************************************
        Edition of an existing list

            - Stopwords or keywords: Introduce the words separated by commas (stw1,stw2, ...)
            - Equivalences: Introduce equivalences separated by commas in the format
              orig:target (orig1:tgt1, orig2:tgt2, ...)
        *************************************************************************************
        """
        printgr(displaytext)

        self.logger.info(f'-- -- Modifying an existing list of words')
        # First thing to do is to select a list
        allWdLists = json.loads(self.allWdLists)
        wdLsts = [wlst for wlst in allWdLists.keys()]
        displaywdLsts = [allWdLists[wlst]['name'] + ': ' +
                         allWdLists[wlst]['description'] for wlst in wdLsts]
        selection = query_options(
            displaywdLsts, "Select the list you wish to modify")
        WdLst = allWdLists[wdLsts[selection]]
        self.logger.info(f'-- -- Selected list is {WdLst["name"]}')

        Y_or_N = input(
            f"\nDo you wish to visualize existing words in list [Y/N]?: ")
        if Y_or_N.upper() == "Y":
            print('\n'.join(WdLst['wordlist']))

        wds = input(
            'Introduce the elements you wish to remove (separated by commas): ')
        wds = [el.strip() for el in wds.split(',') if len(el)]
        WdLst['wordlist'] = sorted(list(set([wd for wd in WdLst['wordlist']
                                             if wd not in wds])))

        wds = input(
            'Introduce new elements for the list (separated by commas): ')
        wds = [el.strip() for el in wds.split(',') if len(el)]
        WdLst['wordlist'] = sorted(list(set(wds + WdLst['wordlist'])))

        # The list will be saved replacing existing list
        return self.create_List(WdLst)

    def DelWdList(self):
        """
        Delete a wordlist from wordlist folder
        """

        # Show available wordlists
        self.listAllWdLists()

        allWdLists = json.loads(self.allWdLists)
        for WdLst in allWdLists.keys():
            Y_or_N = input(
                f"\nRemove Word List {allWdLists[WdLst]['name']} [Y/N]?: ")
            if Y_or_N.upper() == "Y":
                if request_confirmation(
                        msg='Word List ' + allWdLists[WdLst]['name'] + ' will be deleted. Proceed?'):
                    self.delete_WdLst(WdLst)

        return

    def listTM(self):
        """
        This method shows all available topic models in the terminal

        This is an extremely simple method for the taskmanager that does not
        require any user interaction

        """

        allTMmodels = json.loads(self.allTMmodels)
        for TMmodel in allTMmodels.keys():
            printmag('\nTraining Dataset ' + allTMmodels[TMmodel]['name'])
            print('\tDescription:', allTMmodels[TMmodel]['description'])
            print('\tTraining Dataset:', allTMmodels[TMmodel]['TrDtSet'])
            print('\tTrainer:', allTMmodels[TMmodel]['trainer'])
            print('\tCreation date:', allTMmodels[TMmodel]['creation_date'])
            print('\tVisibility:', allTMmodels[TMmodel]['visibility'])

        return

    def trainTM(self, trainer):
        """
        Topic modeling trainer. Initial training of a topic model

        Parameters
        ----------
        trainer : string
            Optimizer to use for training the topic model
            Possible values are mallet|sparkLDA|prodLDA|ctm
        """

        ############################################################
        # IMT Interface: Interactive Model Trainer Window
        ############################################################

        self.logger.info(f'-- Topic Model Training')

        displaytext = """
        *************************************************************************************
        We will retrieve all parameters needed for training the topic model
        We start with common settings independent of the method used for the training
        *************************************************************************************
        """
        printgr(displaytext)

        # First thing to do is to select a corpus
        # Ask user which dataset should be used for model training
        allTrDtsets = json.loads(self.allTrDtsets)
        dtSets = [dts for dts in allTrDtsets.keys()]
        displaydtSets = [allTrDtsets[dts]['name'] + ': ' +
                         allTrDtsets[dts]['description'] for dts in dtSets]
        selection = query_options(displaydtSets, "Select Training Dataset")
        TrDtSet = dtSets[selection]
        self.logger.info(
            f'-- -- Selected corpus is {allTrDtsets[TrDtSet]["name"]}')

        displaytext = """
        *************************************************************************************
        We will retrieve all parameters needed for the preprocessing of the lemmas
        This is also needed for all available topic models

        Many of these settings may be for advanced users. We will need to check with the
        users which parameters are basic and which ones should only appear for the advanced
        *************************************************************************************
        """
        printgr(displaytext)

        # Default values are read from config file
        min_lemas = int(self.cf.get('Preproc', 'min_lemas'))
        no_below = int(self.cf.get('Preproc', 'no_below'))
        no_above = float(self.cf.get('Preproc', 'no_above'))
        keep_n = int(self.cf.get('Preproc', 'keep_n'))

        # The following settings will only be accessed in the "advanced settings panel"
        Y_or_N = input(
            f"Do you wish to access the advance settings panel [Y/N]?: ")
        if Y_or_N.upper() == "Y":
            # Some of them can be confirmed/modified by the user
            min_lemas = var_num_keyboard('int', min_lemas,
                                         'Enter minimum number of lemas for the documents in the training set')
            no_below = var_num_keyboard('int', no_below,
                                        'Minimum number occurrences to keep words in vocabulary')
            no_above = var_num_keyboard('float', no_above,
                                        'Maximum proportion of documents to keep a word in vocabulary')
            keep_n = var_num_keyboard('int', keep_n,
                                      'Maximum vocabulary size')

        # Stopword selection
        allWdLists = json.loads(self.allWdLists)
        StwLists = [swl for swl in allWdLists.keys()
                    if allWdLists[swl]['valid_for'] == 'stopwords']
        displayStwLists = [allWdLists[swl]['name'] + ': ' +
                           allWdLists[swl]['description'] for swl in StwLists]
        print('\nAvailable lists of stopwords:')
        print('\n'.join([str(el[0]) + '. ' + el[1]
              for el in enumerate(displayStwLists)]))
        msg = "Select all lists of stopwords that should be used (separated by commas): "
        selection = input(msg)
        if len(selection):
            StwLists = [StwLists[int(el)] for el in selection.split(',')]
        else:
            StwLists = []

        # Lists of equivalences
        EqLists = [eql for eql in allWdLists.keys()
                   if allWdLists[eql]['valid_for'] == 'equivalences']
        displayEqLists = [allWdLists[eql]['name'] + ': ' +
                          allWdLists[eql]['description'] for eql in EqLists]
        print('\nAvailable lists of equivalent terms:')
        print('\n'.join([str(el[0]) + '. ' + el[1]
              for el in enumerate(displayEqLists)]))
        msg = "Select all lists of equivalent terms that should be used (separated by commas): "
        selection = input(msg)
        if len(selection):
            EqLists = [EqLists[int(el)] for el in selection.split(',')]
        else:
            EqLists = []

        Preproc = {
            "min_lemas": min_lemas,
            "no_below": no_below,
            "no_above": no_above,
            "keep_n": keep_n,
            "stopwords": StwLists,
            "equivalences": EqLists
        }

        displaytext = """
        *************************************************************************************
        We will retrieve all parameters needed for the topic modeling itself

        Most of these settings may be for advanced users. We will need to check with the
        users which parameters are basic and which ones should only appear for the advanced
        *************************************************************************************
        """
        printgr(displaytext)

        # We also need the user to select/confirm number of topics
        ntopics = int(self.cf.get('TM', 'ntopics'))
        ntopics = var_num_keyboard('int', ntopics,
                                   'Please, select the number of topics')

        # Retrieve parameters for training. These are dependent on the training algorithm
        if trainer == "mallet":
            # Default values are read from config file
            mallet_path = self.cf.get('MalletTM', 'mallet_path')
            alpha = float(self.cf.get('MalletTM', 'alpha'))
            optimize_interval = int(self.cf.get(
                'MalletTM', 'optimize_interval'))
            num_threads = int(self.cf.get('MalletTM', 'num_threads'))
            num_iterations = int(self.cf.get('MalletTM', 'num_iterations'))
            doc_topic_thr = float(self.cf.get('MalletTM', 'doc_topic_thr'))
            thetas_thr = float(self.cf.get('MalletTM', 'thetas_thr'))
            token_regexp = self.cf.get('MalletTM', 'token_regexp')

            # The following settings will only be accessed in the "advanced settings panel"
            Y_or_N = input(
                f"Do you wish to access the advanced settings panel [Y/N]?:")
            if Y_or_N.upper() == "Y":
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
            LDAparam = {
                "mallet_path": mallet_path,
                "ntopics": ntopics,
                "alpha": alpha,
                "optimize_interval": optimize_interval,
                "num_threads": num_threads,
                "num_iterations": num_iterations,
                "doc_topic_thr": doc_topic_thr,
                "thetas_thr": thetas_thr,
                "token_regexp": token_regexp
            }

        elif trainer == "sparkLDA":
            LDAparam = {}
        elif trainer == "prodLDA":
            LDAparam = {}
        elif trainer == "ctm":
            LDAparam = {}

        displaytext = """
        *************************************************************************************
        We will finally request other general information, modelname, description, etc
        *************************************************************************************
        """
        printgr(displaytext)
        modelname = ''
        while not len(modelname):
            modelname = input('Enter a name to save the new model: ')

        # Introduce a description for the model
        ModelDesc = ""
        while not len(ModelDesc):
            ModelDesc = input('Introduce a description for the model: ')

        # Define privacy level of dataset
        privacy = ['Public', 'Private']
        opt = query_options(privacy, 'Define visibility for the model')
        privacy = privacy[opt]

        # Actual training of the topic model takes place
        super().trainTM(modelname, ModelDesc, privacy, trainer,
                        TrDtSet, Preproc, LDAparam)

        return

    def corpus2JSON(self):
        """
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

    def extractPipe(self, corpus):

        # A proper corpus with BoW, vocabulary, etc .. should exist
        path_corpus = self.p2p.joinpath(
            corpus).joinpath(self._dir_struct['corpus'])
        path_corpus = path_corpus.joinpath(
            corpus).joinpath(corpus + '_corpus.mallet')
        if not path_corpus.is_file():
            self.logger.error(
                '-- Pipe extraction: Could not locate corpus file')
            return

        # Create auxiliary file with only first line from the original corpus file
        path_txt = self.p2p.joinpath(corpus).joinpath(
            self._dir_struct['corpus'])
        path_txt = path_txt.joinpath(corpus).joinpath(corpus + '_corpus.txt')
        with path_txt.open('r', encoding='utf8') as f:
            first_line = f.readline()

        path_aux = self.p2p.joinpath(corpus).joinpath(
            self._dir_struct['corpus'])
        path_aux = path_aux.joinpath(corpus).joinpath('corpus_aux.txt')
        with path_aux.open('w', encoding='utf8') as fout:
            fout.write(first_line + '\n')

        ##################################################
        # We perform the import with the only goal to keep a small
        # file containing the pipe
        self.logger.info('-- Extracting pipeline')
        mallet_path = Path(self.cf.get('TM', 'mallet_path'))
        path_pipe = self.p2p.joinpath(corpus).joinpath(
            self._dir_struct['corpus'])
        path_pipe = path_pipe.joinpath(corpus).joinpath('import.pipe')
        cmd = str(mallet_path) + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_corpus, path_aux, path_pipe)

        try:
            self.logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error('-- Failed to extract pipeline. Revise command')

        # Remove auxiliary file
        path_aux.unlink()

        return

    def inference(self, corpus):

        # A proper corpus should exist with the corresponding ipmortation pipe
        path_pipe = self.p2p.joinpath(corpus).joinpath(
            self._dir_struct['corpus'])
        path_pipe = path_pipe.joinpath(corpus).joinpath('import.pipe')
        if not path_pipe.is_file():
            self.logger.error(
                '-- Inference error. Importation pipeline not found')
            return

        # Ask user which model should be used for inference
        # Final models are enumerated as corpus_givenName
        path_model = self.p2p.joinpath(
            corpus).joinpath(self._dir_struct['modtm'])
        models = sorted([d for d in path_model.iterdir() if d.is_dir()])
        display_models = [' '.join(d.name.split('_')) for d in models]
        selection = query_options(
            display_models, 'Select model for the inference')
        path_model = models[selection]
        inferencer = path_model.joinpath('inferencer.mallet')

        # Ask user to provide a valid text file for performing inference
        # Format of the text will be one document per line, only text
        # Note all created files will be hosted in same directory, so a good idea
        # would be to put the file into an empty directory for this purpose
        while True:
            txt_file = input(
                'Introduce complete path to file with texts for the inference: ')
            txt_file = Path(txt_file)
            if not txt_file.is_file():
                print('Please provide a valid file name')
                continue
            else:
                break

        # The following files will be generated in the same folder
        corpus_file = Path(str(txt_file) + '_corpus.txt')  # lemmatized texts
        corpus_mallet_inf = Path(
            str(txt_file) + '_corpus.mallet')  # mallet serialized
        doc_topics_file = Path(
            str(txt_file) + '_doc-topics.txt')  # Topic proportions
        # Reorder topic proportions in numpy format
        doc_topics_file_npy = Path(str(txt_file) + '_doc-topics.npy')

        # Start processing pipeline

        # ========================
        # 1. Lemmatization
        # ========================
        self.logger.info('-- Inference: Lemmatizing Titles and Abstracts ...')
        lemmas_server = self.cf.get('Lemmatizer', 'server')
        stw_file = Path(self.cf.get('Lemmatizer', 'default_stw_file'))
        dict_eq_file = Path(self.cf.get('Lemmatizer', 'default_dict_eq_file'))
        POS = self.cf.get('Lemmatizer', 'POS')
        concurrent_posts = int(self.cf.get('Lemmatizer', 'concurrent_posts'))
        removenumbers = self.cf.get('Lemmatizer', 'removenumbers') == 'True'
        keepSentence = self.cf.get('Lemmatizer', 'keepSentence') == 'True'

        # Initialize lemmatizer
        ENLM = ENLemmatizer(lemmas_server=lemmas_server, stw_file=stw_file,
                            dict_eq_file=dict_eq_file, POS=POS, removenumbers=removenumbers,
                            keepSentence=keepSentence, logger=self.logger)
        with txt_file.open('r', encoding='utf8') as fin:
            docs = fin.readlines()
        docs = [[el.split()[0], ' '.join(el.split()[1:])] for el in docs]
        docs = [[el[0], clean_utf8(el[1])] for el in docs]
        lemasBatch = ENLM.lemmatizeBatch(docs, processes=concurrent_posts)
        # Remove entries that where not lemmatized correctly
        lemasBatch = [[el[0], clean_utf8(el[1])]
                      for el in lemasBatch if len(el[1])]

        # ========================
        # 2. Tokenization and application of specific stopwords
        #    and equivalences for the corpus
        # ========================
        self.logger.info(
            '-- Inference: Applying corpus specific stopwords and equivalences')
        token_regexp = javare.compile(
            self.cf.get('CorpusGeneration', 'token_regexp'))
        corpus_stw = Path(self.cf.get(corpus, 'stw_file'))
        corpus_eqs = Path(self.cf.get(corpus, 'eq_file'))

        # Initialize Cleaner
        stwEQ = stwEQcleaner(stw_files=[stw_file, corpus_stw], dict_eq_file=corpus_eqs,
                             logger=self.logger)
        # tokenization with regular expression
        id_lemas = [[el[0], ' '.join(token_regexp.findall(el[1]))]
                    for el in lemasBatch]
        # stopwords and equivalences
        id_lemas = [[el[0], stwEQ.cleanstr(el[1])] for el in id_lemas]
        # No need to apply other transformations, because only known words
        # in the vocabulary will be used by Mallet for the topic-inference
        with corpus_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + ' 0 ' + el[1] + '\n') for el in id_lemas]

        # ========================
        # 3. Importing Data to mallet
        # ========================
        self.logger.info('-- Inference: Mallet Data Import')
        mallet_path = Path(self.cf.get('TM', 'mallet_path'))

        cmd = str(mallet_path) + \
            ' import-file --use-pipe-from %s --input %s --output %s'
        cmd = cmd % (path_pipe, corpus_file, corpus_mallet_inf)

        try:
            self.logger.info(f'-- Running command {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self.logger.error(
                '-- Mallet failed to import data. Revise command')
            return

        # ========================
        # 4. Get topic proportions
        # ========================
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

        # ========================
        # 5. Apply model editions
        # ========================
        self.logger.info(
            '-- Inference: Applying model edition transformations')
        # Load thetas file, apply model edition actions, and save as a numpy array
        # We need to read the number of topics, e.g. from train_config file
        train_config = path_model.joinpath('train.config')
        with train_config.open('r', encoding='utf8') as fin:
            num_topics = [el for el in fin.readlines(
            ) if el.startswith('num-topics')][0]
            num_topics = int(num_topics.strip().split(' = ')[1])
        cols = [k for k in np.arange(2, num_topics + 2)]
        thetas32 = np.loadtxt(doc_topics_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)
        model_edits = path_model.joinpath('model_edits.txt')
        if model_edits.is_file():
            with model_edits.open('r', encoding='utf8') as fin:
                for line in fin:
                    line_els = line.strip().split()
                    if line_els[0] == 's':
                        idx = [int(el) for el in line_els[1:]]
                        thetas32 = thetas32[:, idx]
                    elif line_els[0] == 'd':
                        tpc = int(line_els[1])
                        ntopics = thetas32.shape[1]
                        tpc_keep = [k for k in range(ntopics) if k != tpc]
                        thetas32 = thetas32[:, tpc_keep]
                        thetas32 = normalize(thetas32, axis=1, norm='l1')
                    elif line_els[0] == 'f':
                        tpcs = [int(el) for el in line_els[1:]]
                        thet = np.sum(thetas32[:, tpcs], axis=1)
                        thetas32[:, tpcs[0]] = thet
                        thetas32 = np.delete(thetas32, tpcs[1:], 1)

        thetas32 = normalize(thetas32, axis=1, norm='l1')
        np.save(doc_topics_file_npy, thetas32)

        return

    def editTM(self, corpus):

        # Select model for edition
        # Final models are enumerated as corpus_givenName
        path_model = self.p2p.joinpath(
            corpus).joinpath(self._dir_struct['modtm'])
        models = sorted([d for d in path_model.iterdir() if d.is_dir()])
        display_models = [' '.join(d.name.split('_')) for d in models]
        selection = query_options(
            display_models, 'Select the topic model that you want to edit:')
        path_model = models[selection]
        tm = TMmodel(from_file=path_model.joinpath(
            'modelo.npz'), logger=self.logger)

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

                # Y por último seleccionamos las palabras y hacemos la intersección
                wordsOth = []
                for tpc in tpcsOth:
                    wordstpc = tm.most_significant_words_per_topic(
                        n_palabras=10000, tfidf=True, tpc=[tpc])
                    if wordstpc[0][-1][1] / wordstpc[0][0][1] > weighWordsOth:
                        printred(
                            'Se supera el límite preestablecido de palabras para el tópico ' + str(tpc))
                    else:
                        wordstpc = [el[0] for el in wordstpc[0]
                                    if el[1] / wordstpc[0][0][1] > weighWordsOth]
                        wordsOth += wordstpc
                wordsOth = set(wordsOth)

                for tpc in tpcsGbg:
                    wordstpc = tm.most_significant_words_per_topic(
                        n_palabras=10000, tfidf=True, tpc=[tpc])
                    if wordstpc[0][-1][1] / wordstpc[0][0][1] > weighWordsGbg:
                        printred(
                            'Se supera el límite preestablecido de palabras para el tópico ' + str(tpc))
                    else:
                        wordstpc = [el[0] for el in wordstpc[0]
                                    if el[1] / wordstpc[0][0][1] > weighWordsGbg]
                        printgr(40 * '=')
                        printgr('Tópico ' + str(tpc))
                        printgr('Seleccionadas ' +
                                str(len(wordstpc)) + ' palabras')
                        printgr(40 * '=')
                        stwCandidates = [
                            el for el in wordstpc if el not in wordsOth]
                        printmag('Candidatas a StopWord (' +
                                 str(len(stwCandidates)) + '):')
                        print(stwCandidates)
                        nonStwCandidates = [
                            el for el in wordstpc if el in wordsOth]
                        printmag('Coincidentes con otros tópicos (' +
                                 str(len(nonStwCandidates)) + '):')
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
                    r = input('\nIntroduce the description for the topic, write "wd" use Word Description,\n' +
                              'or press enter to keep current:\n')
                    if r == 'wd':
                        tm.set_description(worddesc, tpc)
                    elif r != '':
                        tm.set_description(r, tpc)
                modified = True

            elif selection == 'Eliminar un tópico del modelo':
                lista = [(ndoc, desc) for ndoc, desc in zip(
                    tm.ndocs_active_topic(), tm.get_topic_word_descriptions())]
                for k in sorted(lista, key=lambda x: -x[0]):
                    print('=' * 5)
                    perc = '(' + \
                        str(round(100 * float(k[0]) / corpus_size)) + ' %)'
                    print('ID del tópico:', k[1][0])
                    print('Número de documentos en los que está activo:',
                          k[0], perc)
                    print('Palabras más significativas:', k[1][1])
                msg = '\nIntroduce el ID de los tópicos que deseas eliminar separados por comas'
                msg += '\no presiona ENTER si no deseas eliminar ningún tópico\n'
                r = input(msg)
                try:
                    tpcs = [int(n) for n in r.split(',')]
                    # Eliminaremos los tópicos en orden decreciente
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
                    msg = 'Correlación de los tópicos {0:d} y {1:d}: {2:.2f}%'.format(
                        pair[0], pair[1], 100 * pair[2])
                    printmag(msg)
                    printmag(20 * '=')
                    tm.muestra_perfiles(n_palabras=nwords,
                                        tpc=[pair[0], pair[1]])
                printred(20 * '=')
                printred(
                    'Cuidado: los ids de los tópicos cambian tras la fusión o eliminación')
                printred(20 * '=')

            elif selection == 'Tópicos similares por palabras':
                msg = '\nIntroduzca umbral para selección de palabras'
                thr = var_num_keyboard('float', 1e-3, msg)
                msg = '\nIntroduzca el número de pares de tópicos a mostrar'
                npairs = var_num_keyboard('int', 5, msg)
                msg = '\nIntroduzca número de palabras para mostrar'
                nwords = var_num_keyboard('int', 10, msg)
                selected = tm.get_similar_JSdist(npairs, thr)
                for pair in selected:
                    msg = 'Similitud de los tópicos {0:d} y {1:d}: {2:.2f}%'.format(
                        pair[0], pair[1], 100 * pair[2])
                    printmag(msg)
                    printmag(20 * '=')
                    tm.muestra_perfiles(n_palabras=nwords,
                                        tpc=[pair[0], pair[1]])
                printred(20 * '=')
                printred(
                    'Cuidado: los ids de los tópicos cambian tras la fusión o eliminación')
                printred(20 * '=')

            elif selection == 'Fusionar dos tópicos del modelo':
                tm.muestra_descriptions()
                msg = '\nIntroduce el ID de los tópicos que deseas fusionar separados por comas'
                msg += '\no presiona ENTER si no deseas fusionar ningún tópico\n'
                r = input(msg)
                try:
                    tpcs = [int(n) for n in r.split(',')]
                    if len(tpcs) >= 2:
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
        `self.validate_topic_models()`

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
        `self.validate_topic_models()`

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


##############################################################################
#                          ITMTTaskManagerGUI                                #
##############################################################################
class ITMTTaskManagerGUI(ITMTTaskManager):
    """
    Provides extra functionality to the task manager, to be used by the
    Graphical User Interface (GUI)
    """

    def __init__(self, p2p, p2parquet, p2wdlist, config_fname='config.cf',
                 metadata_fname='metadata.yaml'):
        """
        Initializes an ITMTTaskManagerGUI object.

        Parameters
        ----------
        p2p : pathlib.Path
            Path to the application project
        p2parquet : pathlib.Path
            Path to the folder hosting the parquet datasets
        p2wdlist : pathlib.Path
            Path to the folder hosting the wordlists (stopwords, keywords, etc)
        config_fname : str, optional (default='config.cf')
            Name of the configuration file
        metadata_fname : str or None, optional (default=metadata.yaml)
            Name of the project metadata file.
            If None, no metadata file is used.
        """

        super().__init__(
            p2p, p2parquet, p2wdlist, config_fname=config_fname, metadata_fname=metadata_fname)

    def listDownloaded(self, gui):
        """
        This method gets all Datasets that have been retrieved from HDFS and are available for the Model Trainer and
        displays them in the corresponding table within the GUI.

        Parameters
        ----------
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allDtsets:
            allDtsets = json.loads(self.allDtsets)
            table = gui.table_available_local_corpus
            table.setRowCount(len(allDtsets.keys()))
            row = 0
            for Dts in allDtsets.keys():
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(
                    allDtsets[Dts]['name']))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                    allDtsets[Dts]['source']))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(
                    allDtsets[Dts]['description']))
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(
                    ', '.join([el for el in allDtsets[Dts]['schema']])))
                table.setItem(row, 4, QtWidgets.QTableWidgetItem(
                    str(allDtsets[Dts]['records'])))
                table.setItem(row, 5, QtWidgets.QTableWidgetItem(
                    allDtsets[Dts]['download_date']))
                table.setItem(row, 6, QtWidgets.QTableWidgetItem(
                    allDtsets[Dts]['visibility']))
                row += 1

        return

    def createTMCorpus(self, dict_to_tm_corpus, dtsName, dtsDesc, privacy):

        # We need first to get all available (downloaded) datasets
        allDtsets = json.loads(self.allDtsets)

        TM_Dtset = []
        list_ids_dts = [int(el) for el in dict_to_tm_corpus.keys()]
        for corpus_id in list_ids_dts:
            Dtsets = [el for el in allDtsets.keys()]
            Dtset_loc = Dtsets.pop(corpus_id)
            Dtset_source = allDtsets[Dtset_loc]['source']
            print('\nProcessing dataset', allDtsets[Dtset_loc]['name'])

            # id fld
            dict_tm_corpus_loc = list_ids_dts.pop(corpus_id)
            Dtset_idfld = dict_to_tm_corpus[dict_tm_corpus_loc]['identifier_field']
            print("Field to use as identifier ", str(Dtset_idfld))

            # lemmas fields
            Dtset_lemmas_fld = dict_to_tm_corpus[dict_tm_corpus_loc]['fields_for_lemmas']
            print('Selected lemmas:', ', '.join(Dtset_lemmas_fld))

            # rawtext fields
            Dtset_rawtext_fld = dict_to_tm_corpus[dict_tm_corpus_loc]['fields_for_raw']

            # Spark clause for filtering (advanced users only)
            Dtset_filter = dict_to_tm_corpus[dict_tm_corpus_loc]['filtering_condition']

            TM_Dtset.append({'parquet': Dtset_loc,
                             'source': Dtset_source,
                             'idfld': Dtset_idfld,
                             'lemmasfld': Dtset_lemmas_fld,
                             'rawtxtfld': Dtset_rawtext_fld,
                             'filter': Dtset_filter
                             })

        Dtset = {'name': dtsName,
                 'description': dtsDesc,
                 'valid_for': "TM",
                 'visibility': privacy,
                 'Dtsets': TM_Dtset
                 }

        status = self.save_TrDtset(Dtset)

        return int(status.decode('utf8'))

    def listTMCorpus(self, gui):
        """
        This method shows all (logical) Datasets available for training Topic Models in the corresponding table
        within the GUI.

        Parameters
        ----------
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allTrDtsets:
            allTrDtsets = json.loads(self.allTrDtsets)
            table = gui.table_available_training_datasets
            table.setRowCount(len(allTrDtsets.keys()))
            row = 0
            for TrDts in allTrDtsets.keys():
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(
                    allTrDtsets[TrDts]['name']))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                    allTrDtsets[TrDts]['description']))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(
                    allTrDtsets[TrDts]['valid_for']))
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(
                    allTrDtsets[TrDts]['creation_date']))
                table.setItem(row, 4, QtWidgets.QTableWidgetItem(
                    allTrDtsets[TrDts]['visibility']))
                row += 1

        return

    def deleteTMCorpus(self, corpus_to_delete, gui):
        """
        Deletes the training corpus given by 'corpus_to_delete' from the Interactive Topic Model Trainer dataset folder.

        Parameters
        ----------
        corpus_to_delete : str
            Name of the corpus selected by the user in the GUI to be deleted
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allTrDtsets:
            allTrDtsets = json.loads(self.allTrDtsets)
            for TrDts in allTrDtsets.keys():
                if allTrDtsets[TrDts]['name'] == corpus_to_delete:
                    reply = QMessageBox.question(gui, Constants.SMOOTH_SPOON_MSG, 'Training Dataset ' +
                                                 allTrDtsets[TrDts]['name'] + ' will be deleted. Proceed?',  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        status = self.delete_TrDtset(TrDts)
                        # @TODO: Revise. Status is being returned as a byte object (b'1')
                        if int(status.decode('utf8')) == 0:
                            QMessageBox.warning(gui, Constants.SMOOTH_SPOON_MSG, 'Training Dataset ' +
                                                allTrDtsets[TrDts]['name'] + ' could not be deleted.')
                        elif int(status.decode('utf8')) == 1:
                            QMessageBox.information(
                                gui, Constants.SMOOTH_SPOON_MSG, 'Training Dataset ' + allTrDtsets[TrDts]['name'] + ' was deleted successfully.')

        return

    def listAllWdLists(self, gui):
        """
        This method shows all wordlists available for the project in the corresponding table
        within the GUI.

        Parameters
        ----------
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allWdLists:
            allWdLists = json.loads(self.allWdLists)
            table = gui.table_available_wordlists
            table.setRowCount(len(allWdLists.keys()))
            row = 0
            for TrDts in allWdLists.keys():
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(
                    allWdLists[TrDts]['name']))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                    allWdLists[TrDts]['description']))
                table.setItem(row, 2, QtWidgets.QTableWidgetItem(
                    allWdLists[TrDts]['valid_for']))
                table.setItem(row, 3, QtWidgets.QTableWidgetItem(
                    allWdLists[TrDts]['creation_date']))
                table.setItem(row, 4, QtWidgets.QTableWidgetItem(
                    allWdLists[TrDts]['visibility']))
                table.setItem(row, 5,
                              QtWidgets.QTableWidgetItem(', '.join([el for el in allWdLists[TrDts]['wordlist']])))
                row += 1

        return

    def listWdListsByType(self, table, type):
        """
        This method shows the wordlists of type "type" available for the project in the corresponding table within the GUI.

        Parameters
        ----------
        table : QTableWidget
            GUI's table in which the wordlists are going to be displayed
        type: str
            Type of the lists ('stopwords', 'equivalences')
        """

        if self.allWdLists:
            allWdLists = json.loads(self.allWdLists)
            typeWdLists = [WdList for WdList in allWdLists.keys(
            ) if allWdLists[WdList]['valid_for'] == type]
            table.setRowCount(len(typeWdLists))
            row = 0
            for WdList in allWdLists.keys():
                if allWdLists[WdList]['valid_for'] == type:
                    table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                        allWdLists[WdList]['name']))
                    table.setItem(row, 2, QtWidgets.QTableWidgetItem(
                        allWdLists[WdList]['description']))
                    row += 1
        return

    def NewWdList(self, listType, wds, lst_name, lst_privacy, lst_desc):
        """
        This method creates a new List of words that can be later used for
        corpus preprocessing. 

        Parameters
        ----------
        listType : string
            type of list that will be created [keywords|stopwords|equivalences]
        wds : string
            List of words in the format required by the listType with which the new wordlist will be conformed
        lst_privacy : string
            String describing the new wordlist's privacy level [private|public]
        lst_desc : string
            String with the new wordlist's description
        """

        WdList = {'name': lst_name,
                  'description': lst_desc,
                  'valid_for': listType,
                  'visibility': lst_privacy,
                  'wordlist': wds
                  }
        status = self.create_List(WdList)

        return int(status.decode('utf8'))

    def EditWdList(self, wdlist):
        """
        This method allows the edition of an existing list of words, i.e.
        adding new words or removing existing words

        Parameters
        ----------
        wds : dict
            Dictionary describing the edited wordlist
        """

        # The list will be saved replacing existing list
        status = self.create_List(wdlist)

        return int(status.decode('utf8'))

    def DelWdList(self, wdlst_to_delete, gui):
        """
        Deletes a wordlist from wordlist folder

        Parameters
        ----------
        wdlst_to_delete : str
            Name of the wordlist selected by the user in the GUI to be deleted
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allWdLists:
            allWdLists = json.loads(self.allWdLists)
            for WdLst in allWdLists.keys():
                if allWdLists[WdLst]['name'] == wdlst_to_delete:
                    reply = QMessageBox.question(gui, Constants.SMOOTH_SPOON_MSG, 'Wordlist ' +
                                                 allWdLists[WdLst]['name'] +
                                                 ' will be deleted. Proceed?',
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                 QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        status = self.delete_WdLst(WdLst)
                        # @TODO: Revise. Status is being returned as a byte object (b'1')
                        if int(status.decode('utf8')) == 0:
                            QMessageBox.warning(gui, Constants.SMOOTH_SPOON_MSG, 'Wordlist ' +
                                                allWdLists[WdLst]['name'] + ' could not be deleted.')
                        elif int(status.decode('utf8')) == 1:
                            QMessageBox.information(gui, Constants.SMOOTH_SPOON_MSG, 'Wordlist ' +
                                                    allWdLists[WdLst]['name'] + ' was deleted successfully.')

        return

    def get_wdlist_info(self, wlst_to_edit):
        """
        Deletes a wordlist from wordlist folder

        Parameters
        ----------
        wlst_to_edit : str
            Name of the wordlist selected by the user in the GUI to be edited

        Returns
        -------
        wdList_info : dict
            Dictionary with the wordlist information
        """

        wdList_info = {}
        if self.allWdLists:
            allWdLists = json.loads(self.allWdLists)
            for WdLst in allWdLists.keys():
                if allWdLists[WdLst]['name'] == wlst_to_edit:
                    wdList_info = {'name': wlst_to_edit,
                                   'description': allWdLists[WdLst]['description'],
                                   'valid_for': allWdLists[WdLst]['valid_for'],
                                   'visibility': allWdLists[WdLst]['visibility'],
                                   'wordlist': allWdLists[WdLst]['wordlist']
                                   }
        return wdList_info

    def trainTM(self, trainer, TrDts_name, preproc_settings, training_params, modelname, ModelDesc, privacy):
        """
        Topic modeling trainer. Initial training of a topic model

        Parameters
        ----------
        trainer : string
            Optimizer to use for training the topic model
            Possible values are mallet|sparkLDA|prodLDA|ctm
        """

        # First thing to do is to select a corpus
        if self.allTrDtsets:
            allTrDtsets = json.loads(self.allTrDtsets)
            for TrDts in allTrDtsets.keys():
                if allTrDtsets[TrDts]['name'] == TrDts_name:
                    TrDtSet = TrDts
        self.logger.info(
            f'-- -- Selected corpus is {allTrDtsets[TrDtSet]["name"]}')

        # Actual training of the topic model takes place
        super().trainTM(modelname, ModelDesc, privacy, trainer,
                        TrDtSet, preproc_settings, training_params)

        return
    
    def load_listTMmodels(self):
        """
        Extends the load_listTMmodels method from the parent class to load into execution time an XML structure of all the available TM models that are going to be used for visualization purposes in the GUI.
        """

        super().load_listTMmodels()

        if self.allTMmodels:
            all_models = self.p2p.joinpath(
                self._dir_struct['LDAmodels']).resolve().as_posix() # @TODO: Change LDAmodels to TMmodels
            
            # Create XML structure of the models for visaulization purposes
            if pathlib.Path(all_models).is_dir():
                self.models_xml = get_model_xml(all_models)

        return
    

    def listAllTMmodels(self, gui):
        """
        This method shows all topic models available for the project in the corresponding tree view.

        Parameters
        ----------
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        """

        if self.allTMmodels:
            if self.models_xml:
                clearQTreeWidget(gui.treeView_trained_models)
                printTree(self.models_xml, gui.treeView_trained_models)

        return
    
    def listTMmodel(self, gui, model_name):
        """
        This method show the description in the table 'table_available_trained_models_desc' of the topic selected in the 'treeView_trained_models'.

        Parameters
        ----------
        gui : src.gui.main_window.MainWindow
            QMainWindow object associated which the GUI
        model_name: str
            Name of the topic model whose information is going to be displayed
        """        

        if self.allTMmodels:

            # Get dictionary with the information of all models
            allTMmodels = json.loads(self.allTMmodels)

            # Get table where TMmodel information is going to be displayed
            table = gui.table_available_trained_models_desc
            table.setRowCount(1)

            for TMmodel in allTMmodels.keys():
                if allTMmodels[TMmodel]['name'] == model_name:
                    table.setItem(0, 0, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['name']))
                    table.setItem(0, 1, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['description']))
                    table.setItem(0, 2, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['visibility']))
                    table.setItem(0, 3, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['trainer']))
                    table.setItem(0, 4, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['TrDtSet']))
                    table.setItem(0, 5, QtWidgets.QTableWidgetItem(
                        ""))
                    table.setItem(0, 6, QtWidgets.QTableWidgetItem(
                        ""))
                    table.setItem(0, 7, QtWidgets.QTableWidgetItem(
                        allTMmodels[TMmodel]['creation_date']))
            
        return
                