"""
* *IntelComp H2020 project*

Base Task Manager for the Interactive Topic Model Trainer App
It implements the functions needed to create, load and set up an
execution project from the main application.
"""

import configparser
import logging
from pathlib import Path
import shutil
import yaml


class BaseTaskManager(object):
    """
    Main class to manage functionality of the Topic Model Interactive Trainer

    The behavior of this class depends on the state of the project, which is stored 
    in the dictionary self.state, characterized by the following entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file successfully loaded. Datamanager
                      activated.
    """

    def __init__(self, p2p, p2parquet, p2wdlist, config_fname='config.cf',
                 metadata_fname='metadata.yaml'):
        """
        Sets the main attributes to manage tasks over a specific application
        project.

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

        # Important directories for the project
        self.p2p = p2p
        self.p2parquet = p2parquet
        self.p2wdlist = p2wdlist

        # Configuration file
        self.p2config = self.p2p / config_fname
        self.cf = None  # Handler to the config file

        # Metadata file
        self.path2metadata = self.p2p / metadata_fname
        # Metadata attributes
        self.metadata_fname = metadata_fname

        # These are the default file and folder names for the folder
        # structure of the project. It can be modified by entering other
        # names as arguments of the create or the load method.
        self._dir_struct = {}

        # State variables that will be loaded from the metadata file
        # when the project was loaded.
        self.state = {
            'isProject': False,  # True if the project exist.
            'configReady': False}  # True if config file could be loaded

        # The default metadata dictionary only contains the state dictionary.
        self.metadata = {'state': self.state}

        # Logger object (that will be activated by _set_logs() method)
        self.logformat = None
        self.logger = None

        # Other class variables
        self.ready2setup = False  # True after create() or load() are called
        print('-- Task Manager object successfully initialized')

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

    def _update_folders(self, _dir_struct=None):
        """
        Creates or updates the project folder structure using the file and
        folder names in _dir_struct.
        Parameters
        ----------
        _dir_struct: dict or None, optional (default=None)
            Contains all information related to the structure of project files
            and folders:
                - paths (relative to the project path in self.path2projetc)
                - file names
                - suffixes, prefixes or extensions that could be used to define
                  other files or folders.
            If None, names are taken from the current self._dir_struct attribute
        """

        # ######################
        # Project file structure

        # Overwrite default names in self._dir_struct dictionary by those
        # specified in _dir_struct
        if _dir_struct is not None:
            self._dir_struct.update(_dir_struct)

        # In the following, we assume that all files in self._dir_struct are
        # subfolders of self.p2p. If this is not the case, this method
        # should be modified by a child class
        for d in self._dir_struct:
            path2d = self.p2p / self._dir_struct[d]
            if not path2d.exists():
                path2d.mkdir()

        return

    def _save_metadata(self):
        """
        Save metadata into a pickle file
        """

        # Save metadata
        with open(self.path2metadata, 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False)

        return

    def _load_metadata(self):
        """
        Loads metadata file

        Returns
        -------
        metadata : dict
            Metadata dictionary
        """

        # Save metadata
        print('-- Loading metadata file...')
        with open(self.path2metadata, 'r', encoding='utf8') as f:
            metadata = yaml.safe_load(f)

        return metadata

    def create(self):
        """
        Creates a project instance for the Topic Model Trainer
        To do so, it defines the main folder structure, and creates (or cleans)
        the project folder, specified in self.p2p

        """

        print("\n*** CREATING NEW PROJECT")

        # #####################
        # Create project folder

        # Check and clean project folder location
        if self.p2p.exists():

            # Remove current backup folder, if it exists
            old_p2p = Path(str(self.p2p) + '_old')
            if old_p2p.exists():
                shutil.rmtree(old_p2p)

            # Copy current project folder to the backup folder.
            shutil.move(self.p2p, old_p2p)
            print(f'-- -- Existing project with same name moved to {old_p2p}')

        # Create project folder
        self.p2p.mkdir()

        # ########################
        # Add files and subfolders

        # Subfolders
        self._update_folders(None)

        # Place a copy of a default configuration file in the project folder.
        shutil.copyfile('config.cf.default', self.p2config)

        # #####################
        # Update project status

        # Update the state of the project.
        self.state['isProject'] = True
        self.metadata.update({'state': self.state})

        # Save metadata
        self._save_metadata()

        # The project is ready to setup, but the user should edit the
        # configuration file first
        self.ready2setup = True

        print(f"-- Project {self.p2p} created.")
        print("---- Project metadata saved in {0}".format(self.metadata_fname))
        print("---- A default config file has been located in the project "
              "folder.")
        print("---- Open it and set your configuration variables properly.")
        print("---- Once the config file is ready, activate it.")

        self.setup()

        return

    def load(self):
        """
        Loads an existing Interactive Topic Modeling Trainer project, by reading the metadata file in the project
        folder.
        It can be used to modify file or folder names, or paths, by specifying
        the new names/paths in the _dir_struct dictionary.
        """

        # ########################
        # Load an existing project
        print("\n*** LOADING PROJECT")

        # Check and clean project folder location
        if not self.path2metadata.exists():
            exit(f'-- ERROR: Metadata file {self.path2metadata} does not'
                 '   exist.\n'
                 '   This is likely not a project folder. Select another '
                 'project or create a new one.')

        else:
            # Load project metadata
            self.metadata = self._load_metadata()

            # Store state
            self.state = self.metadata['state']

            # The following is used to automatically update any changes in the
            # keys of the self._dir_struct dictionary. This will be likely
            # unnecesary once a stable version of the code is reached, but it
            # is useful to update older application projects.
            self._update_folders(self._dir_struct)

            if self.state['configReady']:
                self.ready2setup = True
                self.setup()
                print(f'-- Project {self.p2p} succesfully loaded.')
            else:
                exit(f'-- WARNING: Project {self.p2p} loaded, but '
                     'configuration file could not be activated. You can: \n'
                     '(1) revise and reactivate the configuration file, or\n'
                     '(2) delete the project folder to restart')

        return

    def setup(self):
        """
        Sets up the project. To do so:
            - Loads the configuration file
            - Activates the logger objects
        """

        # #################################################
        # Activate configuration file and load data Manager
        print("\n*** ACTIVATING CONFIGURATION FILE")

        if self.ready2setup is False:
            exit("---- Error: you cannot setup a project that has not been "
                 "created or loaded")

        # Loads configuration file
        self.cf = configparser.ConfigParser()
        self.cf.optionxform = str  # Preserves case of keys in config file
        self.cf.read(self.p2config)
        self.state['configReady'] = True

        # Set up the logging format
        self._set_logs()

        # Save the state of the project.
        self._save_metadata()

        self.logger.info('Project setup finished')

        return
