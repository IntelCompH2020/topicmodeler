"""
* *IntelComp H2020 project*

Class that defines the Graphical user interface's main window for the Interactive Topic Model Trainer App
It implements the functions needed to

    - Configure the GUI widgets defined in the corresponding UI file (main_window.ui)
    - Connect the GUI and the project manager (ITMTTaskManager)
    - Implement handler functions to control the interaction between the user and the 
      different widgets
    - Control the opening of secondary windows for specific functionalities (training, curation, etc.)
"""

import configparser
import os
import pathlib
from functools import partial

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThreadPool, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QPushButton
from PyQt6.uic import loadUi
from src.gui.manage_corpus.generate_tm_corpus_window import GenerateTMCorpus
from src.gui.manage_lists.create_sw_lst_window import CreateSwLstWindow
from src.gui.manage_lists.edit_sw_lst_window import EditSwLstWindow
from src.gui.topic_modeling.preprocessing_window import PreprocessingWindow
from src.gui.topic_modeling.train_model_window import TrainModelWindow
from src.gui.utils import utils
from src.gui.utils.constants import Constants
from src.gui.utils.output_wrapper import OutputWrapper
from src.project_manager.itmt_task_manager import ITMTTaskManagerGUI


class MainWindow(QMainWindow):
    def __init__(self, project_folder=None, parquet_folder=None, wordlists_folder=None):
        """
        Initializes the application's main window.

        Parameters
        ----------
        project_folder : pathlib.Path (default=None)
           Path to the application project
        parquet_folder : pathlib.Path (default=None)
           Path to the folder containing the parquet files
        wordlists_folder : pathlib.Path (default=None)
           Path to the folder hosting the wordlists (stopwords, keywords, etc)
        """

        super(MainWindow, self).__init__()

        ########################################################################
        # Load UI
        ########################################################################
        loadUi("src/gui/uis/main_window.ui", self)
        self.setWindowIcon(QtGui.QIcon(
            'src/gui/resources/images/fuzzy_training.png'))
        self.setWindowTitle(Constants.SMOOTH_SPOON_TITLE)
        # Initally, the menu is hidden
        self.frame_menu_title.hide()

        ########################################################################
        # Attributes
        ########################################################################
        # Attributes to redirect stdout and stderr
        self.stdout = OutputWrapper(self, True)
        self.stderr = OutputWrapper(self, False)

        # Attributes for creating TM object
        self.project_folder = project_folder
        self.parquet_folder = parquet_folder
        self.wordlists_folder = wordlists_folder
        self.tm = None
        if self.project_folder and self.parquet_folder and self.wordlists_folder:
            self.configure_tm()

        # Attributes for displaying PyLDAvis in home page
        self.web = None

        # Other attributes
        self.previous_page_button = self.findChild(
            QPushButton, "menu_button_1")
        self.previous_corpus_button = self.findChild(
            QPushButton, "corpus_button_1")
        self.previous_settings_button = self.findChild(
            QPushButton, "settings_button_1")
        self.previous_tm_settings_button = self.findChild(
            QPushButton, "settings_button_5_1")

        # Get home in any operating system
        self.home = str(pathlib.Path.home())

        # Creation of subwindows
        self.train_model_subwindow = None
        self.create_tm_corpus_subwindow = None
        self.create_stopwords_list_subwindow = None
        self.edit_stopwords_list_subwindow = None
        self.preprocessing_subwindow = None

        # Threads for executing in parallel
        self.thread_pool = QThreadPool()
        print("Multithreading with maximum"
              " %d threads" % self.thread_pool.maxThreadCount())

        ########################################################################
        # Connect pages
        ########################################################################
        self.menu_buttons = []
        for id_button in np.arange(Constants.MAX_MENU_BUTTONS):
            menu_button_name = "menu_button_" + str(id_button + 1)
            menu_button_widget = self.findChild(QPushButton, menu_button_name)
            self.menu_buttons.append(menu_button_widget)
        self.extra_menu_buttons = [self.pushButton_corpus_home,
                                   self.pushButton_wordlists_home, 
                                   self.pushButton_models_home]

        for menu_button in self.menu_buttons:
            menu_button.clicked.connect(
                partial(self.clicked_change_menu_button, menu_button))

        # PAGE 1: Home
        ###############
        self.menu_button_1.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_home))

        self.home_recent_buttons = []
        for id_button in np.arange(Constants.MAX_RECENT_PROJECTS):
            home_recent_button_name = "pushButton_recent_project_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_PARQUETS):
            home_recent_button_name = "pushButton_recent_parquet_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)
        for id_button in np.arange(Constants.MAX_RECENT_WORDLISTS):
            home_recent_button_name = "pushButton_recent_wordlists_folder_" + \
                str(id_button + 1)
            home_recent_button_widget = self.findChild(
                QPushButton, home_recent_button_name)
            self.home_recent_buttons.append(home_recent_button_widget)

        for home_recent_button in self.home_recent_buttons:
            home_recent_button.clicked.connect(
                partial(self.get_folder_from_recent, home_recent_button))

        # PAGE 2: Corpus
        #################
        self.menu_button_2.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_corpus))
        self.pushButton_corpus_home.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_corpus))

        self.corpus_buttons = []
        for id_button in np.arange(Constants.MAX_CORPUS_BUTTONS):
            corpus_button_name = "corpus_button_" + str(id_button + 1)
            corpus_button_widget = self.findChild(
                QPushButton, corpus_button_name)
            self.corpus_buttons.append(corpus_button_widget)

        for corpus_button in self.corpus_buttons:
            corpus_button.clicked.connect(
                partial(self.clicked_change_corpus_button, corpus_button))

        self.corpus_button_1.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_local_corpus))
        self.corpus_button_2.clicked.connect(
            lambda: self.corpus_tabs.setCurrentWidget(self.page_training_datasets))

        # PAGE 3: Wordlists
        #####################
        self.menu_button_3.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_wordlists))
        self.pushButton_wordlists_home.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_wordlists))

        # PAGE 4: Models
        #################
        self.menu_button_4.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_models))
        self.pushButton_models_home.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_models))

        # PAGE 5: Settings
        ###################
        self.menu_button_5.clicked.connect(
            lambda: self.content_tabs.setCurrentWidget(self.page_general_settings))

        self.settings_buttons = []
        for id_button in np.arange(Constants.MAX_SETTINGS_BUTTONS):
            settings_button_name = "settings_button_" + str(id_button + 1)
            settings_button_widget = self.findChild(
                QPushButton, settings_button_name)
            self.settings_buttons.append(settings_button_widget)

        for settings_button in self.settings_buttons:
            settings_button.clicked.connect(
                partial(self.clicked_change_settings_button, settings_button))

        self.tm_settings_buttons = []
        for id_button in np.arange(Constants.MAX_TM_SETTINGS_SUBBUTTONS):
            tm_settings_button_name = "settings_button_5_" + str(id_button + 1)
            tm_settings_button_widget = self.findChild(
                QPushButton, tm_settings_button_name)
            self.tm_settings_buttons.append(tm_settings_button_widget)

        for tm_settings_button in self.tm_settings_buttons:
            tm_settings_button.clicked.connect(
                partial(self.clicked_change_tm_settings_button, tm_settings_button))
            tm_settings_button.hide()

        self.settings_button_1.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_gui_settings))
        self.settings_button_2.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_log_format))
        self.settings_button_3.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_spark))
        self.settings_button_4.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_preprocessing))
        self.settings_button_5.clicked.connect(
            self.select_tm_settings_suboption)
        self.settings_button_5_1.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_general))
        self.settings_button_5_2.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_mallet))
        self.settings_button_5_3.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_prod))
        self.settings_button_5_4.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_CTM))
        self.settings_button_5_5.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_spark_lda))
        self.settings_button_5_6.clicked.connect(
            lambda: self.settings_tabs.setCurrentWidget(self.page_tm_hierarchical))

        ########################################################################
        # Widgets initial configuration
        ########################################################################
        # MENU BUTTONS
        # When the app is first opened, menu buttons are disabled until the user selects properly the project and parquet folders
        for menu_button in self.menu_buttons:
            menu_button.setEnabled(False)
        for extra_menu_button in self.extra_menu_buttons:
            extra_menu_button.setEnabled(False)

        # PAGE 1: Home
        utils.set_recent_buttons(self)
        self.previous_page_button.setStyleSheet(
            Constants.HOME_BUTTON_SELECTED_STYLESHEET)

        # PAGE 2: Corpus
        # Configure tables
        utils.configure_table_header(Constants.CORPUS_TABLES, self)
        self.previous_corpus_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        # PAGE 3: Wordlists
        # Configure tables
        utils.configure_table_header(Constants.WORDLISTS_TABLES, self)

        # PAGE 4: Models
        # Configure tables
        utils.configure_table_header(Constants.MODELS_TABLES, self)

        # PAGE 5: Settings

        ########################################################################
        # Connect buttons
        ########################################################################
        self.pushButton_open_project_folder.clicked.connect(
            self.get_project_folder)
        self.pushButton_open_parquet_folder.clicked.connect(
            self.get_parquet_folder)
        self.pushButton_open_wordlists_folder.clicked.connect(
            self.get_wordlists_folder)

        self.pushButton_generate_training_dataset.clicked.connect(
            self.clicked_pushButton_generate_training_dataset)
        self.pushButton_train_trdtst.clicked.connect(
            self.clicked_train_dataset)
        self.pushButton_delete_trdtst.clicked.connect(
            self.clicked_delete_dataset)

        self.pushButton_create_wordlist.clicked.connect(
            self.clicked_pushButton_create_wordlist)
        self.pushButton_edit_wordlist.clicked.connect(
            self.clicked_pushButton_edit_wordlist)
        self.pushButton_delete_wordlist.clicked.connect(
            self.clicked_pushButton_delete_wordlist)
        self.table_available_wordlists.cellClicked.connect(
            self.show_wordlist_description)

        self.treeView_trained_models.clicked.connect(
            self.clicked_treeView_trained_models)
        self.pushButton_models_pyldavis.clicked.connect(
            self.clicked_pushButton_models_pyldavis)
        self.pushButton_return_pyldavis.clicked.connect(
            self.clicked_pushButton_return_pyldavis)
        self.pushButton_train_submodel.clicked.connect(
            self.clicked_pushButton_train_submodel)
        self.pushButton_delete_model.clicked.connect(
            self.clicked_pushButton_delete_model)

        self.pushButton_apply_changes_gui_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_gui_settings)
        self.pushButton_apply_changes_log_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_log_settings)
        self.pushButton_apply_changes_mallet_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_mallet_settings)
        self.pushButton_apply_changes_prodlda_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_prodlda_settings)
        self.pushButton_apply_changes_sparklda_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_sparklda_settings)
        self.pushButton_apply_changes_ctm_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_ctm_settings)
        self.pushButton_apply_changes_hierarchical_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_hierarchical_settings)
        self.pushButton_apply_changes_tm_general_settings.clicked.connect(
            self.clicked_pushButton_apply_changes_tm_general_settings)
        self.pushButton_apply_spark_settings.clicked.connect(
            self.clicked_pushButton_apply_spark_settings)
        self.pushButton_apply_preprocessing_settings.clicked.connect(
            self.clicked_pushButton_apply_preprocessing_settings)

        self.pushButton_restore_gui_settings.clicked.connect(
            self.clicked_pushButton_restore_gui_settings)
        self.pushButton_restore_log_settings.clicked.connect(
            self.clicked_pushButton_restore_log_settings)
        self.pushButton_restore_mallet_settings.clicked.connect(
            self.clicked_pushButton_restore_mallet_settings)
        self.pushButton_restore_prodlda_settings.clicked.connect(
            self.clicked_pushButton_restore_prodlda_settings)
        self.pushButton_restore_sparklda_settings.clicked.connect(
            self.clicked_pushButton_restore_sparklda_settings)
        self.pushButton_restore_ctm_settings.clicked.connect(
            self.clicked_pushButton_restore_ctm_settings)
        self.pushButton_restore_hierarchical_settings.clicked.connect(
            self.clicked_pushButton_restore_hierarchical_settings)
        self.pushButton_restore_tm_general_settings.clicked.connect(
            self.clicked_pushButton_restore_tm_general_settings)
        self.pushButton_restore_spark_settings.clicked.connect(
            self.clicked_pushButton_restore_spark_settings)
        self.pushButton_restore_preprocessing_settings.clicked.connect(
            self.clicked_pushButton_restore_preprocessing_settings)

    #####################################################################################
    # TASK MANAGER COMMUNICATION METHODS
    #####################################################################################
    def configure_tm(self):
        """
        Once proper project and parquet folders have been selected by the user, it instantiates a task manager object and its corresponding configuration functions are invoked according to whether the project selected already existed or was just created.After this, the selected folders are saved in the dictionary of recent folders and the menu buttons are unlocked so the user can proceed with the interaction with the GUI.
        """

        if self.project_folder and self.parquet_folder and self.wordlists_folder:
            # A ITMTTaskManagerGUI is instantiated and configured according to whether the selected project is a new or
            # an already utilized one
            self.tm = ITMTTaskManagerGUI(
                self.project_folder, self.parquet_folder, self.wordlists_folder)
            if len(os.listdir(self.project_folder)) == 0:
                print("A new project folder was selected. Proceeding with "
                      "its configuration...")
                self.tm.create()
                # self.tm.setup()
            else:
                print("An existing project folder was selected. Proceeding with "
                      "its loading...")
            self.tm.load()
            # Project and parquet folders are saved in the dict of recent folders to future user-gui interactions
            utils.save_recent(self.project_folder,
                              self.parquet_folder, self.wordlists_folder)
            self.init_user_interaction()

            return

    def load_data(self):
        """
        It loads the data and its associated metadata (local datasets available in the parquet folder, corpus for training and trained models) into the corresponding GUI's tables.
        """

        # Load datasets available in the parquet folder into "table_available_local_corpora"
        self.tm.listDownloaded(self)
        # Add checkboxes in the last column of "table_available_local_corpora" so the user can select from which of the datasets he wants to create a training corpus
        utils.add_checkboxes_to_table(self.table_available_local_corpora, 0)

        # Load available training corpus (if any) into "table_available_tr_datasets"
        self.tm.listTMCorpus(self)

        # Update the style of the tables in the corpus page
        utils.configure_table_header(Constants.CORPUS_TABLES, self)

        # Load available wordlists (if any) into "table_available_wordlists"
        self.tm.listAllWdLists(self)

        # Update the style of the tables in the wordlists page
        utils.configure_table_header(Constants.WORDLISTS_TABLES, self)

        # Fill settings table
        self.set_default_settings("all", False, False)

        # Load models
        self.tm.listAllTMmodels(self)

        self.train_model_subwindow = TrainModelWindow(
            tm=self.tm, thread_pool=self.thread_pool, stdout=self.stdout, stderr=self.stderr, training_corpus=None, preproc_settings=None)

        return

    def init_user_interaction(self):
        """
        Unlocks the clicking of the menu buttons so the user can proceed with the interaction with the GUI different from the selection of the project and parquet folders.
        """

        # Once project and parquet folders are properly selected, app's functionalities are enabled
        for menu_button in self.menu_buttons:
            menu_button.setEnabled(True)

        for extra_menu_button in self.extra_menu_buttons:
            extra_menu_button.setEnabled(True)

        # Make menu visible
        self.frame_menu_title.show()

        # Already available data is visualized in the corresponding tables
        self.load_data()

        return

    ############################################################################
    # HANDLERS
    ############################################################################
    def get_project_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_project_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the project folder. Once the project is selected, if a proper parquet and wordlists folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.project_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select an existing project or create a new one', self.home))
        self.lineEdit_current_project.setText(self.project_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_parquet_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_parquet_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the parquet folder. Once the parquet folder is selected, if a proper project and wordlists folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.parquet_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder with the parquet files', self.home))
        self.lineEdit_current_parquet.setText(self.parquet_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_wordlists_folder(self):
        """
        Method to control the clicking of the button "pushButton_open_wordlists_folder. When this button is clicked, the folder selector of the user's OS is open so the user can select the folder hosting the wordlists (stopwords, keywords, etc). Once the wordlists folder is selected, if a proper project and parquet folders were also already selected, the GUI's associated Task Manager object is configured.
        """

        self.wordlists_folder = pathlib.Path(
            QFileDialog.getExistingDirectory(
                self, 'Select the folder hosting the wordlists (stopwords, keywords, etc)', self.home))
        self.lineEdit_current_wordlist.setText(
            self.wordlists_folder.as_posix())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def get_folder_from_recent(self, recent_button):
        """
        Method to control the clicking of one of the recent buttons. If the recent button relates to a project folder, such a project folder is loaded directly as the selected project folder and updated in the corresponding line edit. Otherwise, the same applies but for the parquet folder. If both proper project and parquet folders have been selected, the GUI's associated Task Manager object is configured.
        """

        if "project" in recent_button.objectName():
            self.project_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_project.setText(recent_button.text())
        elif "parquet" in recent_button.objectName():
            self.parquet_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_parquet.setText(recent_button.text())
        elif "wordlists" in recent_button.objectName():
            self.wordlists_folder = pathlib.Path(recent_button.text())
            self.lineEdit_current_wordlist.setText(recent_button.text())

        # Create Task Manager object if possible
        self.configure_tm()

        return

    def clicked_change_menu_button(self, menu_button):
        """
        Method to control the selection of one of the buttons in the menu bar.
        """

        # Put unpressed color for the previous pressed menu button
        if self.previous_page_button:
            if self.previous_page_button.objectName() == "menu_button_1":
                self.previous_page_button.setStyleSheet(
                    Constants.HOME_BUTTON_UNSELECTED_STYLESHEET)
            else:
                self.previous_page_button.setStyleSheet(
                    Constants.OTHER_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_page_button = menu_button
        if self.previous_page_button.objectName() == "menu_button_1":
            self.previous_page_button.setStyleSheet(
                Constants.HOME_BUTTON_SELECTED_STYLESHEET)
        else:
            self.previous_page_button.setStyleSheet(
                Constants.OTHER_BUTTONS_SELECTED_STYLESHEET)
        return

    def clicked_change_corpus_button(self, corpus_button):
        """
        Method to control the selection of one of the buttons in the train bar.
        """

        # Put unpressed color for the previous pressed train button
        if self.previous_corpus_button:
            self.previous_corpus_button.setStyleSheet(
                Constants.TRAIN_BUTTONS_UNSELECTED_STYLESHEET)

        self.previous_corpus_button = corpus_button
        self.previous_corpus_button.setStyleSheet(
            Constants.TRAIN_BUTTONS_SELECTED_STYLESHEET)

        return

    # CORPUS FUNCTIONS
    def clicked_pushButton_generate_training_dataset(self):
        """Method for controlling the clicking of the 'pushButton_generate_training_dataset' Once this button is clicked, the datasets to be used for the creation of a new topic modeling corpus are collected by checking which of the checkboxes at the last column of the available local datasets are selected, and a list is created with its corresponding ids. Such a list is utilized for creating a "GenerateTMCorpus" subwindow, from which the user can specify his desired characteristics for the training corpus to be created. Once the former subwindow is closed, the training corpus generation is completed and based on the status returned by the task manager function in charge of the creation, an informative or warning message is shown to the user.
        """

        # Get ids of the datasets that are going to be used for the training corpus generation
        checked_list = []
        for i in range(self.table_available_local_corpora.rowCount()):
            item = self.table_available_local_corpora.item(
                i, 0)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                checked_list.append(i)
        
        if len(checked_list) == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.CREATE_TR_DST_NOT_SELECTED_MSG)
            return

        self.create_tm_corpus_subwindow = GenerateTMCorpus(
            checked_list, self.tm)
        self.create_tm_corpus_subwindow.exec()

        # Update data in table
        self.load_data()

        # Show information message about the TM corpus creation completion
        if self.create_tm_corpus_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.TM_CORPUS_MSG_STATUS_0)
        elif self.create_tm_corpus_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_1)
        elif self.create_tm_corpus_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_CORPUS_MSG_STATUS_2)

        return

    def clicked_delete_dataset(self):
        """It controls the clicking of the 'pushButton_delete_trdtst' button. When such a button is selected, it is first checked whether the user has chosen a corpus to be deleted. In the affirmative case, the TM function in charge of the deletion of training corpus is invoked.
        """

        # Get selected dataset for deletion
        r = self.table_available_tr_datasets.currentRow()
        print(self.table_available_tr_datasets.item(
            r, 0).text())

        # If no training corpus is selected for deletion before clicking the 'pushButton_delete_trdtst' button,
        # a warning message is shown to the user
        if r is None:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.TM_DELETE_NO_CORPUS_MSG)
            return
        # Get name of the corpus to be deleted
        corpus_to_delete = self.table_available_tr_datasets.item(
            r, 0).text()

        # Delete corpus by invoking the corresponding task manager function
        self.tm.deleteTMCorpus(corpus_to_delete, self)

        # Reload the listing of the training corpora
        self.tm.listTMCorpus(self)

        return

    def clicked_pushButton_create_wordlist(self):
        """It controls the clicking of the 'pushButton_create_wordlist' button. When such a button is clicked, a message informing about the format for each wordlist type is shown to the user. Immediately after, a new window appears from which the user can insert the wordlist characteristics. Once this window is closed and the wordlist creation is completed, an informative message is shown specifying whether the creation was successful or not.
        """

        # Show instruction for creation of wordlists
        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.MSG_INSTRUCTIONS_NEW_WORDLIST)

        # Invoke window
        self.create_stopwords_list_subwindow = CreateSwLstWindow(self.tm)
        self.create_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus creation completion
        if self.create_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.WORDLIST_CREATION_MSG_STATUS_0)
        elif self.create_stopwords_list_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_CREATION_MSG_STATUS_1)
        elif self.create_stopwords_list_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_CREATION_MSG_STATUS_2)

        return

    def clicked_pushButton_edit_wordlist(self):
        """It controls the clicking of the 'pushButton_edit_wordlist' button. When such a button is clicked, a new window appears from which the user can edit the selected wordlist. Once this window is closed and the wordlist edition is completed, an informative message is shown specifying whether the edition was successful or not.
        """

        # Get wordlist selected for edition
        r = self.table_available_wordlists.currentRow()

        # If no wordlist for edition is selected  before clicking the 'pushButton_edit_wordlist' button, a warning message is shown to the user
        if not r:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.EDIT_WORDLIST_NOT_SELECTED_MSG)
            return

        wlst_to_edit = self.table_available_wordlists.item(r, 0).text()
        wdList_info = self.tm.get_wdlist_info(wlst_to_edit)
        self.edit_stopwords_list_subwindow = EditSwLstWindow(
            self.tm, wdList_info)
        self.edit_stopwords_list_subwindow.exec()

        # Update data in table
        self.tm.listAllWdLists(self)

        # Show information message about the TM corpus edition completion
        if self.edit_stopwords_list_subwindow.status == 0:
            QMessageBox.warning(self, Constants.SMOOTH_SPOON_MSG,
                                Constants.WORDLIST_EDITION_MSG_STATUS_0)
        elif self.edit_stopwords_list_subwindow.status == 1:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_EDITION_MSG_STATUS_1)
        elif self.edit_stopwords_list_subwindow.status == 2:
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WORDLIST_EDITION_MSG_STATUS_2)

        return

    def clicked_pushButton_delete_wordlist(self):
        """It controls the clicking of the 'pushButton_delete_wordlist' button. When such a button is selected, it is first checked whether the user has chosen a wordlist to be deleted. In the affirmative case, the TM function in charge of the deletion of a wordlist is invoked.
        """

        # Get wordlist selected for deletion
        r = self.table_available_wordlists.currentRow()

        # If no training corpus is selected for deletion before clicking the 'pushButton_delete_wordlist' button, a warning message is shown to the user
        if not r:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.DELETE_WORDLIST_NOT_SELECTED_MSG)
            return

        wlst_to_delete = self.table_available_wordlists.item(r, 0).text()
        self.tm.DelWdList(wlst_to_delete, self)

        # Update data in wordlists table
        self.tm.listAllWdLists(self)

        return

    def show_wordlist_description(self):

        # Get wordlist selected for deletion
        r = self.table_available_wordlists.currentRow()

        wlst = self.table_available_wordlists.item(r, 0).text()

        wlst_dict = self.tm.get_wdlist_info(wlst)

        self.textEdit_wordlist_content.setPlainText(
            ', '.join([el for el in wlst_dict['wordlist']]))

        return

    def clicked_train_dataset(self):

        # Get training dataset
        r = self.table_available_tr_datasets.currentRow()

        # If no training dataset is selected for before clicking the 'train_dataset' button, a warning message is shown to the user
        if r == -1:
            QMessageBox.warning(
                self, Constants.SMOOTH_SPOON_MSG, Constants.WARNING_NO_TR_CORPUS)
            return

        training_corpus = self.table_available_tr_datasets.item(r, 0).text()

        # Get preprocessing settings
        self.preprocessing_subwindow = PreprocessingWindow(tm=self.tm)
        self.preprocessing_subwindow.exec()

        self.train_model_subwindow.TrDts_name = training_corpus
        self.train_model_subwindow.preproc_settings = self.preprocessing_subwindow.preproc_settings
        self.train_model_subwindow.hierarchy_level = 0
        self.train_model_subwindow.initialize_hierarchical_level_settings()
        self.train_model_subwindow.exec()

        # @TODO: Reload models

        return

    # MODELS FUNCTIONS
    def clicked_treeView_trained_models(self):
        """Method to control the clicking of an item within the QTreeWidget 'treeView_trained_models' in the 'Models' tab. At the time one of the items in the QTreeWidget is selected, the information of the model associated with the clicked item, as well as its topics' chemical description is shown in the tables 'table_available_trained_models_desc' and 'tableWidget_trained_models_topics', respectively.
        """

        if self.treeView_trained_models.currentItem() is None or \
                self.treeView_trained_models.currentItem().text(0).lower().startswith("models"):
            return
        else:
            model_selected = self.treeView_trained_models.currentItem().text(0)
            self.tm.listTMmodel(self, model_selected)

            # Show topics in table
            # self.tableWidget_trained_models_topics.clearContents()
            # self.tableWidget_trained_models_topics.setRowCount(
            #     int(model.num_topics))
            # self.tableWidget_trained_models_topics.setColumnCount(2)

            # list_description = []
            # for i in np.arange(0, len(model.topics_models), 1):
            #     if str(type(model.topics_models[i])) == "<class 'src.htms.topic.Topic'>":
            #         description = ' '.join(
            #             str(x) for x in model.topics_models[i].description)
            #         list_description.append(description)
            # for i in np.arange(0, len(list_description), 1):
            #     item_topic_nr = QtWidgets.QTableWidgetItem(str(i))
            #     self.tableWidget_trained_models_topics.setItem(
            #         i, 0, item_topic_nr)
            #     item_topic_description = QtWidgets.QTableWidgetItem(
            #         str(list_description[i]))
            #     self.tableWidget_trained_models_topics.setItem(
            #         i, 1, item_topic_description)

            # Show PyLDAvis
            # if self.web:
            #     self.web.setParent(None)
            # self.web = QWebEngineView()
            # # self.web.setZoomFactor(0.25)
            # url = QUrl.fromLocalFile(pathlib.Path(
            #     model.model_path, "pyLDAvis.html").as_posix())
            # self.web.load(url)
            # self.layout_plot_pyldavis.addWidget(self.web)
            # self.web.show()

        return

    def clicked_pushButton_models_pyldavis(self):
        """Method to control the change within the Models' page to the PyLDAvis view. 
        """

        self.models_tabs.setCurrentWidget(self.page_models_pyldavis)

        return

    def clicked_pushButton_return_pyldavis(self):
        """Method to control the change from the PyLDAvis view back to the Models' page main view. 
        """

        self.models_tabs.setCurrentWidget(self.page_models_main)

        return

    def clicked_pushButton_train_submodel(self):
        """Method to control the clicking of the button 'pushButton_train_submodel', which is in charge of starting the training of a submodel.
        """

        return

    def clicked_pushButton_delete_model(self):
        """Method to control the clicking of the button 'pushButton_train_submodel', which is in charge of starting the deletion of a model.
        """

        return

    # SETTINGS FUNCTIONS
    def clicked_change_settings_button(self, settings_button):
        """
        Method to control the selection of one of the topic modeling settings button.
        """

        # Put unpressed color for the previous pressed settings button
        if self.previous_settings_button:
            self.previous_settings_button.setStyleSheet(
                Constants.SETTINGS_UNSELECTED_STYLESHEET)

        if self.previous_settings_button.objectName() == "settings_button_5":
            for tm_settings_button in self.tm_settings_buttons:
                tm_settings_button.hide()
                tm_settings_button.setStyleSheet(
                    Constants.SETTINGS_SUBBUTTON_UNSELECTED_STYLESHEET)

        self.previous_settings_button = settings_button
        self.previous_settings_button.setStyleSheet(
            Constants.SETTINGS_SELECTED_STYLESHEET)

        return

    def clicked_change_tm_settings_button(self, tm_settings_button):
        """
        Method to control the selection of one of the topic modeling settings button.
        """

        # Put unpressed color for the previous pressed settings button
        if self.previous_tm_settings_button:
            self.previous_tm_settings_button.setStyleSheet(
                Constants.SETTINGS_SUBBUTTON_UNSELECTED_STYLESHEET)

        self.previous_tm_settings_button = tm_settings_button
        self.previous_tm_settings_button.setStyleSheet(
            Constants.SETTINGS_SUBBUTTON_SELECTED_STYLESHEET)

        self.settings_button_5.setStyleSheet(
            Constants.SETTINGS_SELECTED_STYLESHEET)

        return

    def select_tm_settings_suboption(self):
        """
        Method for showing the Topic Modeling settings subbutons when the button 'settings_button_5' is clicked.
        """

        for tm_settings_button in self.tm_settings_buttons:
            tm_settings_button.show()

        return

    def set_default_settings(self, option, save, msg):
        """
        Method for reading settings from the TM's configuration file and writing them in the correspondng view within the Settings' page.
        """

        # Get config object
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config_dft)

        if option == "gui":
            # GUI settings
            self.lineEdit_font_size.setText(str(14))
            self.fontComboBox.setCurrentText("Arial")
            if save:
                self.clicked_pushButton_apply_changes_gui_settings()
        elif option == "log":
            # LOG SETTINGS
            self.lineEdit_file_name.setText(
                str(cf.get('logformat', 'filename')))
            self.lineEdit_date_format.setText(
                str(cf.get('logformat', 'datefmt')))
            self.lineEdit_file_format.setText(
                str(cf.get('logformat', 'file_format')))
            self.lineEdit_file_level.setText(
                str(cf.get('logformat', 'file_level')))
            self.lineEdit_cons_format.setText(
                str(cf.get('logformat', 'cons_format')))
            self.lineEdit_cons_level.setText(
                str(cf.get('logformat', 'cons_level')))
            if save:
                self.clicked_pushButton_apply_changes_log_settings()
        elif option == "mallet":
            # MALLET SETTINGS
            self.lineEdit_settings_mallet_path.setText(
                str(cf.get('MalletTM', 'mallet_path')))
            self.lineEdit_settings_regexp.setText(
                str(cf.get('MalletTM', 'token_regexp')))
            self.lineEdit_settings_alpha.setText(
                str(cf.get('MalletTM', 'alpha')))
            self.lineEdit_settings_optimize_interval.setText(
                str(cf.get('MalletTM', 'optimize_interval')))
            self.lineEdit_settings_nr_threads.setText(
                str(cf.get('MalletTM', 'num_threads')))
            self.lineEdit_settings_nr_iter.setText(
                str(cf.get('MalletTM', 'num_iterations')))
            self.lineEdit_settings_doc_top_thr.setText(
                str(cf.get('MalletTM', 'doc_topic_thr')))
            self.lineEdit_settings_thetas_thr.setText(
                str(cf.get('MalletTM', 'thetas_thr')))
            self.lineEdit_settings_infer_iter.setText(
                str(cf.get('MalletTM', 'num_iterations_inf')))
            if save:
                self.clicked_pushButton_apply_changes_mallet_settings()
        elif option == "prodlda":
            # PRODLDA
            self.lineEdit_settings_model_type_prod.setText(
                str(cf.get('ProdLDA', 'model_type')))
            self.lineEdit_settings_hidden_prod.setText(
                str(cf.get('ProdLDA', 'hidden_sizes')))
            self.lineEdit_settings_activation_prod.setText(
                str(cf.get('ProdLDA', 'activation')))
            self.lineEdit_settings_dropout_prod.setText(
                str(cf.get('ProdLDA', 'dropout')))
            self.lineEdit_settings_learn_priors_prod.setText(
                str(cf.get('ProdLDA', 'learn_priors')))
            self.lineEdit_settings_lr_prod.setText(
                str(cf.get('ProdLDA', 'lr')))
            self.lineEdit_settings_momentum_prod.setText(
                str(cf.get('ProdLDA', 'momentum')))
            self.lineEdit_settings_solver_prod.setText(
                str(cf.get('ProdLDA', 'solver')))
            self.lineEdit_settings_nr_epochs_prod.setText(
                str(cf.get('ProdLDA', 'num_epochs')))
            self.lineEdit_settings_plateau_prod.setText(
                str(cf.get('ProdLDA', 'reduce_on_plateau')))
            self.lineEdit_settings_batch_size_prod.setText(
                str(cf.get('ProdLDA', 'batch_size')))
            self.lineEdit_settings_topic_prior_mean_prod.setText(
                str(cf.get('ProdLDA', 'topic_prior_mean')))
            self.lineEdit_settings_topic_prior_var_prod.setText(
                str(cf.get('ProdLDA', 'topic_prior_variance')))
            self.lineEdit_settings_nr_samples_prod.setText(
                str(cf.get('ProdLDA', 'num_samples')))
            self.lineEdit_settings_workers_prod.setText(
                str(cf.get('ProdLDA', 'num_data_loader_workers')))
            self.lineEdit_settings_thetas_thr_prod.setText(
                str(cf.get('ProdLDA', 'thetas_thr')))
            if save:
                self.clicked_pushButton_apply_changes_prodlda_settings()
        elif option == "ctm":
            # CTM
            self.lineEdit_settings_under_ctm_type.setText(
                str(cf.get('CTM', 'model_type')))
            self.lineEdit_settings_model_type_ctm.setText(
                str(cf.get('CTM', 'ctm_model_type')))
            self.lineEdit_settings_hidden_ctm.setText(
                str(cf.get('CTM', 'hidden_sizes')))
            self.lineEdit_settings_activation_ctm.setText(
                str(cf.get('CTM', 'activation')))
            self.lineEdit_settings_dropout_ctm.setText(
                str(cf.get('CTM', 'dropout')))
            self.lineEdit_settings_priors_ctm.setText(
                str(cf.get('CTM', 'learn_priors')))
            self.lineEdit_settings_lr_ctm.setText(
                str(cf.get('CTM', 'batch_size')))
            self.lineEdit_settings_momentum_ctm.setText(
                str(cf.get('CTM', 'lr')))
            self.lineEdit_settings_solver_ctm.setText(
                str(cf.get('CTM', 'momentum')))
            self.lineEdit_settings_nr_epochs_ctm.setText(
                str(cf.get('CTM', 'solver')))
            self.lineEdit_settings_plateau_ctm.setText(
                str(cf.get('CTM', 'num_epochs')))
            self.lineEdit_settings_nr_samples_ctm.setText(
                str(cf.get('CTM', 'num_samples')))
            self.lineEdit_settings_plateau_ctm.setText(
                str(cf.get('CTM', 'reduce_on_plateau')))
            self.lineEdit_settings_topic_prior_mean_ctm.setText(
                str(cf.get('CTM', 'topic_prior_mean')))
            self.lineEdit_settings_topic_prior_std_ctm.setText(
                str(cf.get('CTM', 'topic_prior_variance')))
            self.lineEdit_settings_nr_workers_ctm.setText(
                str(cf.get('CTM', 'num_data_loader_workers')))
            self.lineEdit_settings_label_size_ctm.setText(
                str(cf.get('CTM', 'label_size')))
            self.lineEdit_settings_loss_weights_ctm.setText(
                str(cf.get('CTM', 'loss_weights')))
            self.lineEdit_settings_thetas_thr_ctm.setText(
                str(cf.get('CTM', 'thetas_thr')))
            self.lineEdit_settings_sbert_model_ctm.setText(
                str(cf.get('CTM', 'sbert_model_to_load')))
            if save:
                self.clicked_pushButton_apply_changes_ctm_settings()
        elif option == "tm_general":
            # GENERAL TM SETTINGS
            self.lineEdit_settings_nr_topics.setText(
                str(cf.get('TM', 'ntopics')))
            self.lineEdit_settings_nr_words.setText(
                str(cf.get('TMedit', 'n_palabras')))
            self.lineEdit_settings_round_size.setText(
                str(cf.get('TMedit', 'round_size')))
            self.lineEdit_settings_netl_workers.setText(
                str(cf.get('TMedit', 'NETLworkers')))
            self.lineEdit_settings_ldavis_nr_docs.setText(
                str(cf.get('TMedit', 'LDAvis_ndocs')))
            self.lineEdit_settings_ldavis_nr_jobs.setText(
                str(cf.get('TMedit', 'LDAvis_njobs')))
            if save:
                self.clicked_pushButton_apply_changes_tm_general_settings()
        elif option == "spark":
            # SPARK
            self.lineEdit_settings_spark_available.setText(
                str(cf.get('Spark', 'spark_available')))
            self.lineEdit_settings_script_spark.setText(
                str(cf.get('Spark', 'script_spark')))
            self.lineEdit_settings_token_spark.setText(
                str(cf.get('Spark', 'token_spark')))
            if save:
                self.clicked_pushButton_apply_spark_settings()
        elif option == "preproc":
            # PREPROCESSING
            self.lineEdit_settings_min_lemas.setText(
                str(cf.get('Preproc', 'min_lemas')))
            self.lineEdit_settings_nr_below.setText(
                str(cf.get('Preproc', 'no_below')))
            self.lineEdit_settings_nr_above.setText(
                str(cf.get('Preproc', 'no_above')))
            self.lineEdit_settings_keep_n.setText(
                str(cf.get('Preproc', 'keep_n')))
            if save:
                self.clicked_pushButton_apply_preprocessing_settings()
        elif option == "all":
            for op in Constants.SETTINGS_OPTIONS:
                self.set_default_settings(op, False, False)
            if msg:
                QMessageBox.information(
                    self, Constants.SMOOTH_SPOON_MSG, Constants.RESTORE_DFT_ALL_SETTINGS)
        if msg:
            id_msg = Constants.SETTINGS_OPTIONS.index(option)
            QMessageBox.information(
                self, Constants.SMOOTH_SPOON_MSG, Constants.RESTORE_DFT_MSGS[id_msg])

        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)
        
        return

    def clicked_pushButton_apply_changes_gui_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_gui_settings' that controls the actualization of the GUI' settings.
        """

        font_size = self.lineEdit_font_size.text()
        font_type = self.fontComboBox.currentText()
        font_size_type = font_size + ' "' + font_type + '" '
        stylesheet = "QWidget { font: %s}" % font_size_type
        self.centralwidget.setStyleSheet(stylesheet)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_GUI_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)

        return

    def clicked_pushButton_apply_changes_log_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_log_settings' that controls the actualization of the Topic Modeler's log settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('logformat', 'filename', str(self.lineEdit_file_name.text()))
        cf.set('logformat', 'datefmt', str(self.lineEdit_date_format.text()))
        cf.set('logformat', 'file_format', str(
            self.lineEdit_file_format.text()))
        cf.set('logformat', 'file_level', str(self.lineEdit_file_level.text()))
        cf.set('logformat', 'cons_format', str(
            self.lineEdit_cons_format.text()))
        cf.set('logformat', 'cons_level', str(self.lineEdit_cons_level.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_LOG_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)

        return

    def clicked_pushButton_apply_changes_mallet_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_mallet_settings' that controls the actualization of the Topic Modeler's Mallet settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('MalletTM', 'mallet_path', str(
            self.lineEdit_settings_mallet_path.text()))
        cf.set('MalletTM', 'token_regexp', str(
            self.lineEdit_settings_regexp.text()))
        cf.set('MalletTM', 'alpha', str(self.lineEdit_settings_alpha.text()))
        cf.set('MalletTM', 'optimize_interval', str(
            self.lineEdit_settings_optimize_interval.text()))
        cf.set('MalletTM', 'num_threads', str(
            self.lineEdit_settings_nr_threads.text()))
        cf.set('MalletTM', 'num_iterations', str(
            self.lineEdit_settings_nr_iter.text()))
        cf.set('MalletTM', 'doc_topic_thr', str(
            self.lineEdit_settings_doc_top_thr.text()))
        cf.set('MalletTM', 'thetas_thr', str(
            self.lineEdit_settings_thetas_thr.text()))
        cf.set('MalletTM', 'num_iterations_inf', str(
            self.lineEdit_settings_infer_iter.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_MALLET_SETTINGS)

        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)
        self.train_model_subwindow.get_lda_mallet_params()

        return

    def clicked_pushButton_apply_changes_prodlda_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_prodlda_settings' that controls the actualization of the Topic Modeler's ProdLDA settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('ProdLDA', 'model_type', str(
            self.lineEdit_settings_model_type_prod.text()))
        cf.set('ProdLDA', 'hidden_sizes', str(
            self.lineEdit_settings_hidden_prod.text()))
        cf.set('ProdLDA', 'activation', str(
            self.lineEdit_settings_activation_prod.text()))
        cf.set('ProdLDA', 'dropout', str(
            self.lineEdit_settings_dropout_prod.text()))
        cf.set('ProdLDA', 'learn_priors', str(
            self.lineEdit_settings_learn_priors_prod.text()))
        cf.set('ProdLDA', 'lr', str(self.lineEdit_settings_lr_prod.text()))
        cf.set('ProdLDA', 'momentum', str(
            self.lineEdit_settings_momentum_prod.text()))
        cf.set('ProdLDA', 'solver', str(
            self.lineEdit_settings_nr_epochs_prod.text()))
        cf.set('ProdLDA', 'num_epochs', str(
            self.lineEdit_settings_nr_comp_prod.text()))
        cf.set('ProdLDA', 'reduce_on_plateau', str(
            self.lineEdit_settings_plateau_prod.text()))
        cf.set('ProdLDA', 'batch_size', str(
            self.lineEdit_settings_batch_size_prod.text()))
        cf.set('ProdLDA', 'topic_prior_mean', str(
            self.lineEdit_settings_topic_prior_mean_prod.text()))
        cf.set('ProdLDA', 'topic_prior_variance', str(
            self.lineEdit_settings_topic_prior_var_prod.text()))
        cf.set('ProdLDA', 'num_samples', str(
            self.lineEdit_settings_nr_samples_prod.text()))
        cf.set('ProdLDA', 'num_data_loader_workers', str(
            self.lineEdit_settings_workers_prod.text()))
        cf.set('ProdLDA', 'thetas_thr', str(
            self.lineEdit_settings_thetas_thr_prod.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_PRODLDA_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)
        self.train_model_subwindow.get_prodlda_params()

        return

    def clicked_pushButton_apply_changes_sparklda_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_sparklda_settings' that controls the actualization of the Topic Modeler's SparkLDA settings.
        """

        #@TODO

        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)
        self.train_model_subwindow.get_sparklda_params()

        return

    def clicked_pushButton_apply_changes_ctm_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_ctm_settings' that controls the actualization of the Topic Modeler's CTM settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('CTM', 'model_type', str(
            self.lineEdit_settings_under_ctm_type.text()))
        cf.set('CTM', 'ctm_model_type', str(
            self.lineEdit_settings_model_type_ctm.text()))
        cf.set('CTM', 'hidden_sizes', str(
            self.lineEdit_settings_hidden_ctm.text()))
        cf.set('CTM', 'activation', str(
            self.lineEdit_settings_activation_ctm.text()))
        cf.set('CTM', 'dropout', str(self.lineEdit_settings_dropout_ctm.text()))
        cf.set('CTM', 'learn_priors', str(
            self.lineEdit_settings_priors_ctm.text()))
        cf.set('CTM', 'batch_size', str(
            self.lineEdit_settings_batch_size_ctm.text()))
        cf.set('CTM', 'lr', str(self.lineEdit_settings_lr_ctm.text()))
        cf.set('CTM', 'momentum', str(
            self.lineEdit_settings_momentum_ctm.text()))
        cf.set('CTM', 'solver', str(self.lineEdit_settings_solver_ctm.text()))
        cf.set('CTM', 'num_epochs', str(
            self.lineEdit_settings_nr_epochs_ctm.text()))
        cf.set('CTM', 'num_samples', str(
            self.lineEdit_settings_nr_samples_ctm.text()))
        cf.set('CTM', 'reduce_on_plateau', str(
            self.lineEdit_settings_plateau_ctm.text()))
        cf.set('CTM', 'topic_prior_mean', str(
            self.lineEdit_settings_topic_prior_mean_ctm.text()))
        cf.set('CTM', 'topic_prior_variance', str(
            self.lineEdit_settings_topic_prior_std_ctm.text()))
        cf.set('CTM', 'num_data_loader_workers', str(
            self.lineEdit_settings_nr_workers_ctm.text()))
        cf.set('CTM', 'label_size', str(
            self.lineEdit_settings_label_size_ctm.text()))
        cf.set('CTM', 'loss_weights', str(
            self.lineEdit_settings_loss_weights_ctm.text()))
        cf.set('CTM', 'thetas_thr', str(
            self.lineEdit_settings_thetas_thr_ctm.text()))
        cf.set('CTM', 'sbert_model_to_load', str(
            self.lineEdit_settings_sbert_model_ctm.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_CTM_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)
        self.train_model_subwindow.get_ctm_params()

        return

    def clicked_pushButton_apply_changes_hierarchical_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_hierarchical_settings' that controls the actualization of the Topic Modeler's hierarchical settings.
        """

        return

    def clicked_pushButton_apply_changes_tm_general_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_changes_tm_general_settings' that controls the actualization of the Topic Modeler's general settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('TMedit', 'n_palabras', str(
            self.lineEdit_settings_nr_words.text()))
        cf.set('TMedit', 'round_size', str(
            self.lineEdit_settings_round_size.text()))
        cf.set('TMedit', 'NETLworkers', str(
            self.lineEdit_settings_netl_workers.text()))
        cf.set('TMedit', 'LDAvis_ndocs', str(
            self.lineEdit_settings_ldavis_nr_docs.text()))
        cf.set('TMedit', 'LDAvis_njobs', str(
            self.lineEdit_settings_ldavis_nr_jobs.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_TM_GENERAL_SETTINGS)

        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)

        return

    def clicked_pushButton_apply_spark_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_spark_settings' that controls the actualization of the Topic Modeler's Spark settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('Spark', 'spark_available', str(
            self.lineEdit_settings_spark_available.text()))
        cf.set('Spark', 'script_spark', str(
            self.lineEdit_settings_script_spark.text()))
        cf.set('Spark', 'token_spark', str(
            self.lineEdit_settings_token_spark.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_SPARK_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)

        return

    def clicked_pushButton_apply_preprocessing_settings(self):
        """
        Method for controling the clicking of the button 'pushButton_apply_preprocessing_settings' that controls the actualization of the Topic Modeler's preprocessing settings.
        """

        # Save user's configuration into config file
        cf = configparser.ConfigParser()
        cf.read(self.tm.p2config)

        cf.set('Preproc', 'min_lemas', str(
            self.lineEdit_settings_min_lemas.text()))
        cf.set('Preproc', 'no_below', str(
            self.lineEdit_settings_nr_below.text()))
        cf.set('Preproc', 'no_above', str(
            self.lineEdit_settings_nr_above.text()))
        cf.set('Preproc', 'keep_n', str(
            self.lineEdit_settings_keep_n.text()))

        with open(self.tm.p2config, 'w') as configfile:
            cf.write(configfile)

        QMessageBox.information(
            self, Constants.SMOOTH_SPOON_MSG, Constants.UPDATE_PREPROC_SETTINGS)
        
        # Reload config in task manager
        self.tm.cf = configparser.ConfigParser()
        self.tm.cf.read(self.tm.p2config)

        return

    def clicked_pushButton_restore_gui_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_tm_gui_settings' that manages the setting of the GUI settings' default values.
        """

        self.set_default_settings("gui", True, True)

        return

    def clicked_pushButton_restore_log_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_log_settings' that manages the setting of the Topic Modeler's log settings' default values.
        """

        self.set_default_settings("log", True, True)

        return

    def clicked_pushButton_restore_mallet_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_mallet_settings' that manages the setting of the Topic Modeler's Mallet settings' default values.
        """

        self.set_default_settings("mallet", True, True)

        return

    def clicked_pushButton_restore_prodlda_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_tm_prodlda_settings' that manages the setting of the Topic Modeler's ProdLDA settings' default values.
        """

        self.set_default_settings("prodlda", True, True)

        return

    def clicked_pushButton_restore_sparklda_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_tm_sparklda_settings' that manages the setting of the Topic Modeler's SparkLDA settings' default values.
        """

        return

    def clicked_pushButton_restore_ctm_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_ctm_settings' that manages the setting of the Topic Modeler's CTM settings' default values.
        """

        self.set_default_settings("ctm", True, True)

        return

    def clicked_pushButton_restore_hierarchical_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_hierarchical_settings' that manages the setting of the Topic Modeler's hierarchical settings' default values.
        """

        return

    def clicked_pushButton_restore_tm_general_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_tm_general_settings' that manages the setting of the Topic Modeler's general settings' default values.
        """

        self.set_default_settings("tm_general", True, True)

        return

    def clicked_pushButton_restore_spark_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_spark_settings' that manages the setting of the Topic Modeler's Spark settings' default values.
        """

        self.set_default_settings("spark", True, True)

        return

    def clicked_pushButton_restore_preprocessing_settings(self):
        """
        Method for controlling the clicking of the button 'pushButton_restore_preprocessing_settings' that manages the setting of the Topic Modeler's preprocessing settings' default values.
        """

        self.set_default_settings("preproc", True, True)

        return
